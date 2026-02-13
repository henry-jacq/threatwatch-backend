import logging
import os
import time
import docker
from docker.errors import APIError, NotFound

from app.core.live.lab_state import (
    ATTACK_STATE,
    TRAFFIC_STATE,
    LAB_CONTAINERS,
    SUPPORTED_ATTACKS,
    SUPPORTED_TRAFFIC,
)

logger = logging.getLogger(__name__)

_client = None
ATTACK_TARGET_HOST = os.getenv("LAB_TARGET_HOST", "victim")

def _start_bg(pid_path: str, cmd: str) -> bool:
    # Start cmd in a new session so we can kill its whole process group by PID.
    wrapped = (
        "sh -lc \""
        f"rm -f {pid_path}; "
        f"setsid sh -lc '{cmd}' >/dev/null 2>&1 & "
        f"echo $! > {pid_path}"
        "\""
    )
    return _exec_attacker(wrapped, detach=True)


def _stop_bg(pid_path: str) -> bool:
    wrapped = (
        "sh -lc '"
        f"if [ -f {pid_path} ]; then "
        f"pid=$(cat {pid_path}); "
        "kill -TERM -$pid >/dev/null 2>&1 || true; "
        "sleep 0.2; "
        "kill -KILL -$pid >/dev/null 2>&1 || true; "
        f"rm -f {pid_path}; "
        "fi; "
        "true'"
    )
    return _exec_attacker(wrapped, detach=False)


def _init_client():
    global _client

    if _client:
        return _client

    try:
        _client = docker.DockerClient(
            base_url="unix:///var/run/docker.sock",
            version="auto",
        )
        _client.ping()
        logger.info("Docker connected successfully")
        return _client
    except Exception as exc:
        logger.error("Docker unavailable: %s", exc)
        _client = None
        return None


def _get_container(name: str):
    client = _init_client()
    if not client:
        return None

    try:
        return client.containers.get(name)
    except NotFound:
        return None
    except Exception as exc:
        logger.error("Failed to fetch container '%s': %s", name, exc)
        return None


def container_running(name: str) -> bool:
    container = _get_container(name)
    if not container:
        return False

    container.reload()
    return container.status == "running"


def process_running(container_name: str, pattern: str) -> bool:
    """
    Best-effort check that a process is running in the container.
    Used for 'agent' when agent runs inside victim container.
    """
    container = _get_container(container_name)
    if not container:
        return False

    # Avoid dependencies on procps/pgrep: scan /proc cmdlines via python.
    try:
        code = (
            "python - <<'PY'\n"
            "import os\n"
            f"needle = {pattern!r}\n"
            "for pid in os.listdir('/proc'):\n"
            "  if not pid.isdigit():\n"
            "    continue\n"
            "  try:\n"
            "    with open(f'/proc/{pid}/cmdline','rb') as f:\n"
            "      cmd = f.read().replace(b'\\x00', b' ')\n"
            "    if needle.encode() in cmd:\n"
            "      raise SystemExit(0)\n"
            "  except FileNotFoundError:\n"
            "    pass\n"
            "  except PermissionError:\n"
            "    pass\n"
            "raise SystemExit(1)\n"
            "PY"
        )
        r = container.exec_run(f"sh -lc \"{code}\"", detach=False)
        exit_code = getattr(r, "exit_code", None)
        if exit_code is None and isinstance(r, tuple):
            exit_code = r[0]
        return int(exit_code) == 0
    except Exception:
        return False


def _exec_attacker(command: str, detach: bool = True) -> bool:
    attacker_name = LAB_CONTAINERS["attacker"]
    container = _get_container(attacker_name)

    if not container:
        logger.warning("Cannot run attack command. Missing container: %s", attacker_name)
        return False

    if container.status != "running":
        logger.warning("Cannot run attack command. Container not running: %s", attacker_name)
        return False

    try:
        container.exec_run(command, detach=detach)
        return True
    except APIError as exc:
        logger.error("Docker exec failed: %s", exc)
        return False


def start_attack(attack_type: str = "syn") -> bool:
    attack_type = attack_type.lower().strip()

    if attack_type not in SUPPORTED_ATTACKS:
        logger.warning("Unsupported attack type requested: %s", attack_type)
        return False

    stop_attack()

    attack_cmd = {
        "syn": f"hping3 -S --faster -p 80 {ATTACK_TARGET_HOST}",
        "udp": f"hping3 --udp --faster -p 53 {ATTACK_TARGET_HOST}",
        "http": f"while true; do curl -m 1 -s http://{ATTACK_TARGET_HOST} >/dev/null; done",
        "random": (
            "while true; do "
            "n=$(( $(date +%s%N) % 4 )); "
            f"if [ \"$n\" -eq 0 ]; then ping -c 1 -W 1 {ATTACK_TARGET_HOST} >/dev/null 2>&1; "
            f"elif [ \"$n\" -eq 1 ]; then curl -m 1 -s http://{ATTACK_TARGET_HOST} >/dev/null 2>&1; "
            f"elif [ \"$n\" -eq 2 ]; then hping3 --udp -c 20 -i u3000 -p 53 {ATTACK_TARGET_HOST} >/dev/null 2>&1; "
            f"else hping3 -S -c 20 -i u3000 -p 80 {ATTACK_TARGET_HOST} >/dev/null 2>&1; fi; "
            "sleep 0.2; done"
        ),
    }[attack_type]

    started = _start_bg("/tmp/threatwatch_attack.pid", attack_cmd)

    if started:
        ATTACK_STATE["running"] = True
        ATTACK_STATE["type"] = attack_type
        ATTACK_STATE["started_at"] = time.time()
        logger.warning("Started %s attack", attack_type)

    return started


def stop_attack() -> bool:
    stopped = _stop_bg("/tmp/threatwatch_attack.pid")

    ATTACK_STATE["running"] = False
    ATTACK_STATE["type"] = None
    ATTACK_STATE["started_at"] = None

    return stopped


def start_traffic(traffic_type: str = "mixed") -> bool:
    traffic_type = traffic_type.lower().strip()

    if traffic_type not in SUPPORTED_TRAFFIC:
        logger.warning("Unsupported traffic type requested: %s", traffic_type)
        return False

    stop_traffic()

    traffic_cmd = {
        "http": f"while true; do curl -m 1 -s http://{ATTACK_TARGET_HOST} >/dev/null 2>&1; sleep 0.1; done",
        "ping": f"while true; do ping -c 1 -W 1 {ATTACK_TARGET_HOST} >/dev/null 2>&1; sleep 0.2; done",
        "mixed": (
            "while true; do "
            "n=$(( $(date +%s%N) % 2 )); "
            f"if [ \"$n\" -eq 0 ]; then curl -m 1 -s http://{ATTACK_TARGET_HOST} >/dev/null 2>&1; "
            f"else ping -c 1 -W 1 {ATTACK_TARGET_HOST} >/dev/null 2>&1; fi; "
            "sleep 0.2; done"
        ),
    }[traffic_type]

    started = _start_bg("/tmp/threatwatch_traffic.pid", traffic_cmd)

    if started:
        TRAFFIC_STATE["running"] = True
        TRAFFIC_STATE["type"] = traffic_type
        TRAFFIC_STATE["started_at"] = time.time()
        logger.warning("Started normal traffic (%s)", traffic_type)

    return started


def stop_traffic() -> bool:
    stopped = _stop_bg("/tmp/threatwatch_traffic.pid")
    TRAFFIC_STATE["running"] = False
    TRAFFIC_STATE["type"] = None
    TRAFFIC_STATE["started_at"] = None
    return stopped
