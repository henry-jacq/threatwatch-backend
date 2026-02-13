import logging
import os
import shlex
import time
import docker
from docker.errors import APIError, NotFound

from app.core.live.lab_state import (
    ATTACK_STATE,
    ATTACKER_STATE,
    TRAFFIC_STATE,
    LAB_CONTAINERS,
    SUPPORTED_ATTACKS,
    SUPPORTED_TRAFFIC,
    SUPPORTED_INTENSITY,
)

logger = logging.getLogger(__name__)

_client = None
ATTACK_TARGET_HOST = os.getenv("LAB_TARGET_HOST", "victim")

MANAGED_ATTACKER_ROLE_KEY = "com.threatwatch.role"
MANAGED_ATTACKER_ROLE_VAL = "attacker-extra"
MANAGED_ATTACKER_OWNER_KEY = "com.threatwatch.owner"
MANAGED_ATTACKER_OWNER_VAL = "threatwatch-live-lab"
MAX_ATTACKERS = 10

_CONTAINER_SNAPSHOT_CACHE = {"ts": 0.0, "rows": None}
_ATTACKER_NAMES_CACHE = {"ts": 0.0, "names": None}


def get_container_snapshot(ttl_s: float = 0.5) -> list[dict]:
    """
    Returns a low-level docker container list snapshot (fast, single daemon call).
    Cached for a short TTL because /api/lab/status and SSE can call frequently.
    """
    now = time.time()
    cached = _CONTAINER_SNAPSHOT_CACHE.get("rows")
    ts = float(_CONTAINER_SNAPSHOT_CACHE.get("ts") or 0.0)
    if cached is not None and (now - ts) < float(ttl_s):
        return cached

    client = _init_client()
    if not client:
        _CONTAINER_SNAPSHOT_CACHE["ts"] = now
        _CONTAINER_SNAPSHOT_CACHE["rows"] = []
        return []

    try:
        rows = client.api.containers(all=True) or []
    except Exception:
        rows = []

    _CONTAINER_SNAPSHOT_CACHE["ts"] = now
    _CONTAINER_SNAPSHOT_CACHE["rows"] = rows
    return rows


def _name_state_label_maps(rows: list[dict]) -> tuple[dict[str, bool], dict[str, dict]]:
    running: dict[str, bool] = {}
    labels: dict[str, dict] = {}
    for r in rows:
        names = r.get("Names") or []
        state = (r.get("State") or "").lower()
        is_running = state == "running"
        lbs = r.get("Labels") or {}
        for n in names:
            if not isinstance(n, str):
                continue
            name = n.lstrip("/")
            if not name:
                continue
            running[name] = is_running
            labels[name] = lbs
    return running, labels


def _sort_attacker_names(names: list[str]) -> list[str]:
    base = LAB_CONTAINERS["attacker"]

    def key(n: str):
        if n == base:
            return (0, 0)
        if n.startswith(base + "-"):
            try:
                return (1, int(n.split("-")[-1]))
            except Exception:
                return (1, 999999)
        return (2, 999999)

    return sorted(names, key=key)


def _list_extra_attacker_names() -> list[str]:
    rows = get_container_snapshot(ttl_s=0.5)
    _, labels_by_name = _name_state_label_maps(rows)
    out: list[str] = []
    for name, lbs in labels_by_name.items():
        if (
            lbs.get(MANAGED_ATTACKER_ROLE_KEY) == MANAGED_ATTACKER_ROLE_VAL
            and lbs.get(MANAGED_ATTACKER_OWNER_KEY) == MANAGED_ATTACKER_OWNER_VAL
        ):
            out.append(name)
    return out


def get_attacker_names(force_refresh: bool = False) -> list[str]:
    """
    Returns the base attacker container plus any managed extra attackers.
    Also reconciles ATTACKER_STATE["count"] against actual containers.
    """
    now = time.time()
    cached = _ATTACKER_NAMES_CACHE.get("names")
    ts = float(_ATTACKER_NAMES_CACHE.get("ts") or 0.0)
    if not force_refresh and cached is not None and (now - ts) < 1.0:
        return cached

    base = LAB_CONTAINERS["attacker"]
    extras = _list_extra_attacker_names()
    names = _sort_attacker_names([base, *extras])
    ATTACKER_STATE["count"] = max(1, len(names))
    _ATTACKER_NAMES_CACHE["ts"] = now
    _ATTACKER_NAMES_CACHE["names"] = names
    return names


def get_running_map(names: list[str] | None = None) -> dict[str, bool]:
    """
    Fast running-state map using a single docker snapshot.
    If names is provided, returns only those names.
    """
    rows = get_container_snapshot(ttl_s=0.5)
    running_by_name, _ = _name_state_label_maps(rows)
    if not names:
        return running_by_name
    return {n: bool(running_by_name.get(n, False)) for n in names}


def set_attacker_count(count: int) -> bool:
    """
    Ensure there are exactly `count` attacker containers in the lab network.
    Base container `ddos-attacker` counts as 1; extras are created as ddos-attacker-2..N.
    """
    try:
        count = int(count)
    except Exception:
        return False

    if count < 1 or count > MAX_ATTACKERS:
        logger.warning("Requested attacker count out of range: %s", count)
        return False

    # Topology changes while load generators are running are ambiguous; stop first.
    stop_attack()
    stop_traffic()

    logger.warning("Scaling attackers to %s total containers", count)
    # Force a refresh so we don't operate on stale attacker name caches.
    get_attacker_names(force_refresh=True)

    base_name = LAB_CONTAINERS["attacker"]
    base = _get_container(base_name)
    if not base:
        logger.error("Cannot scale attackers: base attacker container missing: %s", base_name)
        return False

    base.reload()
    try:
        if base.status != "running":
            base.start()
            time.sleep(0.2)
            base.reload()
    except Exception:
        pass

    if base.status != "running":
        logger.error("Cannot scale attackers: base attacker container not running: %s", base_name)
        return False

    image = getattr(base, "image", None)
    image_ref = getattr(image, "id", None) or (getattr(image, "tags", None) or [None])[0]
    if not image_ref:
        logger.error("Cannot scale attackers: failed to determine attacker image")
        return False

    base_nets = list((base.attrs.get("NetworkSettings") or {}).get("Networks") or {})
    if not base_nets:
        logger.error("Cannot scale attackers: base attacker has no networks attached")
        return False

    logger.info("Base attacker networks: %s", base_nets)

    desired_extra_names = [f"{base_name}-{i}" for i in range(2, count + 1)]
    existing_extras = _list_extra_attacker_names()
    existing_set = set(existing_extras)
    logger.info("Existing extra attackers: %s", _sort_attacker_names(existing_extras))
    logger.info("Desired extra attackers: %s", desired_extra_names)

    # Remove extras not in desired set.
    for name in _sort_attacker_names(existing_extras):
        if name in desired_extra_names:
            continue
        c = _get_container(name)
        if not c:
            continue
        try:
            c.remove(force=True)
            logger.warning("Removed extra attacker container: %s", name)
        except Exception as exc:
            logger.error("Failed removing attacker %s: %s", name, exc)
            return False

    # Create missing extras.
    client = _init_client()
    if not client:
        return False

    for name in desired_extra_names:
        if name in existing_set:
            # Ensure it's started.
            c = _get_container(name)
            if c:
                try:
                    c.reload()
                    if c.status != "running":
                        c.start()
                except Exception:
                    pass
            continue
        # Avoid taking over arbitrary containers (name collision) unless it's managed by us.
        if _get_container(name) is not None:
            logger.error("Cannot create attacker %s: container with that name already exists", name)
            return False
        try:
            c = client.containers.run(
                image_ref,
                name=name,
                detach=True,
                command=["sleep", "infinity"],
                network=base_nets[0],
                labels={
                    MANAGED_ATTACKER_ROLE_KEY: MANAGED_ATTACKER_ROLE_VAL,
                    MANAGED_ATTACKER_OWNER_KEY: MANAGED_ATTACKER_OWNER_VAL,
                },
            )
            # Attach to any additional networks used by base attacker.
            for net_name in base_nets[1:]:
                try:
                    net = client.networks.get(net_name)
                    net.connect(c)
                except Exception:
                    # Best-effort; the first network is enough for lab traffic.
                    pass
            logger.warning("Created extra attacker container: %s", name)
        except APIError as exc:
            logger.error("Failed creating attacker %s: %s", name, exc)
            return False
        except Exception as exc:
            logger.error("Failed creating attacker %s: %s", name, exc)
            return False

    ATTACKER_STATE["count"] = count
    get_attacker_names(force_refresh=True)
    return True


def _start_bg(container_name: str, pid_path: str, cmd: str) -> bool:
    # Start cmd in a new session so we can kill its whole process group by PID.
    # Use shlex.quote so complex commands don't break the wrapper quoting.
    inner = shlex.quote(cmd)
    wrapped = (
        "sh -lc \""
        f"rm -f {pid_path}; "
        f"setsid sh -lc {inner} >/dev/null 2>&1 & "
        f"echo $! > {pid_path}"
        "\""
    )
    return _exec_in_container(container_name, wrapped, detach=True)


def _stop_bg(container_name: str, pid_path: str) -> bool:
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
    return _exec_in_container(container_name, wrapped, detach=False)

def _bg_running(container_name: str, pid_path: str) -> bool:
    """
    Checks if the process group leader PID in pid_path is still alive.
    This prevents false 'running' state when the command exits immediately.
    """
    container = _get_container(container_name)
    if not container:
        return False

    try:
        r = container.exec_run(
            "sh -lc '"
            f"test -f {pid_path} || exit 1; "
            f"pid=$(cat {pid_path}); "
            "kill -0 $pid >/dev/null 2>&1 || exit 1; "
            "exit 0"
            "'",
            detach=False,
        )
        exit_code = getattr(r, "exit_code", None)
        if exit_code is None and isinstance(r, tuple):
            exit_code = r[0]
        return int(exit_code) == 0
    except Exception:
        return False


def attack_running() -> bool:
    return attack_running_count() > 0


def traffic_running() -> bool:
    return traffic_running_count() > 0


def attack_running_count() -> int:
    pid_path = "/tmp/threatwatch_attack.pid"
    return sum(1 for n in get_attacker_names() if _bg_running(n, pid_path))


def traffic_running_count() -> int:
    pid_path = "/tmp/threatwatch_traffic.pid"
    return sum(1 for n in get_attacker_names() if _bg_running(n, pid_path))

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


def _exec_in_container(container_name: str, command: str, detach: bool = True) -> bool:
    container = _get_container(container_name)

    if not container:
        logger.warning("Cannot run command. Missing container: %s", container_name)
        return False

    if container.status != "running":
        logger.warning("Cannot run command. Container not running: %s", container_name)
        return False

    try:
        container.exec_run(command, detach=detach)
        return True
    except APIError as exc:
        logger.error("Docker exec failed: %s", exc)
        return False


def _exec_on_attackers(command: str, detach: bool = True) -> bool:
    ok = True
    for name in get_attacker_names():
        ok = _exec_in_container(name, command, detach=detach) and ok
    return ok


def start_attack(attack_type: str = "udp") -> bool:
    return start_attack_with_load(attack_type=attack_type, intensity="medium")


def start_attack_with_load(
    attack_type: str = "udp",
    intensity: str = "medium",
    pps: int | None = None,
    interval_ms: int | None = None,
) -> bool:
    attack_type = attack_type.lower().strip()
    intensity = intensity.lower().strip()

    if attack_type not in SUPPORTED_ATTACKS:
        logger.warning("Unsupported attack type requested: %s", attack_type)
        return False
    if intensity not in SUPPORTED_INTENSITY:
        logger.warning("Unsupported intensity requested: %s", intensity)
        return False

    stop_attack()

    # Defaults tuned to be noticeable but not instantly destructive.
    default_udp_pps = {"low": 200, "medium": 1000, "high": 5000}[intensity]
    default_http_interval_ms = {"low": 200, "medium": 50, "high": 10}[intensity]

    if pps is None and attack_type in {"udp", "dns", "ntp", "ssdp"}:
        pps = default_udp_pps
    if interval_ms is None and attack_type in {"http"}:
        interval_ms = default_http_interval_ms

    # Clamp values for safety.
    if pps is not None:
        pps = int(max(1, min(pps, 200_000)))
    if interval_ms is not None:
        interval_ms = int(max(0, min(interval_ms, 60_000)))

    def hping_interval_us(p: int) -> int:
        return max(1, int(1_000_000 / p))

    # NOTE: do not use a dict literal here. Python evaluates all f-strings eagerly.
    if attack_type == "udp":
        # Generic UDP flood (victim doesn't need to listen; we care about flow patterns).
        attack_cmd = f"hping3 --udp -i u{hping_interval_us(int(pps))} -p 80 {ATTACK_TARGET_HOST}"
    elif attack_type == "dns":
        attack_cmd = f"hping3 --udp -i u{hping_interval_us(int(pps))} -p 53 -d 32 {ATTACK_TARGET_HOST}"
    elif attack_type == "ntp":
        attack_cmd = f"hping3 --udp -i u{hping_interval_us(int(pps))} -p 123 -d 48 {ATTACK_TARGET_HOST}"
    elif attack_type == "ssdp":
        attack_cmd = f"hping3 --udp -i u{hping_interval_us(int(pps))} -p 1900 -d 120 {ATTACK_TARGET_HOST}"
    elif attack_type == "http":
        interval_s = float(int(interval_ms)) / 1000.0
        attack_cmd = (
            f"while true; do curl -m 1 -s http://{ATTACK_TARGET_HOST}/ >/dev/null; "
            f"sleep {interval_s:.3f}; done"
        )
    else:
        logger.warning("Unsupported attack type requested: %s", attack_type)
        return False

    attacker_names = get_attacker_names()
    started_all = True
    for name in attacker_names:
        started_all = _start_bg(name, "/tmp/threatwatch_attack.pid", attack_cmd) and started_all

    if started_all:
        # If the command exits immediately (bad command / missing binary), don't lie about running state.
        time.sleep(0.1)
        running_count = attack_running_count()
        if running_count != len(attacker_names):
            logger.error("Attack process exited immediately (type=%s intensity=%s)", attack_type, intensity)
            stop_attack()
            return False
        ATTACK_STATE["running"] = True
        ATTACK_STATE["type"] = attack_type
        ATTACK_STATE["intensity"] = intensity
        ATTACK_STATE["pps"] = pps
        ATTACK_STATE["interval_ms"] = interval_ms
        ATTACK_STATE["attacker_count"] = len(attacker_names)
        ATTACK_STATE["started_at"] = time.time()
        logger.warning("Started %s attack (intensity=%s pps=%s interval_ms=%s)", attack_type, intensity, pps, interval_ms)

    return started_all


def stop_attack() -> bool:
    stopped = True
    for name in get_attacker_names():
        stopped = _stop_bg(name, "/tmp/threatwatch_attack.pid") and stopped

    ATTACK_STATE["running"] = False
    ATTACK_STATE["type"] = None
    ATTACK_STATE["pps"] = None
    ATTACK_STATE["interval_ms"] = None
    ATTACK_STATE["attacker_count"] = None
    ATTACK_STATE["started_at"] = None

    return stopped


def start_traffic(traffic_type: str = "mixed") -> bool:
    return start_traffic_with_load(traffic_type=traffic_type, intensity="medium")


def start_traffic_with_load(
    traffic_type: str = "mixed",
    intensity: str = "medium",
    interval_ms: int | None = None,
) -> bool:
    traffic_type = traffic_type.lower().strip()
    intensity = intensity.lower().strip()

    if traffic_type not in SUPPORTED_TRAFFIC:
        logger.warning("Unsupported traffic type requested: %s", traffic_type)
        return False
    if intensity not in SUPPORTED_INTENSITY:
        logger.warning("Unsupported intensity requested: %s", intensity)
        return False

    stop_traffic()

    default_interval_ms = {"low": 200, "medium": 100, "high": 20}[intensity]
    if interval_ms is None:
        interval_ms = default_interval_ms
    interval_ms = int(max(0, min(interval_ms, 60_000)))

    traffic_cmd = {
        "http": f"while true; do curl -m 1 -s http://{ATTACK_TARGET_HOST} >/dev/null 2>&1; sleep {interval_ms/1000:.3f}; done",
        "ping": f"while true; do ping -c 1 -W 1 {ATTACK_TARGET_HOST} >/dev/null 2>&1; sleep {interval_ms/1000:.3f}; done",
        "mixed": (
            "while true; do "
            "n=$(( $(date +%s%N) % 2 )); "
            f"if [ \"$n\" -eq 0 ]; then curl -m 1 -s http://{ATTACK_TARGET_HOST} >/dev/null 2>&1; "
            f"else ping -c 1 -W 1 {ATTACK_TARGET_HOST} >/dev/null 2>&1; fi; "
            f"sleep {interval_ms/1000:.3f}; done"
        ),
        "cic_benign": (
            # CICDDoS2019-style benign profile approximation:
            # - periodic HTTP requests to a few paths (web browsing-like)
            # - occasional ICMP echo
            # - occasional low-rate UDP to common service ports (DNS/NTP-ish) to diversify flows
            # Keep it low volume; this is "normal".
            "while true; do "
            "n=$(( $(date +%s%N) % 7 )); "
            f"if [ \"$n\" -eq 0 ]; then curl -m 1 -s http://{ATTACK_TARGET_HOST}/ >/dev/null 2>&1; "
            f"elif [ \"$n\" -eq 1 ]; then curl -m 1 -s http://{ATTACK_TARGET_HOST}/index.html >/dev/null 2>&1; "
            f"elif [ \"$n\" -eq 2 ]; then curl -m 1 -s http://{ATTACK_TARGET_HOST}/static/app.js >/dev/null 2>&1; "
            f"elif [ \"$n\" -eq 3 ]; then ping -c 1 -W 1 {ATTACK_TARGET_HOST} >/dev/null 2>&1; "
            f"elif [ \"$n\" -eq 4 ]; then hping3 --udp -c 1 -p 53 -d 24 {ATTACK_TARGET_HOST} >/dev/null 2>&1; "
            f"elif [ \"$n\" -eq 5 ]; then hping3 --udp -c 1 -p 123 -d 48 {ATTACK_TARGET_HOST} >/dev/null 2>&1; "
            f"else true; fi; "
            f"sleep {interval_ms/1000:.3f}; "
            "done"
        ),
    }[traffic_type]

    attacker_names = get_attacker_names()
    started_all = True
    for name in attacker_names:
        started_all = _start_bg(name, "/tmp/threatwatch_traffic.pid", traffic_cmd) and started_all

    if started_all:
        time.sleep(0.1)
        running_count = traffic_running_count()
        if running_count != len(attacker_names):
            logger.error("Traffic process exited immediately (type=%s intensity=%s)", traffic_type, intensity)
            stop_traffic()
            return False
        TRAFFIC_STATE["running"] = True
        TRAFFIC_STATE["type"] = traffic_type
        TRAFFIC_STATE["intensity"] = intensity
        TRAFFIC_STATE["interval_ms"] = interval_ms
        TRAFFIC_STATE["attacker_count"] = len(attacker_names)
        TRAFFIC_STATE["started_at"] = time.time()
        logger.warning("Started normal traffic (%s) intensity=%s interval_ms=%s", traffic_type, intensity, interval_ms)

    return started_all


def stop_traffic() -> bool:
    stopped = True
    for name in get_attacker_names():
        stopped = _stop_bg(name, "/tmp/threatwatch_traffic.pid") and stopped
    TRAFFIC_STATE["running"] = False
    TRAFFIC_STATE["type"] = None
    TRAFFIC_STATE["interval_ms"] = None
    TRAFFIC_STATE["attacker_count"] = None
    TRAFFIC_STATE["started_at"] = None
    return stopped
