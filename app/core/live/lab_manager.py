import time
import json

import redis

from app.core.live.lab_controller import (
    process_running,
    start_attack,
    stop_attack,
    start_traffic,
    stop_traffic,
    start_attack_with_load,
    start_traffic_with_load,
    attack_running,
    traffic_running,
    get_attacker_names,
    attack_running_count,
    traffic_running_count,
    get_running_map,
)
from app.core.live.lab_state import (
    ATTACK_STATE,
    ATTACKER_STATE,
    TRAFFIC_STATE,
    LAB_CONTAINERS,
    LATEST_LAB_STATE,
)
from app.core.live.redis_consumer import get_consumer_status, get_latest_result
from app.core.live.redis_consumer import STREAM_NAME
from app.config import settings

_LAB_STATUS_CACHE = {"ts": 0.0, "value": None}
_LAB_STATUS_TTL_S = 0.8


def _get_redis_debug(consumer_status: dict):
    debug = {
        "host": settings.redis_host,
        "stream": STREAM_NAME,
        "has_data": False,
        "stream_length": 0,
        # With XDEL draining enabled, the stream is expected to be near-empty.
        "latest_entry_id": None,
        "latest_flow_count": 0,
        "latest_payload_timestamp": None,
        "latest_payload_age_s": None,
        "latest_flow_sample": None,
        "latest_unique_source_count": None,
        "latest_unique_source_sample": None,
        "error": None,
    }

    try:
        client = redis.Redis(host=settings.redis_host, port=6379, decode_responses=True)
        debug["stream_length"] = int(client.xlen(STREAM_NAME))
        debug["has_data"] = debug["stream_length"] > 0
    except Exception as exc:
        debug["error"] = str(exc)

    # Redis is drained on consume; use consumer memory for "latest payload" debugging.
    debug["latest_entry_id"] = consumer_status.get("last_entry_id")
    debug["latest_flow_count"] = int(consumer_status.get("last_flow_count") or 0)
    debug["latest_flow_sample"] = consumer_status.get("last_flow_sample")
    debug["latest_unique_source_count"] = consumer_status.get("last_unique_source_count")
    debug["latest_unique_source_sample"] = consumer_status.get("last_unique_source_sample")
    debug["latest_payload_timestamp"] = consumer_status.get("last_message_at")
    ts = debug["latest_payload_timestamp"]
    if isinstance(ts, (int, float)):
        debug["latest_payload_age_s"] = max(0.0, time.time() - float(ts))

    return debug


def get_lab_status():
    # Cache status briefly to avoid hammering Docker/Redis when UI polls + SSE ticks.
    now = time.time()
    cached = _LAB_STATUS_CACHE.get("value")
    ts = float(_LAB_STATUS_CACHE.get("ts") or 0.0)
    if cached is not None and (now - ts) < _LAB_STATUS_TTL_S:
        return cached

    attacker_names = get_attacker_names()
    names_to_check = [LAB_CONTAINERS["redis"], LAB_CONTAINERS["victim"], *attacker_names]
    running_map = get_running_map(names_to_check)

    LATEST_LAB_STATE["redis"] = bool(running_map.get(LAB_CONTAINERS["redis"], False))
    LATEST_LAB_STATE["victim"] = bool(running_map.get(LAB_CONTAINERS["victim"], False))
    attacker_running_count_ = sum(1 for n in attacker_names if running_map.get(n, False))
    LATEST_LAB_STATE["attacker_names"] = attacker_names
    LATEST_LAB_STATE["attacker_count"] = len(attacker_names)
    LATEST_LAB_STATE["attacker_running_count"] = attacker_running_count_
    # True only if all configured attackers are running.
    LATEST_LAB_STATE["attacker"] = bool(attacker_names) and attacker_running_count_ == len(attacker_names)
    ATTACKER_STATE["count"] = len(attacker_names)
    # Agent now runs inside victim container.
    LATEST_LAB_STATE["agent"] = bool(
        LATEST_LAB_STATE["victim"]
        and process_running(LAB_CONTAINERS["victim"], "agent.py")
    )

    latest_result = get_latest_result()
    consumer_status = get_consumer_status()

    # Streaming is healthy only if infra + consumer + fresh inference updates are present.
    has_fresh_result = False
    ts = latest_result.get("timestamp") if latest_result else None

    if isinstance(ts, (int, float)):
        has_fresh_result = (time.time() - float(ts)) < 15

    redis_debug = _get_redis_debug(consumer_status)
    # Since stream entries are drained (XDEL), use consumer's last seen payload timestamp.
    agent_last_capture_at = consumer_status.get("last_message_at")
    agent_last_capture_age_s = None
    if isinstance(agent_last_capture_at, (int, float)):
        agent_last_capture_age_s = max(0.0, time.time() - float(agent_last_capture_at))
    agent_capturing = False
    if isinstance(agent_last_capture_at, (int, float)):
        agent_capturing = (time.time() - float(agent_last_capture_at)) < 15

    # Traffic stream = fresh Redis payload (capture path alive), independent of inference.
    traffic_stream = bool(
        isinstance(agent_last_capture_at, (int, float))
        and (time.time() - float(agent_last_capture_at)) < 15
        and (consumer_status.get("last_flow_count") or 0) > 0
    )

    LATEST_LAB_STATE["streaming"] = bool(
        LATEST_LAB_STATE["redis"]
        and LATEST_LAB_STATE["agent"]
        and consumer_status.get("running")
        and consumer_status.get("connected")
        and has_fresh_result
    )

    # Sync attack/traffic running flags with actual attacker process, not just in-memory state.
    real_attack_running = bool(ATTACK_STATE["running"] and attack_running())
    if ATTACK_STATE["running"] and not real_attack_running:
        ATTACK_STATE["running"] = False
        ATTACK_STATE["type"] = None
        ATTACK_STATE["pps"] = None
        ATTACK_STATE["interval_ms"] = None
        ATTACK_STATE["attacker_count"] = None

    real_traffic_running = bool(TRAFFIC_STATE["running"] and traffic_running())
    if TRAFFIC_STATE["running"] and not real_traffic_running:
        TRAFFIC_STATE["running"] = False
        TRAFFIC_STATE["type"] = None
        TRAFFIC_STATE["interval_ms"] = None
        TRAFFIC_STATE["attacker_count"] = None

    LATEST_LAB_STATE["attack_running"] = real_attack_running
    LATEST_LAB_STATE["attack_type"] = ATTACK_STATE["type"]
    LATEST_LAB_STATE["attack_intensity"] = ATTACK_STATE.get("intensity")
    LATEST_LAB_STATE["attack_pps"] = ATTACK_STATE.get("pps")
    LATEST_LAB_STATE["attack_interval_ms"] = ATTACK_STATE.get("interval_ms")
    # Avoid expensive per-attacker PID checks when nothing is running.
    LATEST_LAB_STATE["attack_running_attackers"] = int(attack_running_count()) if real_attack_running else 0
    LATEST_LAB_STATE["traffic_running"] = real_traffic_running
    LATEST_LAB_STATE["traffic_type"] = TRAFFIC_STATE["type"]
    LATEST_LAB_STATE["traffic_intensity"] = TRAFFIC_STATE.get("intensity")
    LATEST_LAB_STATE["traffic_interval_ms"] = TRAFFIC_STATE.get("interval_ms")
    LATEST_LAB_STATE["traffic_running_attackers"] = int(traffic_running_count()) if real_traffic_running else 0
    LATEST_LAB_STATE["agent_capturing"] = bool(LATEST_LAB_STATE["agent"] and agent_capturing)
    LATEST_LAB_STATE["agent_last_capture_at"] = agent_last_capture_at
    LATEST_LAB_STATE["agent_last_capture_age_s"] = agent_last_capture_age_s
    LATEST_LAB_STATE["latest_results"] = latest_result
    LATEST_LAB_STATE["consumer"] = consumer_status
    LATEST_LAB_STATE["redis_debug"] = redis_debug
    LATEST_LAB_STATE["traffic_stream"] = traffic_stream

    _LAB_STATUS_CACHE["ts"] = now
    _LAB_STATUS_CACHE["value"] = LATEST_LAB_STATE
    return LATEST_LAB_STATE


def trigger_attack(attack_type: str):
    return start_attack(attack_type)


def trigger_attack_load(attack_type: str, intensity: str, pps: int | None, interval_ms: int | None):
    return start_attack_with_load(attack_type=attack_type, intensity=intensity, pps=pps, interval_ms=interval_ms)


def stop_lab_attack():
    return stop_attack()


def trigger_traffic(traffic_type: str):
    return start_traffic(traffic_type)


def trigger_traffic_load(traffic_type: str, intensity: str, interval_ms: int | None):
    return start_traffic_with_load(traffic_type=traffic_type, intensity=intensity, interval_ms=interval_ms)


def stop_lab_traffic():
    return stop_traffic()
