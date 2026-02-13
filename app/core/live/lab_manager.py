import time
import json

import redis

from app.core.live.lab_controller import (
    container_running,
    process_running,
    start_attack,
    stop_attack,
    start_traffic,
    stop_traffic,
)
from app.core.live.lab_state import (
    ATTACK_STATE,
    TRAFFIC_STATE,
    LAB_CONTAINERS,
    LATEST_LAB_STATE,
)
from app.core.live.redis_consumer import get_consumer_status, get_latest_result
from app.core.live.redis_consumer import STREAM_NAME
from app.config import settings


def _get_redis_debug():
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
        "error": None,
    }

    try:
        client = redis.Redis(host=settings.redis_host, port=6379, decode_responses=True)
        debug["stream_length"] = int(client.xlen(STREAM_NAME))
        debug["has_data"] = debug["stream_length"] > 0
    except Exception as exc:
        debug["error"] = str(exc)

    return debug


def get_lab_status():
    LATEST_LAB_STATE["redis"] = container_running(LAB_CONTAINERS["redis"])
    LATEST_LAB_STATE["victim"] = container_running(LAB_CONTAINERS["victim"])
    LATEST_LAB_STATE["attacker"] = container_running(LAB_CONTAINERS["attacker"])
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

    redis_debug = _get_redis_debug()
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

    LATEST_LAB_STATE["attack_running"] = ATTACK_STATE["running"]
    LATEST_LAB_STATE["attack_type"] = ATTACK_STATE["type"]
    LATEST_LAB_STATE["traffic_running"] = TRAFFIC_STATE["running"]
    LATEST_LAB_STATE["traffic_type"] = TRAFFIC_STATE["type"]
    LATEST_LAB_STATE["agent_capturing"] = bool(LATEST_LAB_STATE["agent"] and agent_capturing)
    LATEST_LAB_STATE["agent_last_capture_at"] = agent_last_capture_at
    LATEST_LAB_STATE["agent_last_capture_age_s"] = agent_last_capture_age_s
    LATEST_LAB_STATE["latest_results"] = latest_result
    LATEST_LAB_STATE["consumer"] = consumer_status
    LATEST_LAB_STATE["redis_debug"] = redis_debug
    LATEST_LAB_STATE["traffic_stream"] = traffic_stream

    return LATEST_LAB_STATE


def trigger_attack(attack_type: str):
    return start_attack(attack_type)


def stop_lab_attack():
    return stop_attack()


def trigger_traffic(traffic_type: str):
    return start_traffic(traffic_type)


def stop_lab_traffic():
    return stop_traffic()
