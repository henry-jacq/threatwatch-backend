"""Global runtime state for live lab orchestration."""

LAB_CONTAINERS = {
    "attacker": "ddos-attacker",
    "victim": "ddos-victim",
    "redis": "ddos-redis",
}

SUPPORTED_ATTACKS = {"syn", "udp", "http", "random"}
SUPPORTED_TRAFFIC = {"http", "ping", "mixed"}

ATTACK_STATE = {
    "running": False,
    "type": None,
    "started_at": None,
}

TRAFFIC_STATE = {
    "running": False,
    "type": None,
    "started_at": None,
}

LATEST_LAB_STATE = {
    "redis": False,
    "agent": False,
    "victim": False,
    "attacker": False,
    # True when inference is producing fresh results.
    "streaming": False,
    # True when Redis has fresh traffic windows (capture -> Redis is alive).
    "traffic_stream": False,
    "attack_running": False,
    "attack_type": None,
    "traffic_running": False,
    "traffic_type": None,
    "agent_capturing": False,
    "agent_last_capture_at": None,
    "agent_last_capture_age_s": None,
    "latest_results": None,
    "redis_debug": {
        "host": None,
        "stream": None,
        "has_data": False,
        "stream_length": 0,
        "latest_entry_id": None,
        "latest_flow_count": 0,
        "latest_payload_timestamp": None,
        "latest_payload_age_s": None,
        "latest_flow_sample": None,
        "error": None,
    },
}
