"""Global runtime state for live lab orchestration."""

LAB_CONTAINERS = {
    "attacker": "ddos-attacker",
    "victim": "ddos-victim",
    "redis": "ddos-redis",
}

# CICDDoS2019-style attack families exposed in the lab UI.
# Keep these aligned with the model's training domain.
SUPPORTED_ATTACKS = {"udp", "http", "dns", "ntp", "ssdp"}
SUPPORTED_TRAFFIC = {"http", "ping", "mixed", "cic_benign"}
SUPPORTED_INTENSITY = {"low", "medium", "high"}

ATTACK_STATE = {
    "running": False,
    "type": None,
    "intensity": "medium",
    "pps": None,
    "interval_ms": None,
    "started_at": None,
}

ATTACKER_STATE = {
    # Desired attacker count (including the base ddos-attacker container).
    # On backend restart, this will be reconciled against actual running containers.
    "count": 1,
}

TRAFFIC_STATE = {
    "running": False,
    "type": None,
    "intensity": "medium",
    "interval_ms": None,
    "started_at": None,
}

LATEST_LAB_STATE = {
    "redis": False,
    "agent": False,
    "victim": False,
    "attacker": False,
    "attacker_count": 1,
    "attacker_running_count": 0,
    "attacker_names": [],
    # True when inference is producing fresh results.
    "streaming": False,
    # True when Redis has fresh traffic windows (capture -> Redis is alive).
    "traffic_stream": False,
    "attack_running": False,
    "attack_type": None,
    "attack_intensity": None,
    "attack_pps": None,
    "attack_interval_ms": None,
    "traffic_running": False,
    "traffic_type": None,
    "traffic_intensity": None,
    "traffic_interval_ms": None,
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
