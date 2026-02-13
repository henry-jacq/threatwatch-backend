import asyncio
import json
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.live.lab_manager import get_lab_status, stop_lab_attack, trigger_attack
from app.core.live.lab_manager import stop_lab_traffic, trigger_traffic
from app.core.live.lab_state import SUPPORTED_ATTACKS, SUPPORTED_TRAFFIC
from app.core.live.lab_controller import _get_container

router = APIRouter(prefix="/api/lab", tags=["lab"])


@router.get("/status")
async def status():
    return get_lab_status()


@router.get("/stream")
async def stream_status():
    async def event_generator():
        last_key = None
        while True:
            current = get_lab_status()
            latest_results_ts = (current.get("latest_results") or {}).get("timestamp")
            latest_payload_ts = (current.get("redis_debug") or {}).get("latest_payload_timestamp")
            attack_state = (current.get("attack_running"), current.get("attack_type"))
            traffic_state = (current.get("traffic_running"), current.get("traffic_type"))
            key = (latest_results_ts, latest_payload_ts, attack_state, traffic_state)

            if key != last_key:
                last_key = key
                yield f"data: {json.dumps(current)}\n\n"
            else:
                # keep the SSE connection alive behind proxies.
                yield f": heartbeat {int(time.time())}\n\n"

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/attack/start")
async def start_attack(type: Literal["syn", "udp", "http", "random"] = "syn"):
    success = trigger_attack(type)

    if not success:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Failed to start attack",
                "type": type,
                "supported_types": sorted(SUPPORTED_ATTACKS),
            },
        )

    return {"attack_started": True, "type": type}


@router.post("/attack/stop")
async def stop_attack():
    success = stop_lab_attack()

    return {
        "attack_stopped": bool(success),
    }


@router.post("/traffic/start")
async def start_traffic(type: Literal["http", "ping", "mixed"] = "mixed"):
    success = trigger_traffic(type)

    if not success:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Failed to start normal traffic",
                "type": type,
                "supported_types": sorted(SUPPORTED_TRAFFIC),
            },
        )

    return {"traffic_started": True, "type": type}


@router.post("/traffic/stop")
async def stop_traffic():
    success = stop_lab_traffic()
    return {"traffic_stopped": bool(success)}


@router.get("/debug/agent")
async def debug_agent():
    # Agent runs inside the victim container now.
    c = _get_container("ddos-victim")
    if not c:
        raise HTTPException(status_code=404, detail="ddos-victim container not found")

    try:
        out = c.exec_run(
            "sh -lc '"
            "python -c \"from scapy.all import get_if_list; print('ifaces:', get_if_list())\"; "
            "python -c \"import os; print('pid1:', open('/proc/1/cmdline','rb').read().replace(b'\\x00',b' '))\"; "
            "python -c \"import os; "
            "needle=b'agent.py'; "
            "hits=[]; "
            "import glob; "
            "import pathlib; "
            "for p in glob.glob('/proc/[0-9]*/cmdline'): "
            "  try: "
            "    b=open(p,'rb').read(); "
            "    "
            "    "
            "    "
            "    "
            "  except Exception: "
            "    continue; "
            "  if needle in b: "
            "    hits.append((p.split('/')[2], b.replace(b'\\x00',b' '))); "
            "print('agent_hits:', hits[:5])\"; "
            "true'",
            detach=False,
        )
        stdout = out.output.decode(errors="replace") if hasattr(out, "output") else str(out)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"agent_debug": stdout}
