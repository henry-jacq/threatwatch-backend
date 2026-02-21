"""Live Redis stream consumer used by the lab status API."""

import asyncio
import json
import logging
import os
import time
from collections import deque
from copy import deepcopy

import numpy as np
import pandas as pd
import redis.asyncio as redis
from redis.exceptions import ResponseError

from app.config import settings
from app.core.datasets.ftg_dataset import FTGDataset
from app.core.ml.inference import InferenceEngine
from app.core.ml.model_loader import model_manager
from app.core.ml.preprocessing import preprocess_and_split_data

logger = logging.getLogger(__name__)

STREAM_NAME = "ddos_stream"
GROUP_NAME = "backend-group"
CONSUMER_NAME = "backend-1"
PENDING_MIN_IDLE_MS = 30_000

LATEST_RESULT = None
CONSUMER_STATUS = {
    "running": False,
    "connected": False,
    "last_error": None,
    "pid": None,
    "started_at": None,
    "last_message_at": None,
    "last_entry_id": None,
    "last_flow_count": None,
    "last_flow_sample": None,
    # Debug: number of distinct sources seen in the last payload window.
    "last_unique_source_count": None,
    "last_unique_source_sample": None,
    # Debug: pipeline counts for last payload.
    "last_raw_flow_count": None,
    "last_post_preprocess_flow_count": None,
    "last_slot_count": None,
    "last_total_flow_graphs": None,
}

# Adaptive state
DRIFT_HISTORY = deque(maxlen=500)
TRAINING_BASELINE_MEAN = 0.1
DRIFT_THRESHOLD = 0.2


def get_latest_result():
    return deepcopy(LATEST_RESULT)


def get_consumer_status():
    return deepcopy(CONSUMER_STATUS)


async def _ack_and_delete(redis_client: redis.Redis, entry_id: str) -> None:
    """
    Remove the message from both the consumer group pending list and the stream.
    Keeping Redis empty is a lab requirement.
    """
    try:
        pipe = redis_client.pipeline()
        pipe.xack(STREAM_NAME, GROUP_NAME, entry_id)
        pipe.xdel(STREAM_NAME, entry_id)
        await pipe.execute()
    except Exception as exc:
        logger.warning("Failed to ACK+DEL Redis entry %s: %s", entry_id, exc)


async def _read_pending(redis_client: redis.Redis):
    """
    Reclaim and process pending messages (e.g. after a crash) so they don't remain in Redis.
    Returns a list of (entry_id, data) pairs.
    """
    try:
        # redis-py asyncio returns: (next_start_id, [(id, {fields})...], deleted_count?)
        res = await redis_client.xautoclaim(
            STREAM_NAME,
            GROUP_NAME,
            CONSUMER_NAME,
            min_idle_time=PENDING_MIN_IDLE_MS,
            start_id="0-0",
            count=50,
        )
    except Exception as exc:
        logger.debug("XAUTOCLAIM failed: %s", exc)
        return []

    if not res:
        return []

    # Handle both old/new return shapes defensively.
    entries = []
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        entries = res[1] or []

    return entries


async def redis_live_consumer():
    """Continuously consume Redis stream records and run FTG-NET inference."""
    global LATEST_RESULT

    redis_host = settings.redis_host
    CONSUMER_STATUS["running"] = True
    CONSUMER_STATUS["last_error"] = None
    CONSUMER_STATUS["pid"] = os.getpid()
    CONSUMER_STATUS["started_at"] = time.time()

    engine = None
    scaler = None

    while True:
        redis_client = None
        try:
            logger.info("Connecting live consumer to Redis at %s", redis_host)
            redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
            await redis_client.ping()
            CONSUMER_STATUS["connected"] = True

            try:
                await redis_client.xgroup_create(STREAM_NAME, GROUP_NAME, id="$", mkstream=True)
                logger.info("Created Redis consumer group: %s", GROUP_NAME)
            except ResponseError as exc:
                if "BUSYGROUP" in str(exc):
                    logger.info("Redis consumer group already exists: %s", GROUP_NAME)
                else:
                    raise

            if engine is None or scaler is None:
                model, scaler, _ = model_manager.load_model(settings.default_model_id)
                engine = InferenceEngine(model, model_manager.device)
                logger.info("Live Redis consumer started")

            while True:
                # 1) Drain stale pending messages first.
                pending_entries = await _read_pending(redis_client)
                if pending_entries:
                    messages = [(STREAM_NAME, pending_entries)]
                else:
                    messages = None

                # 2) Then block for new messages.
                try:
                    if messages is None:
                        messages = await redis_client.xreadgroup(
                            groupname=GROUP_NAME,
                            consumername=CONSUMER_NAME,
                            streams={STREAM_NAME: ">"},
                            count=10,
                            block=5000,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    CONSUMER_STATUS["connected"] = False
                    CONSUMER_STATUS["last_error"] = str(exc)
                    logger.warning("Redis read error: %s", exc)
                    await asyncio.sleep(2)
                    continue

                if not messages:
                    continue

                CONSUMER_STATUS["connected"] = True

                for _stream, entries in messages:
                    for entry_id, data in entries:
                        try:
                            payload = json.loads(data["payload"])
                            CONSUMER_STATUS["last_entry_id"] = entry_id
                            CONSUMER_STATUS["last_message_at"] = payload.get("timestamp")
                            df = pd.DataFrame(payload.get("flows", []))
                            CONSUMER_STATUS["last_flow_count"] = int(len(df))
                            flows = payload.get("flows", [])
                            CONSUMER_STATUS["last_raw_flow_count"] = int(len(flows) if isinstance(flows, list) else 0)
                            CONSUMER_STATUS["last_flow_sample"] = flows[0] if flows else None
                            # Unique sources (typically attacker container IPs).
                            srcs = []
                            for f in flows:
                                if not isinstance(f, dict):
                                    continue
                                v = f.get("Source IP") or f.get("Src IP") or f.get("src_ip") or f.get("source_ip")
                                if v:
                                    srcs.append(str(v))
                            uniq = sorted(set(srcs))
                            CONSUMER_STATUS["last_unique_source_count"] = int(len(uniq))
                            CONSUMER_STATUS["last_unique_source_sample"] = uniq[:8]

                            if df.empty:
                                CONSUMER_STATUS["last_post_preprocess_flow_count"] = 0
                                CONSUMER_STATUS["last_slot_count"] = 0
                                CONSUMER_STATUS["last_total_flow_graphs"] = 0
                                continue

                            df.columns = df.columns.str.strip()
                            df, _ = preprocess_and_split_data(df, fit_scaler=False, scaler=scaler)
                            CONSUMER_STATUS["last_post_preprocess_flow_count"] = int(len(df))

                            if df.empty:
                                CONSUMER_STATUS["last_slot_count"] = 0
                                CONSUMER_STATUS["last_total_flow_graphs"] = 0
                                CONSUMER_STATUS["last_error"] = "No valid rows after preprocessing"
                                continue

                            dataset = FTGDataset(df, time_slot_duration="5s", has_labels=False)
                            traffic_graphs = []
                            flow_graphs = []

                            for tg, fg in dataset:
                                traffic_graphs.append(tg)
                                flow_graphs.append(fg)

                            if not traffic_graphs:
                                CONSUMER_STATUS["last_slot_count"] = 0
                                CONSUMER_STATUS["last_total_flow_graphs"] = 0
                                CONSUMER_STATUS["last_error"] = "No graphs built from last payload"
                                continue
                            CONSUMER_STATUS["last_slot_count"] = int(len(traffic_graphs))
                            CONSUMER_STATUS["last_total_flow_graphs"] = int(sum(len(fg) for fg in flow_graphs))

                            batch = engine.predict_batch(traffic_graphs, flow_graphs)
                            results = batch["results"]

                            attack_slots = sum(
                                1
                                for item in results
                                if (
                                    any(item["prediction"])
                                    if isinstance(item["prediction"], list)
                                    else item["prediction"] == 1
                                )
                            )

                            avg_prob = float(
                                np.mean(
                                    [
                                        np.mean(item["probability"])
                                        if isinstance(item["probability"], list)
                                        else item["probability"]
                                        for item in results
                                    ]
                                )
                            )

                            DRIFT_HISTORY.append(avg_prob)
                            drift_score = abs(float(np.mean(DRIFT_HISTORY)) - TRAINING_BASELINE_MEAN)
                            drift_detected = drift_score > DRIFT_THRESHOLD

                            anomaly_ratio = attack_slots / max(len(results), 1)
                            risk_score = (
                                0.5 * avg_prob
                                + 0.3 * anomaly_ratio
                                + 0.2 * min(avg_prob * 2, 1.0)
                            )

                            if risk_score > 0.75:
                                risk_level = "high"
                            elif risk_score > 0.4:
                                risk_level = "medium"
                            else:
                                risk_level = "low"

                            msg_timestamp = payload.get("timestamp")

                            LATEST_RESULT = {
                                "timestamp": msg_timestamp,
                                "slot_count": len(results),
                                "attack_slots": attack_slots,
                                "avg_probability": avg_prob,
                                "risk_score": float(risk_score),
                                "risk_level": risk_level,
                                "drift_detected": drift_detected,
                                "batch_inference_time_ms": float(batch["batch_inference_time_ms"]),
                            }

                            CONSUMER_STATUS["last_message_at"] = msg_timestamp
                            CONSUMER_STATUS["last_error"] = None

                            logger.info(
                                "Live inference | slots=%d | attack=%d | avg_prob=%.4f | risk=%.3f",
                                len(results),
                                attack_slots,
                                avg_prob,
                                risk_score,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            CONSUMER_STATUS["last_error"] = str(exc)
                            logger.error("Live inference error: %s", exc, exc_info=True)
                        finally:
                            await _ack_and_delete(redis_client, entry_id)

        except asyncio.CancelledError:
            logger.info("Live Redis consumer cancelled")
            raise
        except Exception as exc:
            # Any unexpected failure should not permanently stop the consumer task.
            CONSUMER_STATUS["connected"] = False
            CONSUMER_STATUS["last_error"] = str(exc)
            logger.error("Live consumer crashed, restarting: %s", exc, exc_info=True)
            await asyncio.sleep(2)
        finally:
            if redis_client is not None:
                try:
                    await redis_client.close()
                except Exception:
                    pass
