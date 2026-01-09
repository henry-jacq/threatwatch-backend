"""
Model inference endpoints
- /predict: Unlabeled data inference (production use)
- /evaluate: Labeled data evaluation (model testing)
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import json
import numpy as np
import logging
from typing import Dict
import uuid

from app.api.schemas.inference import EvaluationResponse, InferenceResponse
from app.core.datasets.ftg_dataset import FTGDataset
from app.core.csv.reader import read_csv_safe
from app.core.csv.schema import validate_csv_headers
from app.config import settings
from app.core.ml.model_loader import model_manager
from app.core.ml.preprocessing import preprocess_and_split_data
from app.core.ml.inference import InferenceEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/inference", tags=["inference"])

CANCEL_FLAGS: Dict[str, bool] = {}


@router.post("/predict/pcap", response_model=InferenceResponse)
async def predict_from_pcap(file: UploadFile = File(...)):
    """
    Direct PCAP to FTG-NET inference (no CSV roundtrip)
    """
    import time
    start_time = time.time()

    # Load model
    model, scaler, _ = model_manager.load_model(settings.default_model_id)
    engine = InferenceEngine(model, model_manager.device)

    # Convert PCAP to DataFrame
    from app.core.traffic.pcap_converter import pcap_bytes_to_dataframe

    df = pcap_bytes_to_dataframe(await file.read())

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid flows extracted from PCAP"
        )

    # Validate schema (Label exists but ignored for predict)
    df.columns = df.columns.str.strip()
    validate_csv_headers(df, require_label=False)

    # Preprocess
    df, _ = preprocess_and_split_data(
        df,
        fit_scaler=False,
        scaler=scaler
    )

    # Build dataset
    dataset = FTGDataset(df, has_labels=False)

    traffic_graphs, flow_graphs = [], []
    total_slots = len(dataset)
    log_every = max(1, total_slots // 10)

    for idx, (tg, fg) in enumerate(dataset, start=1):
        traffic_graphs.append(tg)
        flow_graphs.append(fg)

        if idx % log_every == 0 or idx == total_slots:
            logger.info(
                "PCAP-Predict: %d/%d slots (%.1f%%)",
                idx, total_slots, (idx / total_slots) * 100
            )

    # Inference
    batch = engine.predict_batch(traffic_graphs, flow_graphs)
    preds = batch["results"]

    attack_count = sum(
        1 for r in preds
        if (
            any(p == 1 for p in r["prediction"])
            if isinstance(r["prediction"], list)
            else r["prediction"] == 1
        )
    )

    benign_count = len(preds) - attack_count

    avg_conf = np.mean([
        np.mean(r["probability"]) if isinstance(r["probability"], list)
        else r["probability"]
        for r in preds
    ])

    logger.info(
        "PCAP Predicted | slots=%d | attacks=%d | avg_conf=%.4f | %.2f ms",
        len(preds),
        attack_count,
        avg_conf,
        (time.time() - start_time) * 1000
    )

    return InferenceResponse(
        total_samples=len(preds),
        attack_count=attack_count,
        benign_count=benign_count,
        average_confidence=float(avg_conf),
        processing_time_ms=(time.time() - start_time) * 1000
    )


# PREDICT (SYNC)

@router.post("/predict", response_model=InferenceResponse)
async def predict_unlabeled(file: UploadFile = File(...)):
    import time
    start_time = time.time()

    model, scaler, _ = model_manager.load_model(settings.default_model_id)

    df = read_csv_safe(await file.read())
    df.columns = df.columns.str.strip()

    validate_csv_headers(df, require_label=False)

    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    df, _ = preprocess_and_split_data(
        df,
        fit_scaler=False,
        scaler=scaler
    )

    dataset = FTGDataset(df, has_labels=False)
    engine = InferenceEngine(model, model_manager.device)

    traffic_graphs, flow_graphs = [], []
    total_slots = len(dataset)
    log_every = max(1, total_slots // 10)

    for idx, (tg, fg) in enumerate(dataset, start=1):
        traffic_graphs.append(tg)
        flow_graphs.append(fg)

        if idx % log_every == 0 or idx == total_slots:
            logger.info(
                "predicting: %d/%d slots (%.1f%%)",
                idx, total_slots, (idx / total_slots) * 100
            )

    batch = engine.predict_batch(traffic_graphs, flow_graphs)
    preds = batch["results"]

    attack_count = sum(
        1 for r in preds
        if (any(p == 1 for p in r["prediction"])
            if isinstance(r["prediction"], list)
            else r["prediction"] == 1)
    )

    avg_conf = np.mean([
        np.mean(r["probability"]) if isinstance(r["probability"], list)
        else r["probability"]
        for r in preds
    ])

    return InferenceResponse(
        total_samples=len(preds),
        attack_count=attack_count,
        benign_count=len(preds) - attack_count,
        average_confidence=float(avg_conf),
        processing_time_ms=(time.time() - start_time) * 1000
    )


# PREDICT (STREAM)

@router.post("/predict/stream")
async def start_stream_job(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    CANCEL_FLAGS[job_id] = False

    # Store file bytes in memory (simple + safe for now)
    file_bytes = await file.read()

    STREAM_JOBS[job_id] = {
        "file": file_bytes,
        "status": "started"
    }

    return {"job_id": job_id}


STREAM_JOBS: Dict[str, dict] = {}

@router.get("/predict/stream/{job_id}")
async def stream_job(job_id: str):

    if job_id not in STREAM_JOBS:
        raise HTTPException(status_code=404, detail="Invalid job id")

    async def event_generator():
        import time, asyncio
        start_time = time.time()

        yield f"data: {json.dumps({'stage': 'job_started', 'job_id': job_id})}\n\n"
        await asyncio.sleep(0)

        model, scaler, _ = model_manager.load_model(settings.default_model_id)
        engine = InferenceEngine(model, model_manager.device)

        yield f"data: {json.dumps({'stage': 'preprocessing'})}\n\n"
        await asyncio.sleep(0)

        df = read_csv_safe(STREAM_JOBS[job_id]["file"])
        df.columns = df.columns.str.strip()
        validate_csv_headers(df, require_label=False)

        df, _ = preprocess_and_split_data(df, fit_scaler=False, scaler=scaler)
        dataset = FTGDataset(df, has_labels=False)

        total = len(dataset)
        step = max(1, total // 10)

        traffic_graphs, flow_graphs = [], []

        for idx, (tg, fg) in enumerate(dataset, start=1):
            if CANCEL_FLAGS.get(job_id):
                yield f"data: {json.dumps({'stage': 'cancelled'})}\n\n"
                await asyncio.sleep(0)
                return

            traffic_graphs.append(tg)
            flow_graphs.append(fg)

            if idx % step == 0 or idx == total:
                yield f"data: {json.dumps({
                    'stage': 'progress',
                    'current': idx,
                    'total': total
                })}\n\n"
                await asyncio.sleep(0)

        batch = engine.predict_batch(traffic_graphs, flow_graphs)
        preds = batch["results"]

        attack_count = sum(
            1 for r in preds
            if (any(r["prediction"]) if isinstance(r["prediction"], list)
                else r["prediction"] == 1)
        )

        avg_conf = np.mean([
            np.mean(r["probability"]) if isinstance(r["probability"], list)
            else r["probability"]
            for r in preds
        ])

        yield f"data: {json.dumps({
            'stage': 'done',
            'total_samples': len(preds),
            'attack_count': attack_count,
            'benign_count': len(preds) - attack_count,
            'average_confidence': float(avg_conf),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        })}\n\n"
        await asyncio.sleep(0)

        STREAM_JOBS.pop(job_id, None)
        CANCEL_FLAGS.pop(job_id, None)

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )



# EVALUATE (SYNC)

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_labeled(file: UploadFile = File(...)):
    model, scaler, _ = model_manager.load_model(settings.default_model_id)

    df = read_csv_safe(await file.read())
    df.columns = df.columns.str.strip()
    validate_csv_headers(df, require_label=True)
    df, _ = preprocess_and_split_data(df, fit_scaler=False, scaler=scaler)

    dataset = FTGDataset(df, has_labels=True)
    engine = InferenceEngine(model, model_manager.device)

    preds, probs, labels = [], [], []

    for i in range(len(dataset)):
        tg, fg = dataset[i]
        r = engine.predict(tg, fg)

        if isinstance(r["prediction"], list):
            preds.append(1 if any(r["prediction"]) else 0)
            probs.append(float(np.mean(r["probability"])))
        else:
            preds.append(int(r["prediction"]))
            probs.append(float(r["probability"]))

        labels.append(int(tg.y.max().item()))

    cm = confusion_matrix(labels, preds)

    return EvaluationResponse(
        total_samples=len(preds),
        attack_count=int(np.sum(np.array(preds) == 1)),
        benign_count=int(np.sum(np.array(preds) == 0)),
        average_confidence=float(np.mean(probs)),
        accuracy=float(accuracy_score(labels, preds)),
        precision=float(precision_score(labels, preds, zero_division=0)),
        recall=float(recall_score(labels, preds, zero_division=0)),
        f1_score=float(f1_score(labels, preds, zero_division=0)),
        confusion_matrix={
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
    )


@router.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    CANCEL_FLAGS[job_id] = True
    logger.warning("Job cancelled: %s", job_id)
    return {"status": "cancelled", "job_id": job_id}


@router.get("/models")
async def list_models():
    """List available models"""
    return model_manager.list_models()


@router.get("/models/active")
async def active_model():
    """Get currently active model"""
    return model_manager.get_active_model()


@router.post("/models/{model_id}")
async def switch_model(model_id: str):
    """Switch active model (hot reload)"""
    try:
        model_manager.set_active_model(model_id)
        return {
            "status": "switched",
            "model": model_manager.get_active_model()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/health")
async def health():
    """
    Liveness check.
    If this responds, the service is UP.
    """
    return {
        "alive": True
    }

@router.get("/status")
async def status():
    """
    Readiness + system + model summary (merged)
    """
    try:
        # Ensure model is loadable
        model_manager.load_model(settings.default_model_id)
        summary = model_manager.get_model_summary()

        return {
            "ready": True,
            "device": str(model_manager.device),
            "cuda_available": torch.cuda.is_available(),

            "model": {
                "model_id": summary["model_id"],
                "name": summary["name"],
                "loaded": summary["loaded"],
                "feature_count": summary["architecture"]["num_features"],
                "hidden_size": summary["architecture"]["hidden_size"],
            }
        }

    except Exception as e:
        return {
            "ready": False,
            "error": str(e)
        }
