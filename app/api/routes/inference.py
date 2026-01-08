"""
Model inference endpoints
- /predict: Unlabeled data inference (production use)
- /evaluate: Labeled data evaluation (model testing)
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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


# ------------------------------------------------------------------
# PREDICT (SYNC)
# ------------------------------------------------------------------

@router.post("/predict", response_model=InferenceResponse)
async def predict_unlabeled(file: UploadFile = File(...)):
    import time
    start_time = time.time()

    model, scaler, _ = model_manager.load_model(settings.model_checkpoint)

    df = read_csv_safe(await file.read())
    df.columns = df.columns.str.strip()
    validate_csv_headers(df, require_label=False)

    df, _ = preprocess_and_split_data(df, fit_scaler=False, scaler=scaler)

    dataset = FTGDataset(df, has_labels=False)
    engine = InferenceEngine(model, model_manager.device)

    traffic_graphs, flow_graphs = [], []
    total_slots = len(dataset)
    log_every = max(1, total_slots // 10)

    for i in range(total_slots):
        tg, fg = dataset[i]
        traffic_graphs.append(tg)
        flow_graphs.append(fg)

        if (i + 1) % log_every == 0 or (i + 1) == total_slots:
            logger.info(
                "predicting: %d/%d slots (%.1f%%)",
                i + 1, total_slots, (i + 1) / total_slots * 100
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

    logger.info(
        "✅ /predict done | slots=%d | attacks=%d | avg_conf=%.4f | %.2f ms",
        len(preds), attack_count, avg_conf,
        (time.time() - start_time) * 1000
    )

    return InferenceResponse(
        total_samples=len(preds),
        attack_count=attack_count,
        benign_count=len(preds) - attack_count,
        average_confidence=float(avg_conf),
        processing_time_ms=(time.time() - start_time) * 1000
    )


# ------------------------------------------------------------------
# PREDICT (STREAM)
# ------------------------------------------------------------------

@router.post("/predict/stream")
async def predict_stream(file: UploadFile = File(...)):
    async def event_generator():
        import time
        start_time = time.time()

        job_id = str(uuid.uuid4())
        CANCEL_FLAGS[job_id] = False

        yield f"data: {json.dumps({'stage': 'job_started', 'job_id': job_id})}\n\n"

        model, scaler, _ = model_manager.load_model(settings.model_checkpoint)
        engine = InferenceEngine(model, model_manager.device)

        yield f"data: {json.dumps({'stage': 'preprocessing'})}\n\n"

        df = read_csv_safe(await file.read())
        df.columns = df.columns.str.strip()
        validate_csv_headers(df, require_label=False)
        df, _ = preprocess_and_split_data(df, fit_scaler=False, scaler=scaler)

        dataset = FTGDataset(df, has_labels=False)
        total = len(dataset)
        step = max(1, total // 10)

        traffic_graphs, flow_graphs = [], []

        for i in range(total):
            if CANCEL_FLAGS.get(job_id):
                yield f"data: {json.dumps({'stage': 'cancelled', 'job_id': job_id})}\n\n"
                CANCEL_FLAGS.pop(job_id, None)
                return

            tg, fg = dataset[i]
            traffic_graphs.append(tg)
            flow_graphs.append(fg)

            if (i + 1) % step == 0 or (i + 1) == total:
                yield f"data: {json.dumps({
                    'stage': 'progress',
                    'current': i + 1,
                    'total': total
                })}\n\n"

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

        yield f"data: {json.dumps({
            'stage': 'done',
            'total_samples': len(preds),
            'attack_count': attack_count,
            'benign_count': len(preds) - attack_count,
            'average_confidence': float(avg_conf),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        })}\n\n"

        CANCEL_FLAGS.pop(job_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ------------------------------------------------------------------
# EVALUATE (SYNC)
# ------------------------------------------------------------------

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_labeled(file: UploadFile = File(...)):
    model, scaler, _ = model_manager.load_model(settings.model_checkpoint)

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


# ------------------------------------------------------------------
# CANCEL
# ------------------------------------------------------------------

@router.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    CANCEL_FLAGS[job_id] = True
    logger.warning("❌ Job cancelled: %s", job_id)
    return {"status": "cancelled", "job_id": job_id}


@router.get("/health")
async def health_check():
    """Health check"""
    try:
        model_manager.load_model(settings.model_checkpoint)
        return {"status": "Healthy", "model": "FTG-NET v1", "device": str(model_manager.device)}
    except Exception as e:
        return {"status": "Unhealthy", "error": str(e)}


@router.get("/stats")
async def stats():
    """Model statistics"""
    try:
        model, scaler, hyperparams = model_manager.load_model(
            settings.model_checkpoint
        )
        return {
            "model": "FTG-NET",
            "version": "1.0",
            "features": len(hyperparams['feature_order']),
            "feature_list": hyperparams['feature_order'],
            "hidden_size": hyperparams['hidden_size'],
            "device": str(model_manager.device)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
