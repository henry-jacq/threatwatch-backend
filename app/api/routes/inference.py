"""
Model inference endpoints
- /predict: Unlabeled data inference (production use)
- /evaluate: Labeled data evaluation (model testing)
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import logging
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


# ENDPOINTS

@router.post("/predict", response_model=InferenceResponse)
async def predict_unlabeled(file: UploadFile = File(...)):
    import time
    start_time = time.time()

    model, scaler, _ = model_manager.load_model(settings.model_checkpoint)

    raw_bytes = await file.read()
    df = read_csv_safe(raw_bytes)
    df.columns = df.columns.str.strip()

    validate_csv_headers(df, require_label=False)

    df, _ = preprocess_and_split_data(df, fit_scaler=False, scaler=scaler)

    dataset = FTGDataset(df, has_labels=False)
    engine = InferenceEngine(model, model_manager.device)
    
    total_slots = len(dataset)
    log_every = max(1, total_slots // 10)

    traffic_graphs = []
    flow_graphs = []

    for i in range(len(dataset)):
        tg, fg = dataset[i]
        traffic_graphs.append(tg)
        flow_graphs.append(fg)
        
        if (i + 1) % log_every == 0 or (i + 1) == total_slots:
            logger.info(
                "predicting: %d/%d slots prepared (%.1f%%)",
                i + 1,
                total_slots,
                ((i + 1) / total_slots) * 100
            )

    batch_result = engine.predict_batch(traffic_graphs, flow_graphs)

    preds = batch_result["results"]

    attack_count = sum(
        1 for r in preds
        if (any(p == 1 for p in r["prediction"])
            if isinstance(r["prediction"], list)
            else r["prediction"] == 1)
    )

    benign_count = len(preds) - attack_count

    avg_conf = np.mean([
        np.mean(r["probability"]) if isinstance(r["probability"], list)
        else r["probability"]
        for r in preds
    ])

    logger.info(
        "Result: slots=%d, attacks=%d, benign=%d, avg_conf=%.4f, time_ms=%.2f",
        len(preds),
        attack_count,
        benign_count,
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


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_labeled(file: UploadFile = File(...)):
    model, scaler, _ = model_manager.load_model(settings.model_checkpoint)

    raw_bytes = await file.read()
    df = read_csv_safe(raw_bytes)
    df.columns = df.columns.str.strip()

    validate_csv_headers(df, require_label=True)

    df, _ = preprocess_and_split_data(df, fit_scaler=False, scaler=scaler)

    dataset = FTGDataset(df, has_labels=True)
    engine = InferenceEngine(model, model_manager.device)

    slot_preds = []
    slot_probs = []
    slot_labels = []

    for i in range(len(dataset)):
        tg, fg = dataset[i]
        r = engine.predict(tg, fg)

        # ---- SLOT PREDICTION ----
        pred = r["prediction"]
        prob = r["probability"]

        if isinstance(pred, list):
            slot_preds.append(1 if any(p == 1 for p in pred) else 0)
            slot_probs.append(float(np.mean(prob)))
        else:
            slot_preds.append(int(pred))
            slot_probs.append(float(prob))

        # ---- SLOT LABEL ----
        # tg.y contains flow labels → reduce to slot label
        slot_labels.append(int(tg.y.max().item()))

    slot_preds = np.array(slot_preds)
    slot_labels = np.array(slot_labels)

    cm = confusion_matrix(slot_labels, slot_preds)

    return EvaluationResponse(
        total_samples=len(slot_preds),                 # number of slots
        attack_count=int((slot_preds == 1).sum()),
        benign_count=int((slot_preds == 0).sum()),
        average_confidence=float(np.mean(slot_probs)),
        accuracy=float(accuracy_score(slot_labels, slot_preds)),
        precision=float(precision_score(slot_labels, slot_preds, zero_division=0)),
        recall=float(recall_score(slot_labels, slot_preds, zero_division=0)),
        f1_score=float(f1_score(slot_labels, slot_preds, zero_division=0)),
        confusion_matrix={
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
    )


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
