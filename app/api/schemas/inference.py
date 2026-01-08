from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    """Response for single prediction"""
    prediction: int
    probability: float
    is_attack: bool
    inference_time_ms: float
    label: str


class InferenceResponse(BaseModel):
    """Response for unlabeled data inference"""
    total_samples: int
    attack_count: int
    benign_count: int
    average_confidence: float
    processing_time_ms: float


class EvaluationResponse(BaseModel):
    """Response for labeled data evaluation"""
    total_samples: int
    attack_count: int
    benign_count: int
    average_confidence: float
    # Evaluation metrics (only when labels available)
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    confusion_matrix: Optional[dict] = None