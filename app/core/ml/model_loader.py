"""
Singleton model loader for efficient model management
"""
import torch
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple
from .models import FTGNet, FlowGNN, TrafficGNN
from app.core.ml.model_registry import MODEL_REGISTRY, DEFAULT_MODEL_ID

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton model manager"""
    _instance: Optional['ModelManager'] = None
    _model: Optional[FTGNet] = None
    _scaler: Optional[object] = None
    _hyperparams: Optional[dict] = None
    _device: Optional[torch.device] = None
    _active_model_id: str = DEFAULT_MODEL_ID

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_id: str, device: str = "auto"):
        if self._model is not None and getattr(self, "_model_id", None) == model_id:
            logger.info("✅ Model already loaded (%s), returning cached instance", model_id)
            return self._model, self._scaler, self._hyperparams

        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_id: {model_id}")

        checkpoint_path = MODEL_REGISTRY[model_id]["checkpoint"]
        self._model_id = model_id

        logger.info("🔄 Loading model [%s] from %s", model_id, checkpoint_path)

        # --- device ---
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)

        self._hyperparams = checkpoint["hyperparams"]
        self._scaler = checkpoint["scaler"]

        flow_gnn = FlowGNN(
            in_channels=len(self._hyperparams["feature_order"]),
            hidden_channels=self._hyperparams["hidden_size"],
            out_channels=self._hyperparams["hidden_size"]
        )
        traffic_gnn = TrafficGNN(
            in_channels=self._hyperparams["hidden_size"],
            hidden_channels=self._hyperparams["hidden_size"]
        )

        self._model = FTGNet(flow_gnn, traffic_gnn, device=self._device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()

        logger.info("✅ Model [%s] loaded on %s", model_id, self._device)
        return self._model, self._scaler, self._hyperparams

    @property
    def model(self) -> FTGNet:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    @property
    def scaler(self):
        if self._scaler is None:
            raise RuntimeError("Scaler not loaded. Call load_model() first.")
        return self._scaler

    @property
    def hyperparams(self) -> dict:
        if self._hyperparams is None:
            raise RuntimeError("Hyperparams not loaded. Call load_model() first.")
        return self._hyperparams

    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self):
        """Reset singleton for testing"""
        self._model = None
        self._scaler = None
        self._hyperparams = None
        self._device = None
        logger.info("ModelManager reset")
        
    def list_models(self):
        return {
            k: {
                "name": v["name"],
                "active": k == self._active_model_id
            }
            for k, v in MODEL_REGISTRY.items()
        }

    def set_active_model(self, model_id: str):
        if model_id not in MODEL_REGISTRY:
            raise ValueError("Invalid model_id")
        self.reset()
        self._active_model_id = model_id
        logger.warning("Switched active model to %s", model_id)

    def get_active_model(self):
        return {
            "model_id": self._active_model_id,
            "name": MODEL_REGISTRY[self._active_model_id]["name"]
        }


# Global singleton instance
model_manager = ModelManager()
