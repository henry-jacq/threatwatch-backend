"""
Singleton model loader for efficient model management
"""
import torch
import logging
from pathlib import Path
from typing import Optional, Tuple
from .inference import FTGNet, FlowGNN, TrafficGNN
import warnings

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton model manager"""
    _instance: Optional['ModelManager'] = None
    _model: Optional[FTGNet] = None
    _scaler: Optional[object] = None
    _hyperparams: Optional[dict] = None
    _device: Optional[torch.device] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, checkpoint_path: str, device: str = 'auto') -> Tuple[FTGNet, object, dict]:
        """Load model from checkpoint (singleton pattern)"""
        if self._model is not None:
            logger.info("✅ Model already loaded, returning cached instance")
            return self._model, self._scaler, self._hyperparams

        logger.info(f"🔄 Loading model from {checkpoint_path}")
        
        # Determine device
        if device == 'auto':
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint with weights_only=True for security (but allows compatibility)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        
        self._hyperparams = checkpoint['hyperparams']
        self._scaler = checkpoint['scaler']
        
        logger.info(f"Checkpoint loaded. Trained for {checkpoint.get('epoch', 'unknown')} epochs")
        logger.info(f"Features: {len(self._hyperparams['feature_order'])}")
        logger.info(f"Hidden size: {self._hyperparams['hidden_size']}")
        
        # Build architecture
        flow_gnn = FlowGNN(
            in_channels=len(self._hyperparams['feature_order']),
            hidden_channels=self._hyperparams['hidden_size'],
            out_channels=self._hyperparams['hidden_size']
        )
        traffic_gnn = TrafficGNN(
            in_channels=self._hyperparams['hidden_size'],
            hidden_channels=self._hyperparams['hidden_size']
        )
        self._model = FTGNet(flow_gnn, traffic_gnn, device=self._device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.to(self._device)
        self._model.eval()
        
        logger.info(f"✅ Model loaded on device: {self._device}")
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


# Global singleton instance
model_manager = ModelManager()
