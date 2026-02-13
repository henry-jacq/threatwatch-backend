"""
Production FTG-NET Inference Engine
Real-time DDoS detection with batching support
"""
import torch
import numpy as np
from torch_geometric.data import Data, Batch
import logging
from typing import List, Dict
import time
from app.core.ml.models import FlowGNN, TrafficGNN, FTGNet

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Production-ready inference engine with caching and timing"""
    
    def __init__(self, model: FTGNet, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.eval()
        logger.info("InferenceEngine initialized")
    
    def predict(self, traffic_graph: Data, flow_graphs: List[Data], 
                threshold: float = 0.5) -> Dict:
        """
        Real-time inference
        
        Args:
            traffic_graph: Traffic graph Data object
            flow_graphs: List of flow graph Data objects
            threshold: Classification threshold
        
        Returns:
            Dict with predictions, probabilities, and timing
        """
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model(traffic_graph, flow_graphs)
            probs = np.atleast_1d(output.squeeze().cpu().numpy())
        
        inference_time = time.time() - start_time
        
        preds = (probs > threshold).astype(int)
        is_attack = bool(np.any(preds))
        avg_prob = float(np.mean(probs))

        return {
            "prediction": preds.tolist(),
            "probability": avg_prob,
            "is_attack": is_attack,
            "inference_time_ms": inference_time * 1000,
            "timestamp": time.time()
        }
        
    def predict_batch(
        self,
        traffic_graphs: list,
        flow_graphs_per_slot: list,
        threshold: float = 0.5
    ):
        """
        Batched inference across multiple time slots.
        """

        start_time = time.time()

        # Flatten flow graphs
        flat_flow_graphs = []
        flow_splits = []

        for fg in flow_graphs_per_slot:
            flow_splits.append(len(fg))
            flat_flow_graphs.extend(fg)

        if not flat_flow_graphs:
            raise ValueError("No flow graphs to run inference on")

        # Batch flow graphs
        flow_batch = Batch.from_data_list(flat_flow_graphs).to(self.device)

        # Batch traffic graphs
        traffic_batch = Batch.from_data_list(traffic_graphs).to(self.device)

        with torch.no_grad():
            outputs = self.model(traffic_batch, flat_flow_graphs)
            probs = np.atleast_1d(outputs.squeeze().cpu().numpy())

        expected = sum(flow_splits)
        if probs.shape[0] != expected:
            raise ValueError(f"Model output size mismatch: got={probs.shape[0]} expected={expected}")

        # Split predictions back per slot
        results = []
        idx = 0

        for count in flow_splits:
            slot_probs = probs[idx: idx + count]
            idx += count

            if count == 1:
                p = float(slot_probs[0])
                results.append({
                    "prediction": int(p > threshold),
                    "probability": p
                })
            else:
                preds = (slot_probs > threshold).astype(int).tolist()
                results.append({
                    "prediction": preds,
                    "probability": slot_probs.tolist()
                })

        return {
            "results": results,
            "batch_inference_time_ms": (time.time() - start_time) * 1000
        }
