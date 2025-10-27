"""
Model inference endpoints
- /predict: Unlabeled data inference (production use)
- /evaluate: Labeled data evaluation (model testing)
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Optional
import io
import torch
from torch_geometric.data import Data

from app.core.ml.model_loader import model_manager
from app.core.ml.preprocessing import preprocess_and_split_data
from app.core.ml.inference import InferenceEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])


# ============================================================================
# RESPONSE MODELS
# ============================================================================

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


# ============================================================================
# DATASET CLASS (Shared)
# ============================================================================

class FTGDataset:
    """
    FTG Dataset for graph construction
    Works with or without labels
    """
    def __init__(self, df, time_slot_duration='5s', min_packets_per_flow=1, 
                 require_shared_ips=False, min_flows_per_slot=1, has_labels=True):
        """
        Args:
            df: Preprocessed dataframe
            time_slot_duration: Time window for grouping
            min_packets_per_flow: Minimum packets per flow
            require_shared_ips: If False, fully connected traffic graphs
            min_flows_per_slot: Minimum flows per time slot
            has_labels: Whether Label column exists
        """
        logger.info(f"Constructing FTG dataset (labels={'yes' if has_labels else 'no'})")
        logger.info(f"  - Time slot: {time_slot_duration}")
        logger.info(f"  - Min packets/flow: {min_packets_per_flow}")
        logger.info(f"  - Fully connected: {not require_shared_ips}")
        
        self.has_labels = has_labels
        df = df.copy()
        
        # ROBUST TIMESTAMP PARSING
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        except Exception as e:
            logger.warning(f"Timestamp parsing failed: {e}, trying formats...")
            for fmt in ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                try:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format=fmt, errors='coerce')
                    logger.info(f"✅ Parsed with format: {fmt}")
                    break
                except:
                    continue
        
        # Remove invalid timestamps
        invalid_count = df['Timestamp'].isna().sum()
        if invalid_count > 0:
            logger.warning(f"⚠️ Removing {invalid_count} invalid timestamps")
            df = df.dropna(subset=['Timestamp'])
        
        if df.empty:
            raise ValueError("No valid timestamps in dataset")
        
        df = df.set_index('Timestamp').sort_index()
        all_time_slots = [group for _, group in df.groupby(pd.Grouper(freq=time_slot_duration))]
        
        self.valid_time_slots = []
        self.feature_cols = ['Average Packet Size', 'Bwd Packets/s', 'FIN Flag Count',
                             'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                             'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Flow Packets/s']
        
        skipped = {'empty': 0, 'few_nodes': 0, 'no_edges': 0}
        
        for slot in all_time_slots:
            if slot.empty or len(slot) < min_packets_per_flow:
                skipped['empty'] += 1
                continue
            
            endpoint_groups = slot.groupby(['Source IP', 'Destination IP'])
            traffic_graph_nodes = [(src_ip, dst_ip) for (src_ip, dst_ip), group in endpoint_groups 
                                  if len(group) >= min_packets_per_flow]
            
            if len(traffic_graph_nodes) < min_flows_per_slot:
                skipped['few_nodes'] += 1
                continue
            
            if require_shared_ips:
                has_edges = any(
                    traffic_graph_nodes[i][0] == traffic_graph_nodes[j][0] or 
                    traffic_graph_nodes[i][1] == traffic_graph_nodes[j][1]
                    for i in range(len(traffic_graph_nodes))
                    for j in range(i + 1, len(traffic_graph_nodes))
                )
                if not has_edges:
                    skipped['no_edges'] += 1
                    continue
            
            self.valid_time_slots.append(slot)
        
        logger.info(f"✅ Created {len(self.valid_time_slots)} valid time slots")
        logger.info(f"Skipped: {skipped}")
    
    def __len__(self) -> int:
        return len(self.valid_time_slots)
    
    def __getitem__(self, idx: int) -> Tuple[Data, List[Data]]:
        """Get graph sample"""
        slot = self.valid_time_slots[idx]
        endpoint_groups = slot.groupby(['Source IP', 'Destination IP'])
        flow_graphs, node_map = [], {}
        
        for flow_idx, ((src_ip, dst_ip), group) in enumerate(endpoint_groups):
            node_map[(src_ip, dst_ip)] = len(node_map)
            node_features = torch.tensor(group[self.feature_cols].values, dtype=torch.float)
            
            # Sequential edges
            if len(node_features) > 1:
                edge_list = [[i, i+1] for i in range(len(node_features) - 1)]
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # Label (use if available, else dummy)
            if self.has_labels and 'Label' in group.columns:
                label = torch.tensor([group['Label'].max()], dtype=torch.float)
            else:
                label = torch.tensor([1.0], dtype=torch.float)  # Dummy
            
            flow_graphs.append(Data(x=node_features, edge_index=edge_index, y=label))
        
        if not flow_graphs:
            raise ValueError(f"No flows in time slot {idx}")
        
        # Traffic graph
        traffic_edge_list = []
        nodes = list(node_map.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                traffic_edge_list.append([i, j])
                traffic_edge_list.append([j, i])
        
        if not traffic_edge_list and len(nodes) == 1:
            traffic_edge_list = [[0, 0]]  # Self-loop
        
        traffic_edge_index = torch.tensor(traffic_edge_list, dtype=torch.long).t().contiguous()
        traffic_graph = Data(
            x=torch.empty((len(flow_graphs), len(self.feature_cols))), 
            edge_index=traffic_edge_index, 
            y=torch.cat([fg.y for fg in flow_graphs])
        )
        
        return traffic_graph, flow_graphs


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/predict", response_model=InferenceResponse)
async def predict_unlabeled(file: UploadFile = File(...)):
    """
    **Real-time inference on UNLABELED data** (production use)
    
    Upload CSV without Label column for DDoS detection.
    Returns attack/benign predictions.
    
    **Required columns:**
    - Source IP, Destination IP, Timestamp
    - Average Packet Size, Bwd Packets/s
    - FIN/SYN/RST/PSH/ACK/URG/CWE/ECE Flag Counts
    - Flow Packets/s
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"📂 Inference request: {file.filename}")
        
        # Load model
        model, scaler, hyperparams = model_manager.load_model(
            "models/checkpoints_v4_metadata/best_model_1.pt"
        )
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), low_memory=False)
        df.columns = df.columns.str.strip()
        
        logger.info(f"✅ Loaded {len(df):,} samples")
        
        # Preprocess (Label optional)
        df_processed, _ = preprocess_and_split_data(df, fit_scaler=False, scaler=scaler)
        
        # Create dataset (no labels)
        dataset = FTGDataset(
            df_processed, 
            time_slot_duration='5s',
            min_packets_per_flow=1,
            require_shared_ips=False,
            min_flows_per_slot=1,
            has_labels=False  # No labels
        )
        
        if len(dataset) == 0:
            raise HTTPException(status_code=400, detail="No valid graphs created")
        
        # Inference
        engine = InferenceEngine(model, model_manager.device)
        predictions, probabilities = [], []
        
        for i in range(len(dataset)):
            try:
                traffic_graph, flow_graphs = dataset[i]
                if not flow_graphs:
                    continue
                result = engine.predict(traffic_graph, flow_graphs)
                predictions.append(result['prediction'])
                probabilities.append(result['probability'])
            except Exception as e:
                logger.debug(f"Graph {i} error: {str(e)[:100]}")
                continue
        
        if not predictions:
            raise HTTPException(status_code=400, detail="No valid predictions generated")
        
        predictions = np.array(predictions, dtype=int)
        probabilities = np.array(probabilities, dtype=float)
        
        attack_count = int((predictions == 1).sum())
        benign_count = int((predictions == 0).sum())
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Inference: {attack_count} attacks, {benign_count} benign ({processing_time:.0f}ms)")
        
        return InferenceResponse(
            total_samples=len(predictions),
            attack_count=attack_count,
            benign_count=benign_count,
            average_confidence=float(np.mean(probabilities)),
            processing_time_ms=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)[:200]}")


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_labeled(file: UploadFile = File(...)):
    """
    **Model evaluation on LABELED data** (testing/validation)
    
    Upload CSV **with Label column** to evaluate model performance.
    Returns predictions + evaluation metrics (accuracy, precision, recall, F1).
    
    **Required columns:**
    - All columns from /predict endpoint
    - **Label** (BENIGN or attack type)
    """
    try:
        logger.info(f"📊 Evaluation request: {file.filename}")
        
        # Load model
        model, scaler, hyperparams = model_manager.load_model(
            "models/checkpoints_v4_metadata/best_model_1.pt"
        )
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), low_memory=False)
        df.columns = df.columns.str.strip()
        
        # Verify Label exists
        if 'Label' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail="Label column required for evaluation. Use /predict for unlabeled data."
            )
        
        logger.info(f"✅ Loaded {len(df):,} labeled samples")
        
        # Preprocess
        df_processed, _ = preprocess_and_split_data(df, fit_scaler=False, scaler=scaler)
        
        # Create dataset (with labels)
        dataset = FTGDataset(
            df_processed, 
            time_slot_duration='5s',
            min_packets_per_flow=1,
            require_shared_ips=False,
            min_flows_per_slot=1,
            has_labels=True  # Has labels
        )
        
        if len(dataset) == 0:
            raise HTTPException(status_code=400, detail="No valid graphs created")
        
        # Inference + label extraction
        engine = InferenceEngine(model, model_manager.device)
        predictions, probabilities, true_labels = [], [], []
        
        for i in range(len(dataset)):
            try:
                traffic_graph, flow_graphs = dataset[i]
                if not flow_graphs:
                    continue
                result = engine.predict(traffic_graph, flow_graphs)
                predictions.append(result['prediction'])
                probabilities.append(result['probability'])
                # Extract true labels
                true_labels.extend(traffic_graph.y.cpu().numpy().tolist())
            except Exception as e:
                logger.debug(f"Graph {i} error: {str(e)[:100]}")
                continue
        
        if not predictions:
            raise HTTPException(status_code=400, detail="No valid predictions generated")
        
        predictions = np.array(predictions, dtype=int)
        probabilities = np.array(probabilities, dtype=float)
        true_labels = np.array(true_labels, dtype=int)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = float(accuracy_score(true_labels, predictions))
        precision = float(precision_score(true_labels, predictions, zero_division=0))
        recall = float(recall_score(true_labels, predictions, zero_division=0))
        f1 = float(f1_score(true_labels, predictions, zero_division=0))
        
        cm = confusion_matrix(true_labels, predictions)
        confusion = {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1])
        }
        
        attack_count = int((predictions == 1).sum())
        benign_count = int((predictions == 0).sum())
        
        logger.info(f"✅ Evaluation: Acc={accuracy:.3f}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        return EvaluationResponse(
            total_samples=len(predictions),
            attack_count=attack_count,
            benign_count=benign_count,
            average_confidence=float(np.mean(probabilities)),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=confusion
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)[:200]}")


@router.get("/health")
async def health_check():
    """Health check"""
    try:
        model_manager.load_model("models/checkpoints_v4_metadata/best_model_1.pt")
        return {"status": "✅ healthy", "model": "FTG-NET v1", "device": str(model_manager.device)}
    except Exception as e:
        return {"status": "❌ unhealthy", "error": str(e)}


@router.get("/stats")
async def stats():
    """Model statistics"""
    try:
        model, scaler, hyperparams = model_manager.load_model(
            "models/checkpoints_v4_metadata/best_model_1.pt"
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
