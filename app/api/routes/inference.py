"""
Model inference endpoints - FIXED for sparse attack data with proper error handling
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple
import io
import torch
from torch_geometric.data import Data

from app.core.ml.model_loader import model_manager
from app.core.ml.preprocessing import preprocess_and_split_data
from app.core.ml.inference import InferenceEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    is_attack: bool
    inference_time_ms: float
    label: str


class BatchPredictionResponse(BaseModel):
    total_samples: int
    attack_count: int
    benign_count: int
    average_confidence: float


class FTGDataset:
    """
    Relaxed FTG Dataset for sparse attack-only traffic
    - Allows single-packet flows
    - Uses fully-connected traffic graphs
    - Better handles sparse time slots
    """
    def __init__(self, df, time_slot_duration='5s', min_packets_per_flow=1, 
                 require_shared_ips=False, min_flows_per_slot=1):
        """
        Args:
            df: Preprocessed dataframe
            time_slot_duration: Time window for grouping
            min_packets_per_flow: Minimum packets per flow
            require_shared_ips: If False, fully connected traffic graphs
            min_flows_per_slot: Minimum flows per time slot
        """
        logger.info(f"Constructing FTG dataset")
        logger.info(f"  - Time slot: {time_slot_duration}")
        logger.info(f"  - Min packets/flow: {min_packets_per_flow}")
        logger.info(f"  - Fully connected: {not require_shared_ips}")
        
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp').sort_index()
        all_time_slots = [group for _, group in df.groupby(pd.Grouper(freq=time_slot_duration))]
        
        self.valid_time_slots = []
        self.feature_cols = ['Average Packet Size', 'Bwd Packets/s', 'FIN Flag Count',
                             'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                             'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Flow Packets/s']
        
        skipped = {'empty': 0, 'few_nodes': 0, 'no_edges': 0, 'few_packets': 0}
        
        for slot in all_time_slots:
            if slot.empty or len(slot) < min_packets_per_flow:
                skipped['empty'] += 1
                continue
            
            endpoint_groups = slot.groupby(['Source IP', 'Destination IP'])
            traffic_graph_nodes = []
            
            for (src_ip, dst_ip), group in endpoint_groups:
                if len(group) >= min_packets_per_flow:
                    traffic_graph_nodes.append((src_ip, dst_ip))
            
            if len(traffic_graph_nodes) < min_flows_per_slot:
                skipped['few_nodes'] += 1
                continue
            
            if require_shared_ips:
                has_edges = False
                for i in range(len(traffic_graph_nodes)):
                    for j in range(i + 1, len(traffic_graph_nodes)):
                        if (traffic_graph_nodes[i][0] == traffic_graph_nodes[j][0] or 
                            traffic_graph_nodes[i][1] == traffic_graph_nodes[j][1]):
                            has_edges = True
                            break
                    if has_edges:
                        break
                
                if not has_edges:
                    skipped['no_edges'] += 1
                    continue
            
            self.valid_time_slots.append(slot)
        
        logger.info(f"✅ Created {len(self.valid_time_slots)} valid time slots")
        logger.info(f"Skipped: {skipped}")
    
    def __len__(self) -> int:
        return len(self.valid_time_slots)
    
    def __getitem__(self, idx: int) -> Tuple[Data, List[Data]]:
        """Get graph sample at index - FIXED"""
        slot = self.valid_time_slots[idx]
        endpoint_groups = slot.groupby(['Source IP', 'Destination IP'])
        flow_graphs, node_map = [], {}
        
        # CRITICAL FIX: Don't use 'i' as packet index, use flow index
        for flow_idx, ((src_ip, dst_ip), group) in enumerate(endpoint_groups):
            node_map[(src_ip, dst_ip)] = len(node_map)
            node_features = torch.tensor(group[self.feature_cols].values, dtype=torch.float)
            
            # Sequential edges within flow (packet temporal ordering)
            if len(node_features) > 1:
                # Create edges between consecutive packets: 0->1, 1->2, etc.
                edge_list = [[packet_idx, packet_idx+1] for packet_idx in range(len(node_features) - 1)]
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            label = torch.tensor([group['Label'].max()], dtype=torch.float)
            flow_graphs.append(Data(x=node_features, edge_index=edge_index, y=label))
        
        if len(flow_graphs) == 0:
            logger.error(f"No flow graphs created for time slot {idx}")
            raise ValueError(f"No flows in time slot {idx}")
        
        # Traffic graph: fully connected between flows
        traffic_edge_list = []
        nodes = list(node_map.keys())
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                traffic_edge_list.append([i, j])
                traffic_edge_list.append([j, i])
        
        if len(traffic_edge_list) == 0:
            # Even single flow needs self-loop for GAT
            traffic_edge_list = [[0, 0]] if len(nodes) == 1 else []
        
        traffic_edge_index = torch.tensor(traffic_edge_list, dtype=torch.long).t().contiguous()
        
        traffic_graph = Data(
            x=torch.empty((len(flow_graphs), len(self.feature_cols))), 
            edge_index=traffic_edge_index, 
            y=torch.cat([fg.y for fg in flow_graphs])
        )
        
        return traffic_graph, flow_graphs


@router.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)) -> BatchPredictionResponse:
    """Real-time batch inference on uploaded CSV"""
    try:
        logger.info(f"📂 Processing file: {file.filename}")
        
        # Load model
        model, scaler, hyperparams = model_manager.load_model(
            "models/checkpoints_v4_metadata/best_model_1.pt"
        )
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), low_memory=False)
        df.columns = df.columns.str.strip()
        
        logger.info(f"✅ Loaded {len(df):,} samples")
        
        # Preprocess
        df_processed, _ = preprocess_and_split_data(df, fit_scaler=False, scaler=scaler)
        
        # Create dataset
        logger.info("🔧 Creating FTG dataset with RELAXED constraints")
        dataset = FTGDataset(
            df_processed, 
            time_slot_duration='5s',
            min_packets_per_flow=1,
            require_shared_ips=False,
            min_flows_per_slot=1
        )
        
        if len(dataset) == 0:
            logger.warning("⚠️ No valid graphs with 5s, trying 10s")
            dataset = FTGDataset(df_processed, time_slot_duration='10s', 
                               min_packets_per_flow=1, require_shared_ips=False, min_flows_per_slot=1)
        
        if len(dataset) == 0:
            raise HTTPException(status_code=400, detail="No valid graphs created")
        
        logger.info(f"Created {len(dataset)} graphs for inference")
        
        # Inference
        engine = InferenceEngine(model, model_manager.device)
        predictions = []
        probabilities = []
        successful = 0
        errors_count = 0
        
        for i in range(len(dataset)):
            try:
                traffic_graph, flow_graphs = dataset[i]
                
                # CRITICAL: Verify flow_graphs is not empty
                if not flow_graphs or len(flow_graphs) == 0:
                    logger.debug(f"Graph {i}: no flow_graphs")
                    errors_count += 1
                    continue
                
                result = engine.predict(traffic_graph, flow_graphs)
                predictions.append(result['prediction'])
                probabilities.append(result['probability'])
                successful += 1
                
                if (i + 1) % max(1, len(dataset) // 10) == 0:
                    logger.info(f"  ✓ Processed {i+1}/{len(dataset)} (Success: {successful}, Errors: {errors_count})")
                    
            except Exception as e:
                logger.debug(f"Graph {i} error: {str(e)[:100]}")
                errors_count += 1
                continue
        
        if not predictions:
            raise HTTPException(
                status_code=400, 
                detail=f"No valid predictions (0/{len(dataset)} successful, {errors_count} errors)"
            )
        
        predictions = np.array(predictions, dtype=int)
        probabilities = np.array(probabilities, dtype=float)
        
        attack_count = int((predictions == 1).sum())
        benign_count = int((predictions == 0).sum())
        
        logger.info(f"✅ Inference complete: {successful}/{len(dataset)} successful")
        logger.info(f"  Attack: {attack_count}, Benign: {benign_count}, Errors: {errors_count}")
        
        return BatchPredictionResponse(
            total_samples=len(predictions),
            attack_count=attack_count,
            benign_count=benign_count,
            average_confidence=float(np.mean(probabilities))
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)[:200]}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model_manager.load_model("models/checkpoints_v4_metadata/best_model_1.pt")
        return {
            "status": "✅ healthy",
            "model": "FTG-NET v1",
            "device": str(model_manager.device)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "❌ unhealthy", "error": str(e)}


@router.get("/stats")
async def stats():
    """Get model statistics"""
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
