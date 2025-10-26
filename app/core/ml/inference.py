"""
Production FTG-NET Inference Engine
Real-time DDoS detection with batching support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
import numpy as np
from torch_geometric.nn import SAGEConv, GATConv, global_max_pool, global_mean_pool
from torch_geometric.data import Data, Batch
import logging
from typing import List, Tuple, Dict
import time

logger = logging.getLogger(__name__)


class FlowGNN(nn.Module):
    """Flow Graph Neural Network"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, pool: str = 'max'):
        super(FlowGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = Linear(hidden_channels, out_channels)
        self.pool = pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))
        if self.pool == 'max':
            x = global_max_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


class TrafficGNN(nn.Module):
    """Traffic Graph Neural Network"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 1, 
                 heads: int = 4, dropout_p: float = 0.5):
        super(TrafficGNN, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.fc = Linear(hidden_channels, out_channels)
        self.dropout = Dropout(p=dropout_p)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.gelu(self.conv1(x, edge_index))
        x = F.gelu(self.conv2(x, edge_index))
        x = F.gelu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        return x


class FTGNet(nn.Module):
    """Complete FTG-NET Model for DDoS Detection"""
    def __init__(self, flow_gnn: FlowGNN, traffic_gnn: TrafficGNN, device=None):
        super(FTGNet, self).__init__()
        self.flow_gnn = flow_gnn
        self.traffic_gnn = traffic_gnn
        self.device = device or next(flow_gnn.parameters()).device

    def forward(self, traffic_graph, flow_graphs: List[Data]) -> torch.Tensor:
        """
        Args:
            traffic_graph: Data object representing traffic graph
            flow_graphs: List of Data objects for individual flows
        
        Returns:
            Predictions tensor
        """
        if len(flow_graphs) == 0:
            raise ValueError("flow_graphs cannot be empty")
        
        flow_batch = Batch.from_data_list(flow_graphs).to(self.device)
        flow_embeddings = self.flow_gnn(flow_batch)
        
        traffic_graph = traffic_graph.to(self.device)
        
        # Align embeddings with traffic graph nodes
        if traffic_graph.x is None or traffic_graph.x.size(0) != flow_embeddings.size(0):
            n_nodes = traffic_graph.num_nodes
            n_emb = flow_embeddings.size(0)
            
            if n_nodes == n_emb:
                traffic_graph.x = flow_embeddings
            elif n_nodes > n_emb:
                repeats = (n_nodes + n_emb - 1) // n_emb
                traffic_graph.x = flow_embeddings.repeat(repeats, 1)[:n_nodes]
            else:
                traffic_graph.x = flow_embeddings[:n_nodes]
        else:
            traffic_graph.x = flow_embeddings
        
        predictions = self.traffic_gnn(traffic_graph)
        return predictions


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
            probs = output.squeeze().cpu().numpy()
        
        preds = (probs > threshold).astype(int) if isinstance(probs, np.ndarray) else int(probs > threshold)
        inference_time = time.time() - start_time
        
        return {
            'prediction': int(preds) if isinstance(preds, (int, np.integer)) else preds.tolist(),
            'probability': float(probs) if isinstance(probs, (int, float, np.number)) else probs.tolist(),
            'is_attack': bool(preds),
            'inference_time_ms': inference_time * 1000,
            'timestamp': time.time()
        }
