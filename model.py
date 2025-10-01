# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv,
    global_max_pool, global_mean_pool
)
from torch_geometric.data import Batch
import warnings
warnings.filterwarnings("ignore", message=".*pyg-lib.*")


def make_conv(conv_type, in_ch, out_ch, **kwargs):
    """Factory for GNN conv layers."""
    conv_type = conv_type.lower()
    if conv_type == "gcn":
        return GCNConv(in_ch, out_ch)
    elif conv_type == "sage":
        return SAGEConv(in_ch, out_ch)
    elif conv_type == "gat":
        heads = kwargs.get("heads", 4)
        concat = kwargs.get("concat", False)
        return GATConv(in_ch, out_ch, heads=heads, concat=concat)
    else:
        raise ValueError(f"Unknown conv_type={conv_type}")


class FlowGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 conv_type="sage", pool="max"):
        super().__init__()
        self.conv1 = make_conv(conv_type, in_channels, hidden_channels)
        self.conv2 = make_conv(conv_type, hidden_channels, hidden_channels)
        self.conv3 = make_conv(conv_type, hidden_channels, hidden_channels)
        self.fc = Linear(hidden_channels, out_channels)
        self.pool = pool

    def forward(self, data):
        if isinstance(data, (list, tuple)):
            x, edge_index, batch = data
        else:
            x, edge_index = data.x, data.edge_index
            batch = getattr(data, "batch", None)

        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))

        if self.pool == "max":
            x = global_max_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)

        return self.fc(x)


class TrafficGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1,
                 conv_type="gat", heads=4, dropout_p=0.5):
        super().__init__()
        self.conv1 = make_conv(conv_type, in_channels, hidden_channels, heads=heads)
        self.conv2 = make_conv(conv_type, hidden_channels, hidden_channels, heads=heads)
        self.conv3 = make_conv(conv_type, hidden_channels, hidden_channels, heads=heads)
        self.fc = Linear(hidden_channels, out_channels)
        self.dropout = Dropout(p=dropout_p)
        self.conv_type = conv_type

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        act = F.gelu if self.conv_type in ("gat", "sage") else F.relu

        x = act(self.conv1(x, edge_index))
        x = act(self.conv2(x, edge_index))
        x = act(self.conv3(x, edge_index))
        x = self.dropout(x)

        return self.fc(x)


class FTGNet(nn.Module):
    def __init__(self, flow_gnn: FlowGNN, traffic_gnn: TrafficGNN, device=None):
        super().__init__()
        self.flow_gnn = flow_gnn
        self.traffic_gnn = traffic_gnn
        self.device = device or next(flow_gnn.parameters()).device

    def forward(self, traffic_graph, flow_graphs):
        if not isinstance(flow_graphs, list):
            flow_graphs = [flow_graphs]

        flow_batch = Batch.from_data_list(flow_graphs).to(self.device)
        flow_embeddings = self.flow_gnn(flow_batch)

        traffic_graph = traffic_graph.to(self.device)
        n_nodes = traffic_graph.num_nodes
        n_emb = flow_embeddings.size(0)

        if traffic_graph.x is None or traffic_graph.x.size(0) != n_emb:
            if n_nodes == n_emb:
                traffic_graph.x = flow_embeddings
            elif n_nodes > n_emb:
                reps = (n_nodes + n_emb - 1) // n_emb
                traffic_graph.x = flow_embeddings.repeat(reps, 1)[:n_nodes]
            else:
                traffic_graph.x = flow_embeddings[:n_nodes]
        else:
            traffic_graph.x = flow_embeddings

        return self.traffic_gnn(traffic_graph)
