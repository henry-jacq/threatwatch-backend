#!/usr/bin/env python3
"""
Load and run inference using a single FTGNet model checkpoint.
"""

import os
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool, global_max_pool

# -------------------------
# Model definitions
# -------------------------
class FlowGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pool='max'):
        super().__init__()
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
        return self.fc(x)


class TrafficGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, heads=4, dropout_p=0.5):
        super().__init__()
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
        return torch.sigmoid(self.fc(x))


class FTGNet(torch.nn.Module):
    def __init__(self, flow_gnn: FlowGNN, traffic_gnn: TrafficGNN, device=None):
        super().__init__()
        self.flow_gnn = flow_gnn
        self.traffic_gnn = traffic_gnn
        self.device = device or torch.device("cpu")

    def forward(self, traffic_graph, flow_graphs):
        flow_batch = Batch.from_data_list(flow_graphs).to(self.device)
        flow_embeddings = self.flow_gnn(flow_batch)

        traffic_graph = traffic_graph.to(self.device)

        n_nodes = traffic_graph.num_nodes
        n_emb = flow_embeddings.size(0)
        if traffic_graph.x is None or traffic_graph.x.size(0) != n_nodes:
            if n_nodes == n_emb:
                traffic_graph.x = flow_embeddings
            elif n_nodes > n_emb:
                repeats = (n_nodes + n_emb - 1) // n_emb
                traffic_graph.x = flow_embeddings.repeat(repeats, 1)[:n_nodes]
            else:
                traffic_graph.x = flow_embeddings[:n_nodes]

        return self.traffic_gnn(traffic_graph)


# -------------------------
# Safe partial checkpoint loader
# -------------------------
def strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def load_checkpoint_partial(model, ckpt_path, verbose=True):
    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("model_state_dict", raw)
    state = strip_module_prefix(state)

    model_state = model.state_dict()
    copied, mismatched = [], []
    for k, v in state.items():
        if k in model_state and v.shape == model_state[k].shape:
            model_state[k] = v
            copied.append(k)
        else:
            mismatched.append(k)
    model.load_state_dict(model_state, strict=False)

    if verbose:
        print(f"✅ Loaded checkpoint: {ckpt_path}")
        print(f"  ✓ Copied: {len(copied)} layers")
        print(f"  ⚠️ Skipped (mismatch): {len(mismatched)} layers")
    return model


# -------------------------
# Dummy graph builder
# -------------------------
def create_sample_graph(num_nodes=8, num_features=11):
    x = torch.randn((num_nodes, num_features))
    edge_index = torch.tensor([[0,1,2,3,4,5,6,7],[1,0,3,2,5,4,7,6]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    ckpt_path = "checkpoints_v4_metadata/ensemble/model_0.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow_gnn = FlowGNN(in_channels=11, hidden_channels=512, out_channels=512)
    traffic_gnn = TrafficGNN(in_channels=512, hidden_channels=512)
    ftg_net = FTGNet(flow_gnn, traffic_gnn, device=device).to(device)

    ftg_net = load_checkpoint_partial(ftg_net, ckpt_path)
    ftg_net.eval()

    # Dummy input
    traffic_graph = create_sample_graph(num_nodes=8, num_features=512)
    flow_graphs = [create_sample_graph(num_nodes=8, num_features=11) for _ in range(3)]

    # Move everything to the same device
    traffic_graph = traffic_graph.to(device)
    for fg in flow_graphs:
        fg.to(device)

    print("🚀 Running single-model inference...")
    with torch.no_grad():
        out = ftg_net(traffic_graph, flow_graphs)
        prob = out.mean().item()
        pred = int(prob > 0.5)
        print(f"✅ Probability: {prob:.4f}")
        print(f"✅ Predicted class: {pred}")

    torch.save({"prob": prob, "pred": pred}, "single_inference_result.pt")
    print("💾 Saved result to single_inference_result.pt")
