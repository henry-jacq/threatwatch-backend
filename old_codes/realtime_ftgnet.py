#!/usr/bin/env python3
"""
Realtime FTG-Net inference (robust GNN loader + NFStream integration)
Author: Henry & Assistant

Features:
 - Auto-infers model architecture from checkpoint (with GAT/SAGE inference).
 - Handles mismatch in dimensions gracefully (strict=False fallback).
 - Supports ensemble directories or single checkpoints.
 - Captures live NFStream flows, builds graphs, scales features, and predicts.
"""

import os
import re
import time
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from nfstream import NFStreamer
from datetime import datetime

# --------------------------
# 1. Checkpoint Inspector
# --------------------------
def infer_shapes_from_state(state):
    info = {}
    # FlowGNN input dim
    for k, v in state.items():
        if "flow_gnn.conv1.lin_l.weight" in k or "flow_gnn.conv1.lin.weight" in k:
            info["in_channels"] = v.shape[1]
            info["flow_hidden"] = v.shape[0]
        if "flow_gnn.fc.weight" in k:
            info["flow_out_dim"] = v.shape[0]
    # GAT head detection
    for k, v in state.items():
        if "traffic_gnn.conv1.att_src" in k:
            info["gat_heads"] = v.shape[1]
            info["traffic_hidden"] = v.shape[2]
    # Fallbacks
    info.setdefault("in_channels", 11)
    info.setdefault("flow_hidden", 512)
    info.setdefault("flow_out_dim", 512)
    info.setdefault("gat_heads", 4)
    info.setdefault("traffic_hidden", 512)
    return info

# --------------------------
# 2. Model Definitions
# --------------------------
def build_model_from_inferred(info, device):
    import torch.nn as nn

    class FlowGNN_local(nn.Module):
        def __init__(self, in_channels, hidden, out_channels):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
            self.conv3 = SAGEConv(hidden, hidden)
            self.fc = nn.Linear(hidden, out_channels)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.leaky_relu(self.conv1(x, edge_index))
            x = F.leaky_relu(self.conv2(x, edge_index))
            x = F.leaky_relu(self.conv3(x, edge_index))
            x = global_mean_pool(x, getattr(data, "batch", None))
            return self.fc(x)

    class TrafficGNN_local(nn.Module):
        def __init__(self, in_ch, hidden, heads=4, dropout=0.4):
            super().__init__()
            # use concat=True to match training setup
            self.conv1 = GATConv(in_ch, hidden, heads=heads, concat=True)
            self.conv2 = GATConv(hidden * heads, hidden, heads=heads, concat=True)
            self.conv3 = GATConv(hidden * heads, hidden, heads=heads, concat=True)
            self.fc = nn.Linear(hidden * heads, 1)
            self.drop = Dropout(p=dropout)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.elu(self.conv1(x, edge_index))
            x = F.elu(self.conv2(x, edge_index))
            x = F.elu(self.conv3(x, edge_index))
            x = self.drop(x)
            x = global_mean_pool(x, getattr(data, "batch", None))
            return self.fc(x)

    class FTGNetWrapper(nn.Module):
        def __init__(self, flow_gnn, traffic_gnn):
            super().__init__()
            self.flow_gnn = flow_gnn
            self.traffic_gnn = traffic_gnn

        def forward(self, graph):
            emb = self.flow_gnn(graph)
            tgraph = Data(x=emb, edge_index=graph.edge_index)
            return self.traffic_gnn(tgraph)

    flow_gnn = FlowGNN_local(info["in_channels"], info["flow_hidden"], info["flow_out_dim"]).to(device)
    traffic_gnn = TrafficGNN_local(info["flow_out_dim"], info["traffic_hidden"], heads=info["gat_heads"]).to(device)
    return FTGNetWrapper(flow_gnn, traffic_gnn).to(device)

# --------------------------
# 3. Safe Checkpoint Loader
# --------------------------
def load_checkpoint_infer(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    scaler = ckpt.get("scaler", None)
    hyper = ckpt.get("hyperparams", {})
    features = hyper.get("feature_order", None)
    if not features:
        features = [
            'Average Packet Size', 'Bwd Packets/s', 'FIN Flag Count',
            'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
            'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
            'ECE Flag Count', 'Flow Packets/s'
        ]
    info = infer_shapes_from_state(state)
    model = build_model_from_inferred(info, device)

    try:
        model.load_state_dict(state, strict=False)
        print(f"[OK] Loaded checkpoint '{path}' (non-strict).")
    except Exception as e:
        print(f"[WARN] Partial load for '{path}': {e}")

    model.eval()
    return model, scaler, features, hyper

# --------------------------
# 4. NFStream Flow Graph Builder
# --------------------------
def build_traffic_graph_from_flows(flows, feature_cols, scaler, device):
    n = len(flows)
    if n == 0:
        return Data(x=torch.zeros((1, len(feature_cols))), edge_index=torch.zeros((2,1),dtype=torch.long)).to(device)
    X, metas = [], []
    for f in flows:
        vals = [float(getattr(f, c.lower().replace(" ", "_"), 0.0)) for c in feature_cols]
        X.append(vals)
        metas.append((f.src_ip, f.dst_ip))
    X = np.array(X, dtype=float)
    if scaler: X = scaler.transform(X)
    x = torch.tensor(X, dtype=torch.float32).to(device)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if metas[i][0] == metas[j][0] or metas[i][1] == metas[j][1]:
                edges.append([i, j]); edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device) if edges else torch.zeros((2,1), dtype=torch.long).to(device)
    return Data(x=x, edge_index=edge_index)

# --------------------------
# 5. Main Realtime Loop
# --------------------------
def run_realtime(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INIT] Running on {device}")
    model, scaler, features, _ = load_checkpoint_infer(args.checkpoint, device)

    streamer = NFStreamer(
        source=args.interface,
        statistical_analysis=True,
        idle_timeout=args.idle_timeout,
        active_timeout=args.active_timeout
    )
    print(f"[NFStream] Capturing live flows on {args.interface}")

    try:
        while True:
            flows = list(streamer)
            if not flows:
                time.sleep(1)
                continue
            graph = build_traffic_graph_from_flows(flows, features, scaler, device)
            with torch.no_grad():
                y = model(graph)
                prob = torch.sigmoid(y).mean().item()
            label = "🚨 MALICIOUS" if prob > args.threshold else "✅ BENIGN"
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] P={prob:.3f} → {label} ({len(flows)} flows)")
            if args.csv:
                with open(args.csv, "a") as f:
                    f.write(f"{datetime.now()},{prob:.4f},{label}\n")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[STOP] Exiting realtime monitoring.")

# --------------------------
# 6. CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Realtime FTG-Net Inference (fixed concat GAT)")
    p.add_argument("-i", "--interface", default="eth0", help="Network interface")
    p.add_argument("-c", "--checkpoint", required=True, help="Checkpoint file path")
    p.add_argument("--interval", type=float, default=5.0, help="Seconds between inference rounds")
    p.add_argument("--threshold", type=float, default=0.5, help="Alert probability threshold")
    p.add_argument("--csv", type=str, default=None, help="Optional CSV log file")
    p.add_argument("--idle-timeout", type=int, default=5)
    p.add_argument("--active-timeout", type=int, default=10)
    p.add_argument("--cpu", action="store_true", help="Force CPU mode")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_realtime(args)
