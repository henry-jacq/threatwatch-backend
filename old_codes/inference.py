#!/usr/bin/env python3
"""
Realtime inference using FTG-Net architecture + NFStream
- Builds flow-level Data objects for FlowGNN (single-node flow graphs)
- Builds traffic-graph by connecting flows that share source OR destination IP (same logic as notebook)
- Loads checkpoint and optional scaler if present
- Runs inference per time-slot (TIME_SLOT_SECONDS), prints score + label
"""

import os
import glob
import math
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from nfstream import NFStreamer
from torch_geometric.data import Data, Batch
from collections import defaultdict

# ----------------------------
# Config
# ----------------------------
INTERFACE = "eth0"                     # change to your NIC or use a pcap file path e.g. "capture.pcap"
TIME_SLOT_SECONDS = 5                  # aggregate flows for this interval (the notebook used 5s slots)
CHECKPOINT_DIR = "./checkpoints"       # directory containing model checkpoint(s)
CHECKPOINT_PATTERNS = ("*.pt", "*.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This list matches the features you earlier specified (11 features)
FEATURE_COLS = [
    'Average Packet Size', 'Bwd Packets/s', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
    'ECE Flag Count', 'Flow Packets/s'
]

# ----------------------------
# Helper: find latest checkpoint
# ----------------------------
def find_latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR, patterns=CHECKPOINT_PATTERNS):
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(checkpoint_dir, pat)))
    if not files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}. Put your .pt/.pth there.")
    files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
    return files_sorted[0]

# ----------------------------
# Recreate the model classes (from your notebook)
# ----------------------------
import torch.nn as nn
from torch.nn import Linear, Dropout
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool

class FlowGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pool='max'):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = Linear(hidden_channels, out_channels)
        self.pool = pool

    def forward(self, data: Data):
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
    def __init__(self, in_channels, hidden_channels, num_classes=1):
        super().__init__()
        # using two SAGE layers for traffic graph (matches the notebook style)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.fc_out = Linear(hidden_channels, num_classes)

    def forward(self, graph: Data):
        x, edge_index, batch = graph.x, graph.edge_index, getattr(graph, 'batch', None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # aggregate per-graph
        g = global_mean_pool(x, batch)
        out = self.fc_out(g)
        return out

class FTGNet(nn.Module):
    def __init__(self, flow_gnn: FlowGNN, traffic_gnn: TrafficGNN):
        super().__init__()
        self.flow_gnn = flow_gnn
        self.traffic_gnn = traffic_gnn

    def forward(self, flow_list, traffic_graph: Data):
        # Accept list[Data] or already Batched flow_list
        if isinstance(flow_list, list):
            batch = Batch.from_data_list(flow_list)
        else:
            batch = flow_list
        flow_embeddings = self.flow_gnn(batch)   # shape [n_flows, hidden]
        traffic_graph = traffic_graph.clone()
        # attach embeddings to traffic_graph.x using same logic as notebook
        if hasattr(traffic_graph, 'num_nodes') and traffic_graph.num_nodes != flow_embeddings.size(0):
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
        preds = self.traffic_gnn(traffic_graph)
        return preds

# ----------------------------
# Attribute mapping helper (robust access to NFStreamer attributes)
# ----------------------------
def get_flow_attr(flow, keys, default=0.0):
    """
    Try several possible attribute names on NFStream flow object.
    keys: list of attribute name candidates in order of preference
    """
    for k in keys:
        # NFStreamer attributes are accessible as flow.some_name
        try:
            val = getattr(flow, k)
            if val is None:
                continue
            return val
        except Exception:
            continue
    return default

# mapping of requested FEATURE_COLS -> candidate flow attribute names in NFStreamer
FEATURE_TO_ATTRS = {
    'Average Packet Size': ['avg_packet_size', 'average_packet_size', 'avg_pkt_len', 'avg_packet_len'],
    'Bwd Packets/s': ['bwd_packets_per_s', 'bwd_packets_per_second', 'bwd_packets/s', 'bwd_pkts_per_s'],
    'FIN Flag Count': ['fin_flag_count', 'fin_flags', 'fin_count'],
    'SYN Flag Count': ['syn_flag_count', 'syn_flags', 'syn_count'],
    'RST Flag Count': ['rst_flag_count', 'rst_flags', 'rst_count'],
    'PSH Flag Count': ['psh_flag_count', 'psh_flags', 'psh_count'],
    'ACK Flag Count': ['ack_flag_count', 'ack_flags', 'ack_count'],
    'URG Flag Count': ['urg_flag_count', 'urg_flags', 'urg_count'],
    'CWE Flag Count': ['cwe_flag_count', 'cwe_flags', 'cwe_count', 'cwr_flag_count', 'cwr_flags'],
    'ECE Flag Count': ['ece_flag_count', 'ece_flags', 'ece_count'],
    'Flow Packets/s': ['flow_packets_per_s', 'packets_per_s', 'packets_per_second', 'flow_pkts_per_s']
}

# ----------------------------
# Build flow Data object (single-node graph for each flow)
# ----------------------------
def flow_to_data(flow, feature_order):
    """
    Returns torch_geometric.data.Data for a single flow.
    We use a 1-node graph with a self-loop edge_index [[0],[0]] so SAGEConv still receives edge_index.
    The 'x' will be the feature vector (tensor float32).
    """
    import torch
    feat = []
    for col in feature_order:
        candidates = FEATURE_TO_ATTRS.get(col, [col])
        val = get_flow_attr(flow, candidates, default=0.0)
        # ensure scalar
        try:
            val = float(val)
        except Exception:
            val = 0.0
        feat.append(val)
    x = torch.tensor([feat], dtype=torch.float32)   # shape [1, in_channels]
    edge_index = torch.tensor([[0],[0]], dtype=torch.long)   # self-loop
    data = Data(x=x, edge_index=edge_index)
    return data

# ----------------------------
# Build traffic graph: nodes = flows, edges if src_ip match OR dst_ip match
# (this matches the logic in your notebook)
# ----------------------------
def build_traffic_graph_from_flows(flows_list, hidden_dim):
    """
    flows_list: list of NFStreamer flow objects (in same order used to create flow Data)
    We will create an edge between nodes i and j if src_ip_i == src_ip_j OR dst_ip_i == dst_ip_j.
    Returns a torch_geometric Data object with placeholder x (zero) which FTGNet will replace with embeddings.
    """
    import torch
    nodes = []
    for f in flows_list:
        # use tuple (src_ip, dst_ip) or fallback fields
        src = getattr(f, "src_ip", None) or getattr(f, "source_ip", None) or getattr(f, "ip_src", None)
        dst = getattr(f, "dst_ip", None) or getattr(f, "destination_ip", None) or getattr(f, "ip_dst", None)
        nodes.append((src, dst))

    traffic_edge_list = []
    n = len(nodes)
    for i in range(n):
        for j in range(i+1, n):
            # connect if share src or share dst (same logic as notebook: nodes[i][0] == nodes[j][0] or nodes[i][1] == nodes[j][1])
            if nodes[i][0] == nodes[j][0] or nodes[i][1] == nodes[j][1]:
                traffic_edge_list.append([i, j])
                traffic_edge_list.append([j, i])
    if len(traffic_edge_list) == 0:
        # create self-edge to avoid empty edge_index
        edge_index = torch.tensor([[0],[0]], dtype=torch.long)
    else:
        edge_index = torch.tensor(traffic_edge_list, dtype=torch.long).t().contiguous()
    # placeholder x - FTGNet will replace this with flow embeddings
    x = torch.zeros((n if n>0 else 1, hidden_dim), dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index)
    return data

# ----------------------------
# Load checkpoint + optional scaler from checkpoint file
# ----------------------------
def load_checkpoint_and_model(checkpoint_path, in_ch, default_hidden=64):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    # try to extract model hyperparams if available
    hidden = default_hidden
    if isinstance(ckpt, dict):
        # if saved with metadata
        if 'hidden' in ckpt:
            hidden = int(ckpt['hidden'])
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            # maybe saved the model.state_dict() directly
            state = ckpt
    else:
        state = ckpt

    flow_gnn = FlowGNN(in_channels=in_ch, hidden_channels=hidden, out_channels=hidden).to(DEVICE)
    traffic_gnn = TrafficGNN(in_channels=hidden, hidden_channels=hidden, num_classes=1).to(DEVICE)
    net = FTGNet(flow_gnn, traffic_gnn).to(DEVICE)

    try:
        net.load_state_dict(state, strict=False)
        print("✅ Loaded checkpoint (state dict) into model (strict=False).")
    except Exception as e:
        print("⚠️ load_state_dict strict=False failed or mismatch:", e)
        # try to load nested keys (e.g., 'ftg_net' or 'model')
        if isinstance(state, dict):
            # try different likely keys
            for candidate in ['model_state_dict','state_dict','ftg_net','net','model']:
                if candidate in state:
                    try:
                        net.load_state_dict(state[candidate], strict=False)
                        print(f"✅ Loaded from nested key '{candidate}'.")
                        break
                    except Exception:
                        pass
    net.eval()
    # try to extract scaler if saved inside ckpt
    scaler = None
    if isinstance(ckpt, dict):
        if 'scaler' in ckpt:
            scaler = ckpt['scaler']
            print("✅ Found scaler inside checkpoint.")
    return net, scaler, hidden

# ----------------------------
# Main live loop using NFStreamer
# ----------------------------
def main():
    print("Device:", DEVICE)
    ckpt = find_latest_checkpoint()
    print("Using checkpoint:", ckpt)
    in_channels = len(FEATURE_COLS)
    net, scaler, hidden = load_checkpoint_and_model(ckpt, in_channels)

    print("Starting NFStreamer. Capturing flows...")
    # NFStreamer: source can be NIC name or pcap path
    streamer = NFStreamer(
        source=INTERFACE,
        statistical_analysis=True,
        idle_timeout=TIME_SLOT_SECONDS,
        active_timeout=TIME_SLOT_SECONDS*2,
        n_dissections=20
    )

    # Buffer for flows gathered in current timeslot
    timeslot_flows = []
    timeslot_start = None

    try:
        for flow in streamer:
            # NFStreamer yields flow objects as they complete or time out.
            # We'll group flows into TIME_SLOT_SECONDS windows based on arrival time.
            # Use flow.start_time or flow.last_seen if available; fallback to now.
            ts = None
            for key in ('start_time','first_seen_timestamp','timestamp','start_ts'):
                ts = getattr(flow, key, None)
                if ts is not None:
                    break
            if ts is None:
                ts = datetime.now().timestamp()

            if timeslot_start is None:
                timeslot_start = ts
            # if flow belongs to current slot, append; else process the slot and start a new one
            if ts - timeslot_start <= TIME_SLOT_SECONDS:
                timeslot_flows.append(flow)
            else:
                # process current slot
                if len(timeslot_flows) > 0:
                    process_timeslot(timeslot_flows, net, scaler, hidden)
                # start a new slot
                timeslot_flows = [flow]
                timeslot_start = ts
    except KeyboardInterrupt:
        print("Stopped by user (Ctrl-C).")
    finally:
        # flush remaining
        if timeslot_flows:
            process_timeslot(timeslot_flows, net, scaler, hidden)

# ----------------------------
# Process a batch/slot of flows
# ----------------------------
def process_timeslot(flow_objs, net, scaler, hidden_dim):
    import torch
    # 1) convert each flow to a flow-Data object
    flow_datas = []
    for f in flow_objs:
        fd = flow_to_data(f, FEATURE_COLS)
        flow_datas.append(fd)

    # 2) optional scaling: if scaler provided, apply to each x
    if scaler is not None:
        # scaler should be able to transform numpy array; try-except to be robust
        try:
            all_feats = torch.cat([d.x for d in flow_datas], dim=0).cpu().numpy()
            scaled = scaler.transform(all_feats)
            for i, d in enumerate(flow_datas):
                d.x = torch.tensor([scaled[i]], dtype=torch.float32)
        except Exception as e:
            print("⚠️ Could not apply scaler:", e)

    # 3) Build traffic graph connecting flows that share src or dst IP
    traffic_graph = build_traffic_graph_from_flows(flow_objs, hidden_dim).to(DEVICE)
    batch = Batch.from_data_list(flow_datas).to(DEVICE)

    # 4) Run model
    with torch.no_grad():
        preds = net(batch, traffic_graph)
        preds = preds.detach().cpu().numpy()

    # 5) Interpret and print results
    # preds shape: (num_graphs, 1) — here we have one graph per timeslot, but FTGNet returns per-graph predictions
    # If model returns a single scalar per timeslot, that's preds[0]
    # We'll convert using sigmoid and threshold 0.5 (same assumption as notebook)
    for i, f in enumerate(flow_objs):
        # If preds length >= number of flows, interpret per-flow else use first value per slot
        if preds.shape[0] == len(flow_objs):
            raw = float(preds[i].squeeze())
        else:
            raw = float(preds[0].squeeze())
        prob = 1.0/(1.0 + math.exp(abs(raw)))
        label = "MALICIOUS" if prob > 0.5 else "BENIGN"
        src = getattr(f, "src_ip", getattr(f, "source_ip", ""))
        dst = getattr(f, "dst_ip", getattr(f, "destination_ip", ""))
        sport = getattr(f, "src_port", getattr(f, "source_port", ""))
        dport = getattr(f, "dst_port", getattr(f, "destination_port", ""))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {src}:{sport} -> {dst}:{dport}  score={raw:.4f} prob={prob:.3f} label={label}")

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    main()
