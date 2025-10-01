#!/usr/bin/env python3
"""
predict.py
Flexible live prediction supporting FTGNet v2 (GCN) and v3 (SAGE/GAT) checkpoints.
Modernized: safe torch.load, auto class-detection, clean predictions.
"""

import sys
import os
import torch
import numpy as np
from nfstream import NFStreamer
from torch_geometric.data import Data

# local imports
from model import FTGNet, FlowGNN, TrafficGNN
from nf_print_features import flow_to_features

# fallback config for legacy raw checkpoints
MODEL_CONFIGS = {
    "best_model_1.pt": {"hidden": 512},
    "best_model_2.pt": {"hidden": 256},
    "best_model_3.pt": {"hidden": 128},
}

CLASS_LABELS = {0: "benign", 1: "attack"}


def flow_to_graphs(flow, traffic_in: int):
    """Convert NFStreamer flow to graph inputs for flow and traffic GNNs."""
    features = flow_to_features(flow)
    flow_x = torch.tensor([features], dtype=torch.float32)

    # traffic_x padded/truncated to traffic_in
    traffic_np = np.zeros((traffic_in,), dtype=np.float32)
    traffic_np[:min(len(features), traffic_in)] = np.array(features[:traffic_in], dtype=np.float32)
    traffic_x = torch.from_numpy(traffic_np.reshape(1, -1)).float()

    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    flow_graph = Data(x=flow_x, edge_index=edge_index)
    traffic_graph = Data(x=traffic_x, edge_index=edge_index)
    return flow_graph, traffic_graph


def try_load_model(model_path: str):
    """Load FTGNet model from checkpoint with safe handling."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        # v3 style checkpoint with metadata
        flow_in = ckpt.get("flow_in", 11)
        traffic_in = ckpt.get("traffic_in", ckpt.get("hidden", 128))
        hidden = ckpt.get("hidden", 128)
        num_classes = ckpt.get("num_classes", 2)
        conv_types = ckpt.get("conv_types", {"flow": "sage", "traffic": "gat"})
        state_dict = ckpt["model_state"]
        print(f"[+] Loaded metadata: flow_in={flow_in}, traffic_in={traffic_in}, "
              f"hidden={hidden}, classes={num_classes}, conv={conv_types}")
    else:
        # legacy checkpoint (no metadata)
        model_file = os.path.basename(model_path)
        if model_file not in MODEL_CONFIGS:
            raise RuntimeError(f"Unknown model file {model_file}. Update MODEL_CONFIGS.")

        flow_in = 11
        hidden = MODEL_CONFIGS[model_file]["hidden"]
        traffic_in = hidden

        # Extract state_dict safely
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise RuntimeError("Unrecognized checkpoint format")

        # Detect conv types
        flow_conv = "sage" if any("lin_l" in k or "lin_r" in k for k in state_dict.keys()) else "gcn"
        traffic_conv = "gat" if any("att_src" in k or "att_dst" in k for k in state_dict.keys()) else "gcn"
        conv_types = {"flow": flow_conv, "traffic": traffic_conv}
        print(f"[!] Heuristic conv detection: {conv_types}")

        # Detect num_classes from final layer shape
        if "traffic_gnn.fc.weight" in state_dict:
            num_classes = state_dict["traffic_gnn.fc.weight"].shape[0]
        else:
            num_classes = 2  # fallback

    # Build model with correct output size
    flow_gnn = FlowGNN(flow_in, hidden, hidden, conv_type=conv_types["flow"])
    traffic_gnn = TrafficGNN(hidden, hidden, out_channels=num_classes,
                             conv_type=conv_types["traffic"])
    model = FTGNet(flow_gnn, traffic_gnn)

    # Load weights strictly
    try:
        model.load_state_dict(state_dict, strict=True)
        print("[+] Model loaded strictly")
    except Exception as e:
        print("[!] Strict load failed:", e)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and model_state[k].shape == v.shape}
        mismatched = [k for k in state_dict if k not in filtered or model_state[k].shape != state_dict[k].shape]
        model_state.update(filtered)
        model.load_state_dict(model_state, strict=False)
        print(f"[+] Filtered load: loaded {len(filtered)} keys, mismatched {len(mismatched)}")

    return model.eval(), flow_in, traffic_in, hidden, num_classes


def predict_output(model, traffic_graph, flow_graph, num_classes: int):
    """Run model inference and return predicted class + probability."""
    with torch.no_grad():
        out = model(traffic_graph, [flow_graph])

    if num_classes == 1:
        prob = torch.sigmoid(out).item()
        pred_class = int(prob > 0.5)
    else:
        probs = torch.softmax(out, dim=-1)[0]
        pred_class = int(probs.argmax().item())
        prob = float(probs[pred_class].item())

    label = CLASS_LABELS.get(pred_class, str(pred_class))
    return pred_class, prob, label


def main():
    if len(sys.argv) < 2:
        print("Usage: sudo venv/bin/python predict.py <model_path> [iface]")
        sys.exit(1)

    model_path = sys.argv[1]
    iface = sys.argv[2] if len(sys.argv) > 2 else "eth0"
    act_timeout = 1

    model, flow_in, traffic_in, hidden, num_classes = try_load_model(model_path)

    print(f"[*] Starting NFStreamer on iface='{iface}' (active_timeout={act_timeout})")
    streamer = NFStreamer(source=iface, active_timeout=act_timeout, statistical_analysis=True)

    try:
        for flow in streamer:
            flow_graph, traffic_graph = flow_to_graphs(flow, traffic_in)
            pred_class, prob, label = predict_output(model, traffic_graph, flow_graph, num_classes)
            print(f"Predicted {label} (class={pred_class}), prob={prob:.6f}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print("Runtime error:", e)


if __name__ == "__main__":
    main()
