#!/usr/bin/env python3
"""
predict.py — slot-based live inference compatible with the FTGDataset training pipeline.

Features:
 - auto-detect checkpoint format and final-head size (1 vs N)
 - load scaler saved during training (model_path + ".scaler.pkl") if present
 - fall back to robust log1p normalization if scaler missing (warns)
 - construct slot DataFrame using flow_to_features() (exact same ordering)
 - build FTGDataset for the slot (same grouping logic as training)
 - aggregate node-level outputs into a slot-level probability
 - configurable threshold for binary head

Usage:
 sudo venv/bin/python predict.py <model_path> [iface] [slot_sec] [threshold]
 Example:
 sudo venv/bin/python predict.py best_model_1.pt eth0 5 0.8
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
import numpy as np
import torch
from nfstream import NFStreamer

# local imports (assumes these files are in same folder or installed)
from model import FTGNet, FlowGNN, TrafficGNN, FTGDataset
from nf_print_features import flow_to_features  # returns list of 11 features in fixed order

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt='%H:%M:%S')
logger = logging.getLogger("predict")

# feature columns must exactly match FTGDataset.feature_cols
FEATURE_COLS = [
    'Average Packet Size', 'Bwd Packets/s', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
    'ECE Flag Count', 'Flow Packets/s'
]

MODEL_CONFIGS = {
    "best_model_1.pt": {"hidden": 512},
    "best_model_2.pt": {"hidden": 256},
    "best_model_3.pt": {"hidden": 128},
}

CLASS_LABELS = {0: "benign", 1: "attack"}


def find_scaler(model_path):
    """Try to find an associated scaler file near the model path."""
    candidates = [
        model_path + ".scaler.pkl",
        os.path.join(os.path.dirname(model_path), "scaler.pkl"),
        os.path.join(os.path.dirname(model_path), "standard_scaler.pkl")
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_scaler(mpath):
    """Load joblib scaler; return None if not found or load fails."""
    scaler_file = find_scaler(mpath)
    if not scaler_file:
        logger.warning("No scaler file found next to model. Inference will use fallback normalization.")
        return None
    try:
        scaler = joblib.load(scaler_file)
        logger.info(f"Loaded scaler from: {scaler_file}")
        return scaler
    except Exception as e:
        logger.warning(f"Failed to load scaler {scaler_file}: {e}. Will use fallback normalization.")
        return None


def safe_state_dict_from_ckpt(ckpt):
    """Extract state_dict from common checkpoint wrappers."""
    if isinstance(ckpt, dict):
        # Common keys used in training script
        for key in ("model_state", "model_state_dict", "state_dict", "model_state_dict_raw"):
            if key in ckpt:
                return ckpt[key]
        # maybe ckpt contains only state dict already
        # filter out non-parameter keys (heuristic)
        # If majority of keys are tensors, return ckpt itself.
        if all(hasattr(v, "shape") for v in ckpt.values()):
            return ckpt
    # otherwise return ckpt as-is (maybe it's a plain state dict)
    return ckpt


def try_load_model(model_path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = safe_state_dict_from_ckpt(ckpt)

    # default hyperparams
    flow_in = 11
    hidden = 128
    traffic_in = 128
    num_classes = None
    conv_types = {"flow": "sage", "traffic": "gat"}

    if isinstance(ckpt, dict):
        flow_in = int(ckpt.get("flow_in", flow_in))
        hidden = int(ckpt.get("hidden", hidden))
        traffic_in = int(ckpt.get("traffic_in", hidden))
        num_classes = ckpt.get("num_classes", None)
        conv_types = ckpt.get("conv_types", conv_types)

    # If state_dict includes traffic head weights, detect the output dim
    if isinstance(state_dict, dict) and "traffic_gnn.fc.weight" in state_dict:
        out_dim = int(state_dict["traffic_gnn.fc.weight"].shape[0])
        num_classes = out_dim if num_classes is None else num_classes
        logger.info(f"Detected traffic_gnn.fc weight shape -> out_dim={out_dim}")

    # fallback using known model filenames
    if num_classes is None:
        # default to 2 unless training metadata says otherwise
        num_classes = 2
        model_file = os.path.basename(model_path)
        if model_file in MODEL_CONFIGS:
            hidden = MODEL_CONFIGS[model_file]["hidden"]
            traffic_in = hidden

    # Build model matching the checkpoint head (binary or multi-class)
    flow_gnn = FlowGNN(flow_in, hidden, hidden, conv_type=conv_types.get("flow", "sage"))
    traffic_gnn = TrafficGNN(hidden, hidden, out_channels=num_classes, conv_type=conv_types.get("traffic", "gat"))
    model = FTGNet(flow_gnn, traffic_gnn)

    # Load weights robustly: strict try, otherwise filtered shape-match
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.info("[+] Model loaded strictly")
    except Exception as exc:
        logger.warning(f"[!] Strict load failed: {exc}")
        # keep only matching shapes
        filtered = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
        logger.info(f"[+] Filtered load: loading {len(filtered)} keys (of {len(model.state_dict())})")
        model.load_state_dict(filtered, strict=False)

    return model.eval(), flow_in, traffic_in, hidden, int(num_classes)


def normalize_features(df_features: pd.DataFrame, scaler):
    """Apply scaler if present, otherwise robust fallback."""
    if scaler is not None:
        # scaler expects 2D array
        transformed = scaler.transform(df_features[FEATURE_COLS])
        df_features[FEATURE_COLS] = transformed
        return df_features

    # fallback: log1p for count-like features + clip to reasonable ranges
    df = df_features.copy()
    # apply per-column fallback: many features are counts or rates -> log1p then z-score per-slot
    arr = df[FEATURE_COLS].astype(float).values
    arr = np.log1p(arr)  # reduce large values
    # z-score per slot (slot-level mean/std)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    arr = (arr - mean) / std
    df[FEATURE_COLS] = arr
    logger.warning("Using fallback normalization (log1p + per-slot zscore). Not a substitute for the training scaler.")
    return df


def build_slot_dataframe(flows):
    """
    Build a DataFrame for the slot using flow_to_features() for exact feature ordering.
    `flows` is an iterable of NFStreamer flow objects for that slot.
    """
    rows = []
    for flow in flows:
        feats = flow_to_features(flow)  # list length == len(FEATURE_COLS)
        if len(feats) != len(FEATURE_COLS):
            # defensive: skip if mismatch
            logger.warning("flow_to_features returned unexpected length; skipping flow")
            continue

        ts = None
        # try to extract a stable timestamp from flow (some NFStreamer builds)
        ts_ms = getattr(flow, "bidirectional_first_seen_ms", None) or getattr(flow, "src2dst_first_seen_ms", None)
        if ts_ms:
            try:
                ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        row = {
            "Timestamp": ts,
            "Source IP": getattr(flow, "src_ip", "0.0.0.0"),
            "Destination IP": getattr(flow, "dst_ip", "0.0.0.0"),
            "Label": 0  # placeholder required by FTGDataset
        }
        for name, val in zip(FEATURE_COLS, feats):
            row[name] = float(val)
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df


def predict_slot(model, df_slot: pd.DataFrame, scaler, num_classes: int, thresh: float = 0.5, debug=False):
    """
    Returns: (pred_class:int, prob:float, label_str)
    - builds FTGDataset from the slot (same as training),
    - aggregates node-level predictions to slot-level by mean.
    """
    if df_slot is None or df_slot.empty:
        return None

    # normalize features
    df_slot = normalize_features(df_slot, scaler)

    # create dataset with exact slot duration so FTGDataset will create items the same way
    dataset = FTGDataset(df_slot, time_slot_duration=f"{int((df_slot['Timestamp'].max() - df_slot['Timestamp'].min()).total_seconds())+1}s")
    # fallback: if that grouping somehow fails, try a generic short window
    if len(dataset) == 0:
        dataset = FTGDataset(df_slot, time_slot_duration="5s")
        if len(dataset) == 0:
            return None

    traffic_graph, flow_graphs = dataset[0]
    if len(flow_graphs) == 0:
        return None

    with torch.no_grad():
        out = model(traffic_graph, flow_graphs)  # shape [n_nodes, out_channels]

    if num_classes == 1:
        # out may be shape [n_nodes, 1] or [n_nodes]
        probs_tensor = torch.sigmoid(out)
        probs = probs_tensor.view(-1).cpu().numpy()
        # aggregate across nodes
        slot_prob = float(np.mean(probs))
        pred_class = int(slot_prob > thresh)
        if debug:
            logger.info(f"Node probs: {probs.tolist()}")
        return pred_class, slot_prob, CLASS_LABELS.get(pred_class, str(pred_class))
    else:
        # multi-class: average the per-node softmax probabilities
        probs_tensor = torch.softmax(out, dim=-1).cpu().numpy()  # shape [n_nodes, C]
        mean_probas = probs_tensor.mean(axis=0)
        pred_class = int(np.argmax(mean_probas))
        prob = float(mean_probas[pred_class])
        if debug:
            logger.info(f"Node softmax mean: {mean_probas.tolist()}")
        return pred_class, prob, CLASS_LABELS.get(pred_class, str(pred_class))


def main():
    parser = argparse.ArgumentParser(description="Live FTG-Net inference (slot-based).")
    parser.add_argument("model", help="model checkpoint path (.pt)")
    parser.add_argument("iface", nargs="?", default="eth0", help="interface or NFStreamer source (default eth0)")
    parser.add_argument("slot", nargs="?", type=int, default=5, help="slot duration seconds (default 5)")
    parser.add_argument("thresh", nargs="?", type=float, default=0.5, help="threshold for binary head (default 0.5)")
    parser.add_argument("--debug", action="store_true", help="print node-level probs for debugging")
    args = parser.parse_args()

    model_path = args.model
    iface = args.iface
    slot_sec = args.slot
    thresh = args.thresh
    debug = args.debug

    model, flow_in, traffic_in, hidden, num_classes = try_load_model(model_path)
    scaler = load_scaler(model_path)

    logger.info(f"Model ready. slot={slot_sec}s, flow_in={flow_in}, traffic_in={traffic_in}, hidden={hidden}, num_classes={num_classes}")
    logger.info(f"Starting NFStreamer on iface='{iface}' (slot={slot_sec}s). Press Ctrl+C to stop.")

    streamer = NFStreamer(source=iface, active_timeout=slot_sec, statistical_analysis=True)

    buffer = []
    slot_start = datetime.now(timezone.utc)

    try:
        for flow in streamer:
            buffer.append(flow)

            if (datetime.now(timezone.utc) - slot_start) >= timedelta(seconds=slot_sec):
                df_slot = build_slot_dataframe(buffer)
                result = predict_slot(model, df_slot, scaler, num_classes=num_classes, thresh=thresh, debug=debug)
                if result:
                    pred_class, prob, label = result
                    logger.info(f"[{slot_start.isoformat()}] Predicted {label}, class={pred_class}, prob={prob:.4f}, flows={len(buffer)}")
                else:
                    logger.info(f"[{slot_start.isoformat()}] No valid graphs built from slot (flows={len(buffer)})")

                buffer = []
                slot_start = datetime.now(timezone.utc)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.exception("Runtime error:")

if __name__ == "__main__":
    main()
