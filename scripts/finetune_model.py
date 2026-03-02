import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from torch_geometric.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

# Localize standard parameters
FEATURE_COLS = [
    'Average Packet Size', 'Bwd Packets/s', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count',
    'ECE Flag Count', 'Flow Packets/s'
]

MODEL_CONFIGS = {
    "best_model_1.pt": {"hidden": 512},
    "best_model_2.pt": {"hidden": 256},
    "best_model_3.pt": {"hidden": 128},
}

def safe_state_dict_from_ckpt(ckpt):
    """Extract state_dict from common checkpoint wrappers."""
    if isinstance(ckpt, dict):
        # Common keys used in training script
        for key in ("model_state", "model_state_dict", "state_dict", "model_state_dict_raw"):
            if key in ckpt:
                return ckpt[key]
        if all(hasattr(v, "shape") for v in ckpt.values()):
            return ckpt
    return ckpt

# Import FTGNet classes directly from old_codes.model
# We manipulate the path to force correct absolute loading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from old_codes.model import FTGNet, FlowGNN, TrafficGNN, FTGDataset

DATASET_PATH = '../custom_datasets/fine_tuning_dataset.csv'
CHECKPOINT_DIR = '../models/checkpoints_v4_metadata'
BASE_MODEL = os.path.join(CHECKPOINT_DIR, 'best_model_1.pt')
OUTPUT_MODEL = os.path.join(CHECKPOINT_DIR, 'best_model_1_finetuned.pt')

EPOCHS = 10
BATCH_SIZE = 16
LR = 5e-5

def normalize_dataset(df, model_path):
    """
    Attempt to load a scaler, otherwise perform fallback per-column normalization
    like the one applied during live inference in predict.py.
    """
    # 1. Try to load the original scaler
    scaler_file = None
    for cand in [model_path + ".scaler.pkl", 
                 os.path.join(os.path.dirname(model_path), "scaler.pkl"),
                 os.path.join(os.path.dirname(model_path), "standard_scaler.pkl")]:
        if os.path.exists(cand):
            scaler_file = cand
            break
            
    if scaler_file:
        print(f"Loading scaler from {scaler_file}")
        try:
            scaler = joblib.load(scaler_file)
            transformed = scaler.transform(df[FEATURE_COLS])
            df[FEATURE_COLS] = transformed
            return df
        except Exception as e:
            print(f"Failed to apply scaler: {e}. Falling back...")
            
    # 2. Fallback normalization (from predict.py fallback)
    print("Using log1p + standard normalization fallback.")
    arr = df[FEATURE_COLS].astype(float).values
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) # FIX NAN/INF
    arr = np.log1p(np.maximum(arr, 0)) # Ensure no negative log domains
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    arr = (arr - mean) / std
    arr = np.nan_to_num(arr, nan=0.0)
    df[FEATURE_COLS] = arr
    return df

def initialize_model(model_path, device):
    """
    Load the FTGNet based on the hyperparams extracted from the checkpoint wrapper
    (or fall back to defaults).
    """
    print(f"Loading checkpoint metadata from {model_path}...")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = safe_state_dict_from_ckpt(ckpt)
    
    flow_in = int(ckpt.get("flow_in", 11))
    hidden = int(ckpt.get("hidden", 128))
    traffic_in = int(ckpt.get("traffic_in", 128))
    num_classes = ckpt.get("num_classes", None)
    conv_types = ckpt.get("conv_types", {"flow": "sage", "traffic": "gat"})
    
    if num_classes is None:
        if isinstance(state_dict, dict) and "traffic_gnn.fc.weight" in state_dict:
            num_classes = int(state_dict["traffic_gnn.fc.weight"].shape[0])
        else:
            num_classes = 2 # default binary head
            if os.path.basename(model_path) in MODEL_CONFIGS:
                hidden = MODEL_CONFIGS[os.path.basename(model_path)]["hidden"]
                traffic_in = hidden
                
    print(f"Model parameters: flow_in={flow_in}, hidden={hidden}, traffic_in={traffic_in}, out={num_classes}")
    
    flow_gnn = FlowGNN(flow_in, hidden, hidden, conv_type=conv_types.get("flow", "sage"))
    traffic_gnn = TrafficGNN(hidden, hidden, out_channels=num_classes, conv_type=conv_types.get("traffic", "gat"))
    model = FTGNet(flow_gnn, traffic_gnn, device=device).to(device)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ Strictly loaded pretrained weights.")
    except Exception as e:
        print(f"⚠ Strict match failed. Using advanced key matching... ({e})")
        # Ensure we only prune the final layer, leaving Traffic and Flow untouched despite older variable name conventions
        filtered = {}
        for k, v in state_dict.items():
            if k in model.state_dict() and v.shape == model.state_dict()[k].shape:
                filtered[k] = v
            elif k in model.state_dict():
                print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model.state_dict()[k].shape}")
        
        # Load whatever fits
        model.load_state_dict(filtered, strict=False)
        print(f"✓ Loaded {len(filtered)} / {len(model.state_dict())} layers from pretrained weights.")
        
    return model, num_classes

def get_class_weights(df, device):
    """
    Calculates empirical class weights based on the heavily imbalanced dataset
    to pass heavily penalized loss values when 0-class (Normal) is seen.
    """
    labels = df['Label'].values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    # We convert back to a tensor dict format the CE Loss function accepts natively
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class Weights computed: {weight_tensor}")
    return weight_tensor

def my_collate(batch):
    from torch_geometric.data import Batch
    traffic_graphs = Batch.from_data_list([item[0] for item in batch])
    flow_lists = [item[1] for item in batch]
    return traffic_graphs, flow_lists

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"\\n--- 1. Data Preparation ---")
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded records: {len(df)}")
    
    # Pre-scale exactly like the predict script 
    df = normalize_dataset(df, BASE_MODEL)
    
    # Compute Weights
    class_weights = get_class_weights(df, device)
    
    # The input CSV uses 'Src IP' and 'Dst IP'
    # FTGDataset expects 'Source IP' and 'Destination IP', so we rename them here
    df = df.rename(columns={'Src IP': 'Source IP', 'Dst IP': 'Destination IP'})
    
    # Initialize Dataset (which uses standard grouping)
    # Note: Using small time slot durations since attacks are aggressive, we can get more samples this way
    print("Building Graphs...")
    dataset = FTGDataset(df, time_slot_duration="2S")
    print(f"Valid graph slots generated: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)

    print(f"\\n--- 2. Model Initialization ---")
    model, num_classes = initialize_model(BASE_MODEL, device)
    model.train() # Make sure dropout and batchNorm are on
    
    # Use correct loss based on model architecture
    if num_classes == 1:
        # Binary Classification Model uses a single output logit -> Sigmoid
        print("Using BCEWithLogitsLoss for 1 output channel.")
        # For BCE, pos_weight targets the ratio of negative to positive examples.
        # Since Attack (1) is 10,000 and Normal (0) is 1,184, we want a pos_weight < 1 (or > 1 if reversed).
        # Actually BCE pos_weight applies to the purely positive cases (label=1). We want to penalize 
        # missing Label 0 more.
        
        # Calculate ratio: negative_samples / positive_samples
        labels = df['Label'].values
        neg_count = (labels == 0).sum()
        pos_count = (labels == 1).sum()
        bce_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)
        print(f"BCE pos_weight: {bce_weight}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=bce_weight)
    else:
        # standard multiclass weighting
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use a low learning rate (5e-5) because we are fine tuning an already converged model
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    print(f"\\n--- 3. Fine-Tuning Execution ---")
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Manually loop over dataset to avoid PyG DataLoader tuple coercion bugs
        # PyG's DataLoader implicitly converts our inner list of Datas to namedtuples
        
        # Shuffle indices
        indices = torch.randperm(len(dataset)).tolist()
        
        # Mini-batches
        for i in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            
            # Manually extract and batch
            batch_t_graphs = []
            batch_f_graphs = []
            
            for idx in batch_indices:
                t_graph, f_list = dataset[idx]
                batch_t_graphs.append(t_graph)
                batch_f_graphs.append(f_list)
                
            from torch_geometric.data import Batch
            traffic_batch = Batch.from_data_list(batch_t_graphs).to(device)
            
            flat_f_graphs = []
            for fg_list in batch_f_graphs:
                for fg in fg_list:
                    flat_f_graphs.append(fg.to(device))
                    
            out = model(traffic_batch, flat_f_graphs)
            
            if num_classes == 1:
                # BCE needs float targets of shape [N, 1]
                target = traffic_batch.y.view(-1, 1).float().to(device)
                loss = criterion(out, target)
                
                preds = (torch.sigmoid(out) > 0.5).float()
                correct += (preds == target).sum().item()
            else:
                target = traffic_batch.y.view(-1).long().to(device)
                loss = criterion(out, target)
                
                preds = out.argmax(dim=1)
                correct += (preds == target).sum().item()
                
            loss.backward()
            optimizer.step()
            
            total += target.size(0)
            total_loss += loss.item()
            
            
        avg_loss = total_loss / max(len(dataloader), 1)
        accuracy = 100.0 * correct / max(total, 1)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
    print(f"\\n--- 4. Saving Artifacts ---")
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"Successfully saved Fine Tuned weights to: {OUTPUT_MODEL}")

if __name__ == '__main__':
    main()
