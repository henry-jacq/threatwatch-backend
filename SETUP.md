## NFStreamer + PyTorch Geometric Environment Setup

## 0. Install System Dependencies

```bash
sudo apt update && sudo apt install -y \
  autoconf \
  automake \
  libtool \
  build-essential \
  cmake \
  libpcap-dev \
  libtins-dev \
  python3-dev \
  git
```

---

## 1. Create and Activate Virtual Environment

```bash
# Create virtual environment (if not already created)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows
```

---

## 2. Upgrade Core Python Packages

```bash
pip install --upgrade pip setuptools wheel
```

---

## 3. Install NFStreamer (Latest Development Version)

```bash
pip install git+https://github.com/nfstream/nfstream.git
```

---

## 4. Install PyTorch (Nightly / CUDA 12.8 Example)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

> **Note:** Replace the CUDA version if your system uses a different one.

---

## 5. Install PyTorch Geometric (PyG) and CUDA Extensions

```bash
# Core PyG
pip install torch_geometric

# Optional CUDA-aware operations
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

**Force Re-Install (if you run into version conflicts):**

```bash
pip install --force-reinstall --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

**Optional Cleanup Before Reinstall:**

```bash
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib -y
```

---

## 6. Allow System Site Packages (Optional)

If you want your virtual environment to access system-wide Python packages:

```bash
nano venv/pyvenv.cfg
```

Set:

```cfg
include-system-site-packages = true
```

---

## 7. Install Other Utilities

```bash
pip install pandas scapy scikit-learn numpy
```

---

## 8. Verify Installation

```bash
python -c "
import torch, torch_geometric;
print('Torch Version:', torch.__version__);
print('CUDA Version:', torch.version.cuda);
print('PyG Version:', torch_geometric.__version__)
"
```

**Expected Output Example:**

```
Torch Version: 2.8.0+cu128
CUDA Version: 12.8
PyG Version: 2.6.1
```

## 9. Run Prediction Script

```bash
sudo venv/bin/python predict.py <path_to_model>
```