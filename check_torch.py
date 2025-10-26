import torch

try:
    import torch
    print("PyTorch is installed.")
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
except ImportError:
    print("PyTorch is not installed.")

