import torch
import sys

print("--- SYSTEM VERIFICATION ---")
# Check if we are in the right environment
print(f"Python Version: {sys.version}")
print(f"Environment Path: {sys.executable}")

# Check PyTorch and CUDA
print(f"\nPyTorch Version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"Is CUDA (GPU) available? {cuda_available}")

if cuda_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Current GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
else:
    print("⚠️ WARNING: GPU not detected. PyTorch is running on your CPU.")