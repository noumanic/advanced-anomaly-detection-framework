"""Check GPU availability and PyTorch CUDA support"""
import torch

print("="*60)
print("GPU/CUDA Availability Check")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
else:
    print("\n⚠️  CUDA is not available!")
    print("Possible reasons:")
    print("  1. No NVIDIA GPU detected")
    print("  2. PyTorch was installed without CUDA support")
    print("  3. CUDA drivers not installed")
    print("\nTo install PyTorch with CUDA support, visit:")
    print("  https://pytorch.org/get-started/locally/")

print("="*60)

