# Installing PyTorch with GPU (CUDA) Support

## Current Status
Your PyTorch installation is **CPU-only**. To use GPU, you need to install PyTorch with CUDA support.

## Step-by-Step GPU Installation

### 1. Check Your GPU
First, verify you have an NVIDIA GPU:
```bash
# Check if NVIDIA GPU is detected
nvidia-smi
```

If this command works, you have an NVIDIA GPU and CUDA drivers installed.

### 2. Check CUDA Version
```bash
nvidia-smi
```
Look for "CUDA Version" in the output (e.g., 11.8, 12.1, etc.)

### 3. Uninstall CPU-Only PyTorch
```bash
pip uninstall torch torchvision torchaudio
```

### 4. Install PyTorch with CUDA

#### Option A: Using pip (Recommended)
Visit https://pytorch.org/get-started/locally/ and select:
- Your OS (Windows)
- Package: pip
- Python version
- CUDA version (match your GPU's CUDA version)

**Example commands:**

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Option B: Using conda (If you use Anaconda)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 5. Verify Installation
```bash
python check_gpu.py
```

You should see:
```
CUDA available: True
GPU Name: [Your GPU Name]
```

### 6. Test GPU Training
```bash
python train.py --device cuda --num_epochs 1 --batch_size 16
```

## Common Issues

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size
```bash
python train.py --batch_size 16  # or smaller
```

### Issue: "No CUDA-capable device is detected"
**Solutions**:
1. Make sure you have an NVIDIA GPU (not AMD/Intel)
2. Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
3. Install CUDA toolkit if needed

### Issue: "CUDA version mismatch"
**Solution**: Install PyTorch version that matches your CUDA version

## Quick Check Commands

```bash
# Check GPU availability
python check_gpu.py

# Check NVIDIA drivers
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## After Installation

Once PyTorch with CUDA is installed, training will automatically use GPU:

```bash
# Auto-detect GPU (default)
python train.py

# Force GPU usage
python train.py --device cuda

# Use specific GPU (if multiple)
python train.py --device cuda --gpu_id 0
```

