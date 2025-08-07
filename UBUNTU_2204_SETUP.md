# Ubuntu 22.04 + CUDA 12.4 Setup Guide

This guide is specifically optimized for Ubuntu 22.04 with CUDA 12.4 for the LigandDiff project.

## System Specifications

- **OS**: Ubuntu 22.04 LTS
- **CUDA**: 12.4
- **Python**: 3.9+
- **GPU**: NVIDIA GPU with CUDA 12.4 support

## Quick Setup

### Automated Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd LigandDiff

# Run the Ubuntu-optimized setup script
./setup_ubuntu2204.sh

# Activate the environment
conda activate liganddiff

# Verify CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Validate complete setup
python validate_environment.py
```

## Environment Optimizations

The `environment.yml` file has been specifically optimized for Ubuntu 22.04 + CUDA 12.4:

### PyTorch Configuration
- **PyTorch 2.0+** with CUDA 12.1 support (compatible with CUDA 12.4)
- **PyTorch Lightning 2.0+** for modern training workflows
- **NVIDIA channel** for optimized packages

### CUDA Compatibility
- Uses CUDA 12.1 wheels (fully compatible with CUDA 12.4)
- Includes `pytorch-cuda=12.1` package
- Optimized PyTorch Geometric installation

### Channel Priority
```yaml
channels:
  - pytorch      # Official PyTorch packages
  - nvidia       # NVIDIA optimized packages
  - conda-forge  # Community packages
  - defaults     # Base packages
```

## CUDA Installation (if needed)

If CUDA 12.4 is not already installed:

```bash
# Check current CUDA installation
nvidia-smi

# Install CUDA 12.4
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-12-4

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Verification Commands

After setup, verify your installation:

```bash
# Check CUDA availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check PyTorch Geometric
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"

# Check chemistry libraries
python -c "from rdkit import Chem; print('RDKit: OK')"
python -c "import openbabel; print('OpenBabel: OK')"

# Run full validation
python validate_environment.py
```

## Performance Optimizations

### GPU Memory Management
```python
# In your training script
import torch
torch.cuda.empty_cache()  # Clear GPU memory
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
```

### Environment Variables
```bash
# Add to ~/.bashrc for optimal performance
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Memory allocation
```

## Troubleshooting

### CUDA Version Mismatch
If you encounter CUDA version issues:
```bash
# Reinstall PyTorch with correct CUDA version
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### PyTorch Geometric Issues
```bash
# Reinstall PyG with CUDA 12.1 wheels
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
```

### Memory Issues
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Reduce batch size in config.yml if needed
```

## Expected Performance

With Ubuntu 22.04 + CUDA 12.4:
- **Training**: 2-3x faster than CPU-only
- **Generation**: 5-10x faster than CPU-only
- **Memory**: Efficient GPU memory usage
- **Stability**: Optimized for long training runs

## Next Steps

1. **Download Dataset**: Get data from the provided Zenodo link
2. **Start Training**: `python train.py --config config.yml`
3. **Generate Ligands**: `python generate.py --model model.ckpt --complex complex.xyz --outdir output/`
4. **Monitor Performance**: Use `nvidia-smi` and `htop` for monitoring

## Support

For Ubuntu 22.04 + CUDA 12.4 specific issues:
1. Check this guide first
2. Verify CUDA installation: `nvidia-smi`
3. Test PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
4. Check the main troubleshooting guide in `ENVIRONMENT_SETUP.md`