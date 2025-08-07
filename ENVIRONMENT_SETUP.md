# LigandDiff Environment Setup

This guide will help you set up the environment for running the LigandDiff model for 3D transition metal complex ligand generation.

## Prerequisites

1. **Anaconda or Miniconda**: Make sure you have conda installed on your system.
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - Or install Anaconda: https://www.anaconda.com/products/distribution

2. **CUDA (Optional)**: For GPU acceleration, install CUDA compatible with PyTorch.
   - Check compatibility: https://pytorch.org/get-started/locally/

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd LigandDiff

# Run the setup script
./setup_environment.sh

# Activate the environment
conda activate liganddiff
```

### Option 2: Manual Setup

```bash
# Create the environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate liganddiff
```

## Environment Details

The `environment.yml` file includes:

### Core Dependencies
- **Python 3.9**: Base Python version
- **PyTorch 1.12+**: Deep learning framework
- **PyTorch Lightning 1.8+**: Training framework
- **PyTorch Geometric 2.2+**: Graph neural networks

### Chemistry Libraries
- **RDKit 2022.9+**: Chemical informatics toolkit
- **OpenBabel 3.1+**: Chemical toolbox
- **molSimplify 2.0+**: Molecular simplification tools

### Scientific Computing
- **NumPy 1.21+**: Numerical computing
- **Pandas 1.3+**: Data manipulation
- **SciPy 1.7+**: Scientific computing
- **Scikit-learn 1.0+**: Machine learning utilities

### Visualization
- **Matplotlib 3.5+**: Plotting library
- **ImageIO 2.9+**: Image processing

### Utilities
- **PyYAML 6.0+**: Configuration file parsing
- **WandB 0.13+**: Experiment tracking
- **Jupyter**: Interactive development

## Verification

After setting up the environment, verify the installation:

```bash
# Activate the environment
conda activate liganddiff

# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test PyTorch Geometric
python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"

# Test chemistry libraries
python -c "from rdkit import Chem; print('RDKit imported successfully')"
python -c "import openbabel; print('OpenBabel imported successfully')"

# Test PyTorch Lightning
python -c "import pytorch_lightning; print(f'PyTorch Lightning version: {pytorch_lightning.__version__}')"
```

## Troubleshooting

### Common Issues

1. **CUDA Issues**: If you encounter CUDA-related errors, try:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

2. **PyTorch Geometric Installation**: If PyG installation fails:
   ```bash
   pip install torch-geometric
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
   ```

3. **molSimplify Issues**: If molSimplify fails to install:
   ```bash
   pip install molSimplify
   ```

4. **OpenBabel Issues**: For OpenBabel installation problems:
   ```bash
   conda install -c conda-forge openbabel
   ```

### Environment Management

```bash
# List all environments
conda env list

# Activate environment
conda activate liganddiff

# Deactivate environment
conda deactivate

# Remove environment (if needed)
conda env remove -n liganddiff

# Update environment
conda env update -f environment.yml
```

## Usage

Once the environment is set up, you can run the LigandDiff model:

```bash
# Activate the environment
conda activate liganddiff

# Training
python train.py --config config.yml

# Generation
python generate.py --model path/to/model.ckpt --complex path/to/complex.xyz --outdir output/
```

## System Requirements

- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 10GB free space
- **GPU**: Optional but recommended for faster training (NVIDIA GPU with CUDA support)
- **OS**: Linux, macOS, or Windows (with WSL for Windows)

## Support

If you encounter issues with the environment setup:

1. Check the troubleshooting section above
2. Ensure all prerequisites are installed
3. Try creating a fresh conda environment
4. Check the project's GitHub issues for similar problems

For more information about the LigandDiff model, see the main README.md file.