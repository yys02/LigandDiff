#!/bin/bash

# LigandDiff Environment Setup Script for Ubuntu 22.04 + CUDA 12.4
# This script creates and activates the conda environment optimized for your system

echo "Setting up LigandDiff environment for Ubuntu 22.04 + CUDA 12.4..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits
else
    echo "⚠️  NVIDIA GPU not detected - will use CPU only"
fi

# Check Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$VERSION_ID" == "22.04" ]]; then
        echo "✓ Ubuntu 22.04 detected - optimized setup"
    else
        echo "⚠️  Ubuntu $VERSION_ID detected - setup should still work"
    fi
fi

# Create the environment from environment.yml
echo "Creating conda environment optimized for Ubuntu 22.04 + CUDA 12.4..."
conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo "Environment created successfully!"
    
    # Install PyTorch Geometric packages
    echo "Installing PyTorch Geometric packages..."
    conda activate liganddiff
    pip install torch-geometric
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv --no-build-isolation
    
    echo ""
    echo "Environment setup completed!"
    echo ""
    echo "To activate the environment, run:"
    echo "conda activate liganddiff"
    echo ""
    echo "To verify CUDA support, run:"
    echo "python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\""
    echo ""
    echo "To validate the complete setup, run:"
    echo "python validate_environment.py"
    echo ""
    echo "To test the model, run:"
    echo "python generate.py --outdir output --model model/pretrained.ckpt --complex your_file.xyz"
    echo ""
    echo "To deactivate the environment, run:"
    echo "conda deactivate"
else
    echo "Error: Failed to create environment. Please check the environment.yml file and try again."
    exit 1
fi