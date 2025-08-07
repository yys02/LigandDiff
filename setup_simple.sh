#!/bin/bash

# Simple LigandDiff Environment Setup Script
# This script creates a Python virtual environment and installs packages via pip

echo "Setting up LigandDiff environment using pip..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits
else
    echo "⚠️  NVIDIA GPU not detected - will use CPU only"
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv liganddiff_env

if [ $? -eq 0 ]; then
    echo "Virtual environment created successfully!"
    
    # Activate environment
    source liganddiff_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch with CUDA support
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install other packages
    echo "Installing other packages..."
    pip install -r requirements_simple.txt
    
    echo ""
    echo "Environment setup completed!"
    echo ""
    echo "To activate the environment, run:"
    echo "source liganddiff_env/bin/activate"
    echo ""
    echo "To verify CUDA support, run:"
    echo "python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\""
    echo ""
    echo "To deactivate the environment, run:"
    echo "deactivate"
else
    echo "Error: Failed to create virtual environment."
    exit 1
fi