#!/bin/bash

# LigandDiff Environment Setup Script
# This script creates and activates the conda environment for the LigandDiff project

echo "Setting up LigandDiff environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create the environment from environment.yml
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo "Environment created successfully!"
    echo ""
    echo "To activate the environment, run:"
    echo "conda activate liganddiff"
    echo ""
    echo "To deactivate the environment, run:"
    echo "conda deactivate"
    echo ""
    echo "To remove the environment if needed, run:"
    echo "conda env remove -n liganddiff"
else
    echo "Error: Failed to create environment. Please check the environment.yml file and try again."
    exit 1
fi