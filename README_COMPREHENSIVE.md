# LigandDiff: Comprehensive Documentation

## Overview

LigandDiff is a diffusion model for generating ligands for 3D transition metal complexes. This repository contains the official implementation of the ligands generation for 3D transition metal complexes, as described in the paper [LigandDiff](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c00232).

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Neon8988/LigandDiff.git
cd LigandDiff

# Install dependencies
pip install torch torch-geometric pytorch-lightning rdkit-python openbabel-python numpy matplotlib wandb
```

### Basic Usage

#### Training a Model

```bash
# Train with default configuration
python train.py --config config.yml --exp_name experiment1 --batch_size 32
```

#### Generating Ligands

```bash
# Generate ligands from a metal complex
python generate.py --outdir ./generated --model ./checkpoints/model.ckpt --complex ./input.xyz --n_samples 10
```

#### Sampling from Trained Model

```bash
# Sample from trained model
python sampling.py --outdir ./samples --model ./checkpoints/model.ckpt --dataset ./test_data.pt --batch_size 64
```

## Documentation Structure

This project includes comprehensive documentation organized as follows:

### 📚 [API Documentation](API_DOCUMENTATION.md)
Complete API reference covering all public functions, classes, and components with examples and usage instructions.

**Sections:**
- Main Entry Points (`generate.py`, `train.py`, `sampling.py`)
- Core Model Classes (`DDPM`, `EDM`, `Dynamics`)
- Data Processing (`molecule_builder.py`)
- Visualization (`visualizer.py`)
- Utilities (`utils.py`)
- Constants and Configuration (`const.py`)

### 🧠 [Neural Network Architecture](NEURAL_NETWORK_DOCUMENTATION.md)
Detailed documentation of the neural network components and architectural decisions.

**Sections:**
- EGNN Architecture
- GVP Components
- Activation Functions
- Layer Components
- Noise Schedules
- Performance Optimizations

## Project Structure

```
LigandDiff/
├── src/                    # Source code
│   ├── lightning.py       # Main PyTorch Lightning module
│   ├── edm.py            # Equivariant Diffusion Model
│   ├── egnn.py           # Equivariant Graph Neural Network
│   ├── gvp.py            # Graph Vector Perceptron
│   ├── gvp_model.py      # GVP network implementation
│   ├── molecule_builder.py # Molecular structure building
│   ├── visualizer.py     # Visualization utilities
│   ├── utils.py          # Utility functions
│   ├── const.py          # Constants and configuration
│   ├── noise.py          # Noise schedule implementations
│   ├── layer_norm.py     # Layer normalization
│   ├── dropout.py        # Dropout implementation
│   └── SA_Score/         # SA Score calculation
├── generate.py            # Main generation script
├── train.py              # Main training script
├── sampling.py           # Sampling script
├── ppr.py               # Post-processing script
├── config.yml           # Configuration file
├── README.md            # Original README
├── API_DOCUMENTATION.md # Comprehensive API docs
└── NEURAL_NETWORK_DOCUMENTATION.md # Architecture docs
```

## Key Components

### 1. Diffusion Model (`DDPM`)
The main PyTorch Lightning module that orchestrates the entire training and generation process.

**Key Features:**
- Equivariant diffusion process
- Ligand-specific masking
- Multi-scale attention
- Comprehensive metrics tracking

### 2. Equivariant Graph Neural Network (`EGNN`)
Handles the graph-based processing of molecular structures while maintaining rotational and translational equivariance.

**Key Features:**
- Equivariant coordinate updates
- Attention mechanisms
- Multi-layer architecture
- Flexible activation functions

### 3. Molecular Building (`molecule_builder.py`)
Utilities for constructing and manipulating molecular structures.

**Key Features:**
- RDKit integration
- OpenBabel bond creation
- Validity checking
- XYZ file I/O

### 4. Visualization (`visualizer.py`)
Tools for visualizing molecular structures and diffusion chains.

**Key Features:**
- 3D molecular plotting
- Diffusion chain visualization
- Animation generation
- Multiple output formats

## Configuration

### Training Configuration (`config.yml`)

```yaml
# Model parameters
hidden_nf: 64
n_layers: 4
attention: true
activation: "silu"

# Diffusion parameters
diffusion_steps: 1000
diffusion_noise_schedule: "learned"
diffusion_loss_type: "vlb"

# Training parameters
lr: 0.001
batch_size: 32
n_epochs: 1000

# Data parameters
data_path: "./data"
train_data: "train.pt"
val_data: "val.pt"
```

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hidden_nf` | Number of hidden features | 64 |
| `n_layers` | Number of EGNN layers | 4 |
| `attention` | Use attention mechanism | true |
| `diffusion_steps` | Number of diffusion timesteps | 1000 |
| `lr` | Learning rate | 0.001 |
| `batch_size` | Training batch size | 32 |

## Dataset

Download all datasets from [Zenodo](https://zenodo.org/records/10651292).

The dataset includes:
- Metal complexes in XYZ format
- Pre-processed PyTorch Geometric data
- Training/validation/test splits

## Usage Examples

### Training

```python
from src.lightning import DDPM
from pytorch_lightning import Trainer

# Initialize model
model = DDPM(
    data_path="./data",
    train_data="train.pt",
    val_data="val.pt",
    in_node_nf=8,
    n_dims=3,
    ligand_group_node_nf=6,
    hidden_nf=64,
    n_layers=4,
    diffusion_steps=1000,
    lr=0.001,
    batch_size=32
)

# Train model
trainer = Trainer(max_epochs=1000)
trainer.fit(model)
```

### Generation

```python
from generate import main

# Generate ligands from complex
main(
    outdir="./generated",
    model="./checkpoints/model.ckpt",
    complex="./input.xyz",
    n_samples=10,
    ligand_sizes='random'
)
```

### Visualization

```python
from src.visualizer import plot_data3d, visualize_chain

# Plot single molecule
plot_data3d(positions, atom_types, save_path="molecule.png")

# Visualize diffusion chain
visualize_chain("./chain_data", save_path="chain.gif")
```

## Performance Considerations

### GPU Usage
- Use `torch_device='cuda:0'` for GPU training
- Ensure sufficient VRAM for batch processing
- Consider mixed precision training for large models

### Memory Management
- Use `torch.no_grad()` for inference
- Implement gradient checkpointing for large models
- Monitor memory usage during training

### Optimization Tips
- Adjust batch size based on available memory
- Use gradient clipping for stable training
- Enable attention mechanisms for better performance

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable mixed precision training

2. **NaN Values**
   - Check for numerical instability
   - Adjust learning rate
   - Verify input data normalization

3. **Poor Convergence**
   - Adjust learning rate
   - Check data preprocessing
   - Verify model architecture

### Debugging

```python
from src.utils import log, FoundNaNException

# Enable logging
log("Training started")

# Handle NaN exceptions
try:
    output = model(input_data)
except FoundNaNException as e:
    print(f"NaN detected in tensor: {e.x}")
```

## Dependencies

### Required
- PyTorch >= 1.9.0
- PyTorch Geometric >= 2.0.0
- PyTorch Lightning >= 1.5.0
- RDKit >= 2021.03.1
- OpenBabel >= 3.1.0
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0

### Optional
- Weights & Biases (wandb) for experiment tracking
- CUDA for GPU acceleration

## Citation

If you use this code in your research, please cite:

```bibtex
@article{liganddiff2024,
  title={LigandDiff: Diffusion Models for 3D Ligand Generation},
  author={...},
  journal={Journal of Chemical Theory and Computation},
  year={2024},
  doi={10.1021/acs.jctc.4c00232}
}
```

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The EGNN architecture is based on the original EGNN paper
- Molecular building utilities use RDKit and OpenBabel
- Visualization tools are built on Matplotlib

## Support

For questions and support:
- Check the [API Documentation](API_DOCUMENTATION.md)
- Review the [Neural Network Documentation](NEURAL_NETWORK_DOCUMENTATION.md)
- Open an issue on GitHub
- Contact the maintainers

---

**Note**: This is the official implementation of LigandDiff. For the latest updates and multi-ligand version, see [multi-LigandDiff](https://github.com/Neon8988/multi_LigandDiff).