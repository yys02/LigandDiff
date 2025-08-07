# LigandDiff API Documentation

## Overview

LigandDiff is a diffusion model for generating ligands for 3D transition metal complexes. This documentation covers all public APIs, functions, and components with examples and usage instructions.

## Table of Contents

1. [Main Entry Points](#main-entry-points)
2. [Core Model Classes](#core-model-classes)
3. [Data Processing](#data-processing)
4. [Molecular Building](#molecular-building)
5. [Visualization](#visualization)
6. [Utilities](#utilities)
7. [Constants and Configuration](#constants-and-configuration)

---

## Main Entry Points

### `generate.py`

Main script for generating ligands from metal complexes.

#### Functions

##### `parse_complex(filename)`
Parses a metal complex from XYZ file format.

**Parameters:**
- `filename` (str): Path to XYZ file containing the metal complex

**Returns:**
- `data_list` (list): List of PyTorch Geometric Data objects representing the complex

**Example:**
```python
from generate import parse_complex

# Parse a metal complex
complex_data = parse_complex("complex.xyz")
```

##### `reform_pos(xyz_file)`
Reorders atoms in XYZ file to put metal first.

**Parameters:**
- `xyz_file` (str): Path to XYZ file

**Example:**
```python
from generate import reform_pos

# Reorder atoms with metal first
reform_pos("complex.xyz")
```

##### `main(outdir, model, complex, batch_size=64, n_samples=1, ligand_sizes='random')`
Main generation function.

**Parameters:**
- `outdir` (str): Output directory for generated ligands
- `model` (str): Path to trained model checkpoint
- `complex` (str): Path to input complex XYZ file
- `batch_size` (int): Batch size for generation (default: 64)
- `n_samples` (int): Number of samples to generate (default: 1)
- `ligand_sizes` (str): Ligand size specification (default: 'random')

**Example:**
```bash
python generate.py --outdir ./generated --model ./checkpoints/model.ckpt --complex ./input.xyz --n_samples 10
```

### `train.py`

Main training script for the LigandDiff model.

#### Functions

##### `find_last_checkpoint(checkpoints_dir)`
Finds the latest checkpoint in a directory.

**Parameters:**
- `checkpoints_dir` (str): Directory containing checkpoints

**Returns:**
- `str`: Path to the latest checkpoint file

**Example:**
```python
from train import find_last_checkpoint

latest_ckpt = find_last_checkpoint("./checkpoints")
```

##### `main(args)`
Main training function.

**Parameters:**
- `args`: ArgumentParser object with training configuration

**Example:**
```bash
python train.py --config config.yml --exp_name experiment1 --batch_size 32
```

### `sampling.py`

Script for sampling from trained models.

#### Functions

##### `reform_data(dataset, device, ligand_sizes='random')`
Reforms dataset for sampling with specified ligand sizes.

**Parameters:**
- `dataset`: PyTorch Geometric dataset
- `device`: PyTorch device
- `ligand_sizes` (str): Ligand size specification

**Returns:**
- `new_data` (list): Reformed dataset

**Example:**
```python
from sampling import reform_data

reformed_data = reform_data(dataset, device, ligand_sizes='random')
```

##### `get_ligand_size(ligand_size='random')`
Gets ligand size for generation.

**Parameters:**
- `ligand_size` (str): Size specification ('random' or integer)

**Returns:**
- `int`: Ligand size

**Example:**
```python
from sampling import get_ligand_size

size = get_ligand_size('random')  # Random size between 6-20
size = get_ligand_size('10')      # Fixed size of 10
```

---

## Core Model Classes

### `DDPM` (src/lightning.py)

Main PyTorch Lightning module for the diffusion model.

#### Constructor

```python
DDPM(
    data_path, train_data, val_data,
    in_node_nf, n_dims, ligand_group_node_nf,
    hidden_nf, attention, n_layers, normalization_factor, normalize_factors,
    drop_rate, activation, tanh, norm_constant,
    inv_sublayers, sin_embedding, aggregation_method, normalization,
    diffusion_steps, diffusion_noise_schedule, diffusion_noise_precision, diffusion_loss_type,
    lr, batch_size, torch_device, model, test_epochs,
    samples_dir=None, center_of_mass='context', clip_grad=False
)
```

**Key Parameters:**
- `in_node_nf` (int): Number of input node features
- `n_dims` (int): Number of spatial dimensions (3)
- `hidden_nf` (int): Number of hidden features
- `diffusion_steps` (int): Number of diffusion timesteps
- `lr` (float): Learning rate
- `batch_size` (int): Training batch size

#### Methods

##### `forward(data)`
Forward pass through the model.

**Parameters:**
- `data`: PyTorch Geometric Data object

**Returns:**
- Model output

##### `sample_and_analyze(dataset, batch_size=None, outdir='generated_samples', animation=False)`
Samples from the model and analyzes results.

**Parameters:**
- `dataset`: Input dataset
- `batch_size` (int): Sampling batch size
- `outdir` (str): Output directory
- `animation` (bool): Whether to generate animations

**Example:**
```python
from src.lightning import DDPM

# Load model
model = DDPM.load_from_checkpoint("model.ckpt")

# Sample and analyze
model.sample_and_analyze(dataset, outdir="./samples")
```

##### `sample_chain(data, keep_frames=None)`
Generates sampling chain.

**Parameters:**
- `data`: Input data
- `keep_frames` (int): Number of frames to keep

**Returns:**
- Sampling chain

### `EDM` (src/edm.py)

Equivariant Diffusion Model implementation.

#### Constructor

```python
EDM(
    dynamics, in_node_nf, n_dims,
    timesteps=1000, noise_schedule='learned',
    noise_precision=1e-4, loss_type='vlb',
    norm_values=(1., 1., 1.), norm_biases=(None, 0., 0.)
)
```

#### Methods

##### `forward(x, h, context, ligand_diff, batch_seg, batch_size, ligand_group=None)`
Forward pass through the EDM.

**Parameters:**
- `x`: Position coordinates
- `h`: Node features
- `context`: Context mask
- `ligand_diff`: Ligand diffusion mask
- `batch_seg`: Batch segmentation
- `batch_size` (int): Batch size
- `ligand_group`: Ligand group information

**Returns:**
- Model loss and predictions

##### `sample_chain(x, h, context, ligand_diff, batch_seg, batch_size, ligand_group, keep_frames=None, timesteps=None)`
Generates sampling chain.

**Parameters:**
- `x`: Initial positions
- `h`: Initial features
- `context`: Context mask
- `ligand_diff`: Ligand diffusion mask
- `batch_seg`: Batch segmentation
- `batch_size` (int): Batch size
- `ligand_group`: Ligand group information
- `keep_frames` (int): Number of frames to keep
- `timesteps` (int): Number of timesteps

**Returns:**
- Sampling chain

### `Dynamics` (src/egnn.py)

Equivariant Graph Neural Network for dynamics prediction.

#### Constructor

```python
Dynamics(
    in_node_nf, n_dims, ligand_group_node_nf,
    hidden_nf=32, activation='silu', n_layers=2,
    attention=False, tanh=True, norm_constant=0.00001,
    inv_sublayers=2, sin_embedding=False,
    normalization_factor=100, aggregation_method='sum',
    drop_rate=0.0, device='cpu', model='egnn_dynamics',
    normalization='batch_norm', condition_time=True
)
```

#### Methods

##### `forward(xh, t, ligand_diff, ligand_group, batch_seg)`
Forward pass through the dynamics network.

**Parameters:**
- `xh`: Concatenated positions and features
- `t`: Time embedding
- `ligand_diff`: Ligand diffusion mask
- `ligand_group`: Ligand group information
- `batch_seg`: Batch segmentation

**Returns:**
- Predicted noise

---

## Data Processing

### `molecule_builder.py`

Functions for building and manipulating molecular structures.

#### Functions

##### `build_mol(positions, atom_types, use_openbabel=True)`
Builds RDKit molecule from positions and atom types.

**Parameters:**
- `positions` (torch.Tensor): N x 3 position coordinates
- `atom_types` (torch.Tensor): N atom type indices
- `use_openbabel` (bool): Whether to use OpenBabel for bond creation

**Returns:**
- `rdkit.Chem.Mol`: RDKit molecule object

**Example:**
```python
from src.molecule_builder import build_mol

mol = build_mol(positions, atom_types)
```

##### `extract_ligand(x, onehot, ligand_diff, batch_seg)`
Extracts ligand coordinates and atom types from batch.

**Parameters:**
- `x` (torch.Tensor): Position coordinates
- `onehot` (torch.Tensor): One-hot encoded atom types
- `ligand_diff` (torch.Tensor): Ligand diffusion mask
- `batch_seg` (torch.Tensor): Batch segmentation

**Returns:**
- `list`: List of ligand coordinates and atom types

##### `write_xyz_file(coords, atom_types, filename, metal)`
Writes molecular structure to XYZ file.

**Parameters:**
- `coords` (torch.Tensor): Atomic coordinates
- `atom_types` (torch.Tensor): Atom type indices
- `filename` (str): Output filename
- `metal` (torch.Tensor): Metal atom type

**Example:**
```python
from src.molecule_builder import write_xyz_file

write_xyz_file(coords, atom_types, "molecule", metal_type)
```

##### `sanitycheck(positions, atom_types, metal)`
Performs sanity checks on molecular structure.

**Parameters:**
- `positions` (torch.Tensor): Atomic positions
- `atom_types` (torch.Tensor): Atom types
- `metal` (torch.Tensor): Metal atom type

**Returns:**
- `bool`: Whether structure passes sanity checks

#### Classes

##### `BasicLigandMetrics`
Class for computing ligand quality metrics.

**Methods:**
- `compute_validity(generated)`: Computes validity of generated molecules
- `compute_connectivity(valid)`: Computes connectivity metrics
- `evaluate_rdmols(rdmols)`: Evaluates RDKit molecules

**Example:**
```python
from src.molecule_builder import BasicLigandMetrics

metrics = BasicLigandMetrics()
validity = metrics.compute_validity(generated_mols)
```

---

## Visualization

### `visualizer.py`

Functions for visualizing molecular structures and diffusion chains.

#### Functions

##### `save_xyz_file(path, one_hot, positions, names, batch_seg)`
Saves molecular structures as XYZ files.

**Parameters:**
- `path` (str): Output directory
- `one_hot` (torch.Tensor): One-hot encoded atom types
- `positions` (torch.Tensor): Atomic positions
- `names` (list): Molecule names
- `batch_seg` (torch.Tensor): Batch segmentation

**Example:**
```python
from src.visualizer import save_xyz_file

save_xyz_file("./output", one_hot, positions, names, batch_seg)
```

##### `load_xyz_files(path)`
Loads XYZ files from directory.

**Parameters:**
- `path` (str): Directory path

**Returns:**
- `list`: List of XYZ file paths

##### `load_molecule_xyz(file)`
Loads molecule from XYZ file.

**Parameters:**
- `file` (str): XYZ file path

**Returns:**
- `tuple`: (positions, one_hot, charges)

##### `plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, fragment_mask=None)`
Plots molecular structure in 3D.

**Parameters:**
- `ax`: Matplotlib 3D axis
- `positions` (torch.Tensor): Atomic positions
- `atom_type` (torch.Tensor): Atom types
- `alpha` (float): Transparency
- `spheres_3d` (bool): Whether to use 3D spheres
- `hex_bg_color` (str): Background color
- `fragment_mask` (torch.Tensor): Fragment mask

##### `plot_data3d(positions, atom_type, camera_elev=0, camera_azim=0, save_path=None, spheres_3d=False, bg='black', alpha=1., fragment_mask=None)`
Creates 3D plot of molecular data.

**Parameters:**
- `positions` (torch.Tensor): Atomic positions
- `atom_type` (torch.Tensor): Atom types
- `camera_elev` (float): Camera elevation
- `camera_azim` (float): Camera azimuth
- `save_path` (str): Path to save plot
- `spheres_3d` (bool): Whether to use 3D spheres
- `bg` (str): Background color
- `alpha` (float): Transparency
- `fragment_mask` (torch.Tensor): Fragment mask

**Example:**
```python
from src.visualizer import plot_data3d

plot_data3d(positions, atom_types, save_path="molecule.png")
```

##### `visualize_chain(path, spheres_3d=False, bg="black", alpha=1.0, wandb=None, mode="chain", fragment_mask=None)`
Visualizes diffusion chain.

**Parameters:**
- `path` (str): Path to chain data
- `spheres_3d` (bool): Whether to use 3D spheres
- `bg` (str): Background color
- `alpha` (float): Transparency
- `wandb`: Weights & Biases logger
- `mode` (str): Visualization mode
- `fragment_mask` (torch.Tensor): Fragment mask

---

## Utilities

### `utils.py`

Utility functions for the LigandDiff project.

#### Classes

##### `Queue`
Simple queue implementation for tracking metrics.

**Methods:**
- `add(item)`: Adds item to queue
- `mean()`: Computes mean of queue items
- `std()`: Computes standard deviation of queue items

**Example:**
```python
from src.utils import Queue

queue = Queue(max_len=50)
queue.add(0.5)
mean_val = queue.mean()
```

##### `Logger`
Logger class for redirecting stdout/stderr to files.

**Constructor:**
```python
Logger(logpath, syspart=sys.stdout)
```

**Example:**
```python
from src.utils import Logger

logger = Logger("log.txt", sys.stdout)
```

##### `EMA`
Exponential Moving Average implementation.

**Constructor:**
```python
EMA(beta)
```

**Methods:**
- `update_model_average(ma_model, current_model)`: Updates model average
- `update_average(old, new)`: Updates average value

#### Functions

##### `get_grad_norm(parameters, norm_type=2.0)`
Computes gradient norm.

**Parameters:**
- `parameters`: Model parameters
- `norm_type` (float): Norm type

**Returns:**
- `torch.Tensor`: Gradient norm

##### `remove_mean(x)`
Removes mean from tensor.

**Parameters:**
- `x` (torch.Tensor): Input tensor

**Returns:**
- `torch.Tensor`: Centered tensor

##### `assert_mean_zero(x)`
Asserts tensor has zero mean.

**Parameters:**
- `x` (torch.Tensor): Input tensor

##### `center_gravity_zero_gaussian_log_likelihood(x)`
Computes log likelihood for zero-centered Gaussian.

**Parameters:**
- `x` (torch.Tensor): Input tensor

**Returns:**
- `torch.Tensor`: Log likelihood

##### `sample_center_gravity_zero_gaussian(size, device)`
Samples from zero-centered Gaussian.

**Parameters:**
- `size` (tuple): Sample size
- `device`: PyTorch device

**Returns:**
- `torch.Tensor`: Sampled tensor

##### `random_rotation(x)`
Applies random rotation to tensor.

**Parameters:**
- `x` (torch.Tensor): Input tensor

**Returns:**
- `torch.Tensor`: Rotated tensor

##### `disable_rdkit_logging()`
Disables RDKit logging.

##### `log(*args)`
Logs messages with timestamp.

**Example:**
```python
from src.utils import log

log("Training started")
```

---

## Constants and Configuration

### `const.py`

Constants and configuration for the LigandDiff project.

#### Atom Type Mappings

```python
ATOM2IDX = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'Br': 4, 'Cl': 5, 'P': 6, 'F': 7}
IDX2ATOM = {0: 'C', 1: 'N', 2: 'O', 3: 'S', 4: 'Br', 5: 'Cl', 6: 'P', 7: 'F'}
```

#### Metal Mappings

```python
idx2metals = {24:'Cr', 25:'Mn', 26:'Fe', 27:'Co', 28:'Ni', 29:'Cu', 30:'Zn', 44:'Ru', 46:'Pd', 78:'Pt'}
metals = ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Pd', 'Pt']
```

#### Nuclear Charges

```python
CHARGES = {'C': 6, 'O': 8, 'N': 7, 'S': 16, 'Cl': 17, 'P': 15, 'Br': 35, 'F': 9,
           'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30, 'Ru':44, 'Pd':46, 'Pt':78}
```

#### Bond Lengths

The module contains comprehensive bond length dictionaries:
- `BONDS_1`: Single bond lengths
- `BONDS_2`: Double bond lengths  
- `BONDS_3`: Triple bond lengths

#### Usage Examples

```python
from src import const

# Get atom index
atom_idx = const.ATOM2IDX['C']  # Returns 0

# Get atom symbol
atom_symbol = const.IDX2ATOM[0]  # Returns 'C'

# Get nuclear charge
charge = const.CHARGES['Fe']  # Returns 26

# Check if atom is metal
is_metal = 'Fe' in const.metals  # Returns True
```

---

## Configuration

### `config.yml`

YAML configuration file for training parameters.

**Example Configuration:**
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

---

## Usage Examples

### Training a Model

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

### Generating Ligands

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

### Sampling from Trained Model

```python
from sampling import main

# Sample from model
main(
    outdir="./samples",
    model="./checkpoints/model.ckpt",
    dataset="./test_data.pt",
    batch_size=64
)
```

### Visualizing Results

```python
from src.visualizer import plot_data3d, visualize_chain

# Plot single molecule
plot_data3d(positions, atom_types, save_path="molecule.png")

# Visualize diffusion chain
visualize_chain("./chain_data", save_path="chain.gif")
```

---

## Error Handling

The codebase includes several exception classes:

### `FoundNaNException` (src/utils.py)

Raised when NaN values are detected in tensors.

**Usage:**
```python
from src.utils import FoundNaNException

try:
    # Model forward pass
    output = model(input_data)
except FoundNaNException as e:
    print(f"NaN detected in tensor: {e.x}")
```

---

## Performance Considerations

1. **GPU Usage**: The model supports both CPU and GPU training. Use `torch_device='cuda:0'` for GPU training.

2. **Batch Size**: Adjust batch size based on available memory. Larger batch sizes generally improve training stability.

3. **Gradient Clipping**: Enable gradient clipping with `clip_grad=True` for stable training.

4. **Memory Management**: Use `torch.no_grad()` for inference to reduce memory usage.

---

## Dependencies

The project requires the following key dependencies:

- PyTorch
- PyTorch Geometric
- PyTorch Lightning
- RDKit
- OpenBabel
- NumPy
- Matplotlib
- Weights & Biases (optional)

Install dependencies with:
```bash
pip install torch torch-geometric pytorch-lightning rdkit-python openbabel-python numpy matplotlib wandb
```

---

This documentation covers all public APIs, functions, and components of the LigandDiff project. For additional information, refer to the original paper and repository.