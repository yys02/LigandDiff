# LigandDiff Quick Reference Guide

## Common Operations

### 1. Training

#### Basic Training
```bash
python train.py --config config.yml --exp_name experiment1 --batch_size 32
```

#### Training with Custom Parameters
```bash
python train.py \
  --config config.yml \
  --exp_name my_experiment \
  --batch_size 64 \
  --lr 0.0001 \
  --n_epochs 2000 \
  --device gpu
```

#### Resume Training
```bash
python train.py \
  --config config.yml \
  --exp_name experiment1 \
  --resume experiment1_epoch_50.ckpt
```

### 2. Generation

#### Generate from Complex
```bash
python generate.py \
  --outdir ./generated \
  --model ./checkpoints/model.ckpt \
  --complex ./input.xyz \
  --n_samples 10 \
  --ligand_sizes random
```

#### Generate with Fixed Ligand Size
```bash
python generate.py \
  --outdir ./generated \
  --model ./checkpoints/model.ckpt \
  --complex ./input.xyz \
  --n_samples 5 \
  --ligand_sizes 12
```

### 3. Sampling

#### Sample from Dataset
```bash
python sampling.py \
  --outdir ./samples \
  --model ./checkpoints/model.ckpt \
  --dataset ./test_data.pt \
  --batch_size 64
```

#### Sample with Custom Parameters
```bash
python sampling.py \
  --outdir ./samples \
  --model ./checkpoints/model.ckpt \
  --dataset ./test_data.pt \
  --batch_size 32 \
  --ligand_sizes random
```

## Python API Quick Reference

### Model Loading
```python
from src.lightning import DDPM

# Load trained model
model = DDPM.load_from_checkpoint("model.ckpt")
model.eval()
```

### Data Processing
```python
from generate import parse_complex
from src.molecule_builder import build_mol, write_xyz_file

# Parse complex
complex_data = parse_complex("complex.xyz")

# Build molecule
mol = build_mol(positions, atom_types)

# Write XYZ file
write_xyz_file(coords, atom_types, "output", metal_type)
```

### Visualization
```python
from src.visualizer import plot_data3d, visualize_chain

# Plot molecule
plot_data3d(positions, atom_types, save_path="molecule.png")

# Visualize chain
visualize_chain("./chain_data", save_path="chain.gif")
```

### Utilities
```python
from src.utils import log, Queue, disable_rdkit_logging
from src import const

# Logging
log("Processing started")

# Queue for metrics
queue = Queue(max_len=50)
queue.add(0.5)

# Disable RDKit logging
disable_rdkit_logging()

# Get atom index
atom_idx = const.ATOM2IDX['C']  # Returns 0
```

## Configuration Examples

### Basic Config (`config.yml`)
```yaml
# Model
hidden_nf: 64
n_layers: 4
attention: true
activation: "silu"

# Diffusion
diffusion_steps: 1000
diffusion_noise_schedule: "learned"
diffusion_loss_type: "vlb"

# Training
lr: 0.001
batch_size: 32
n_epochs: 1000

# Data
data_path: "./data"
train_data: "train.pt"
val_data: "val.pt"
```

### Advanced Config
```yaml
# Model
hidden_nf: 128
n_layers: 6
attention: true
activation: "silu"
tanh: true
norm_constant: 0.00001

# Diffusion
diffusion_steps: 2000
diffusion_noise_schedule: "cosine"
diffusion_loss_type: "vlb"
diffusion_noise_precision: 1e-4

# Training
lr: 0.0005
batch_size: 16
n_epochs: 2000
clip_grad: true

# Data
data_path: "./data"
train_data: "train.pt"
val_data: "val.pt"
center_of_mass: "context"
```

## Common Parameters

### Model Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_nf` | int | 64 | Hidden features |
| `n_layers` | int | 4 | Number of layers |
| `attention` | bool | true | Use attention |
| `activation` | str | "silu" | Activation function |
| `tanh` | bool | false | Use tanh activation |

### Diffusion Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diffusion_steps` | int | 1000 | Number of timesteps |
| `noise_schedule` | str | "learned" | Noise schedule type |
| `loss_type` | str | "vlb" | Loss function type |
| `noise_precision` | float | 1e-4 | Noise precision |

### Training Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 0.001 | Learning rate |
| `batch_size` | int | 32 | Batch size |
| `n_epochs` | int | 1000 | Number of epochs |
| `clip_grad` | bool | false | Gradient clipping |

## File Formats

### Input XYZ Format
```
3

Fe 0.000 0.000 0.000
C 1.500 0.000 0.000
N 2.500 0.000 0.000
```

### Output XYZ Format
```
4

Fe 0.000 0.000 0.000
C 1.500 0.000 0.000
N 2.500 0.000 0.000
O 1.000 1.000 0.000
```

## Error Handling

### Common Errors and Solutions

#### CUDA Out of Memory
```python
# Reduce batch size
batch_size = 16  # Instead of 64

# Use gradient checkpointing
model = DDPM(..., clip_grad=True)

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input_data)
```

#### NaN Values
```python
from src.utils import FoundNaNException

try:
    output = model(input_data)
except FoundNaNException as e:
    print(f"NaN detected: {e.x}")
    # Check input data normalization
```

#### RDKit Errors
```python
from src.utils import disable_rdkit_logging

# Disable RDKit logging
disable_rdkit_logging()

# Check molecule validity
from src.molecule_builder import sanitycheck
is_valid = sanitycheck(positions, atom_types, metal)
```

## Performance Tips

### GPU Optimization
```python
# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Mixed precision
from torch.cuda.amp import GradScaler
scaler = GradScaler()
```

### Memory Management
```python
# Use no_grad for inference
with torch.no_grad():
    output = model(input_data)

# Clear cache
torch.cuda.empty_cache()
```

### Batch Processing
```python
# Process in smaller batches
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    output = model(batch)
```

## Debugging Commands

### Check Model Parameters
```python
# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### Monitor Training
```python
from src.utils import log

# Log metrics
log(f"Loss: {loss.item():.4f}")
log(f"Accuracy: {accuracy:.4f}")

# Check tensor shapes
print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output.shape}")
```

### Validate Data
```python
from src.molecule_builder import sanitycheck

# Check molecule validity
for i, (pos, atoms) in enumerate(molecules):
    is_valid = sanitycheck(pos, atoms, metal)
    if not is_valid:
        print(f"Molecule {i} is invalid")
```

## Integration Examples

### Custom Training Loop
```python
from src.lightning import DDPM
from pytorch_lightning import Trainer

# Custom model
model = DDPM(
    data_path="./data",
    train_data="train.pt",
    val_data="val.pt",
    in_node_nf=8,
    n_dims=3,
    hidden_nf=64,
    n_layers=4,
    diffusion_steps=1000,
    lr=0.001,
    batch_size=32
)

# Custom trainer
trainer = Trainer(
    max_epochs=1000,
    accelerator='gpu',
    devices=1,
    precision=16  # Mixed precision
)

# Train
trainer.fit(model)
```

### Custom Generation
```python
from generate import main

# Generate with custom parameters
main(
    outdir="./custom_generated",
    model="./checkpoints/model.ckpt",
    complex="./input.xyz",
    n_samples=20,
    ligand_sizes='random',
    batch_size=16
)
```

### Custom Visualization
```python
from src.visualizer import plot_data3d

# Custom plot
plot_data3d(
    positions, atom_types,
    camera_elev=30,
    camera_azim=45,
    save_path="custom_molecule.png",
    spheres_3d=True,
    bg='white',
    alpha=0.8
)
```

## Environment Setup

### Conda Environment
```bash
conda create -n liganddiff python=3.8
conda activate liganddiff
pip install torch torch-geometric pytorch-lightning
pip install rdkit-python openbabel-python
pip install numpy matplotlib wandb
```

### Docker Setup
```dockerfile
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN pip install torch-geometric pytorch-lightning
RUN pip install rdkit-python openbabel-python
RUN pip install numpy matplotlib wandb

WORKDIR /workspace
COPY . .

CMD ["python", "train.py", "--config", "config.yml"]
```

---

This quick reference provides the most common operations and code snippets for working with LigandDiff. For detailed documentation, see the [API Documentation](API_DOCUMENTATION.md) and [Neural Network Documentation](NEURAL_NETWORK_DOCUMENTATION.md).