# Neural Network Architecture Documentation

## Overview

This document provides detailed documentation for the neural network components used in LigandDiff, including the Equivariant Graph Neural Network (EGNN), Graph Vector Perceptron (GVP), and related architectural components.

## Table of Contents

1. [EGNN Architecture](#egnn-architecture)
2. [GVP Components](#gvp-components)
3. [Activation Functions](#activation-functions)
4. [Layer Components](#layer-components)
5. [Noise Schedules](#noise-schedules)

---

## EGNN Architecture

### `EGNN` Class (src/egnn.py)

The main Equivariant Graph Neural Network implementation.

#### Constructor

```python
EGNN(
    in_node_nf, in_edge_nf, hidden_nf, device='cpu',
    activation='silu', n_layers=3, attention=False,
    tanh=False, norm_constant=1, inv_sublayers=2,
    sin_embedding=False, normalization_factor=100,
    aggregation_method='sum', norm_diff=True,
    out_node_nf=None, coords_range=15, normalization=None
)
```

**Parameters:**
- `in_node_nf` (int): Number of input node features
- `in_edge_nf` (int): Number of input edge features
- `hidden_nf` (int): Number of hidden features
- `device` (str): Computation device
- `activation` (str): Activation function ('silu', 'siqu', etc.)
- `n_layers` (int): Number of EGNN layers
- `attention` (bool): Whether to use attention
- `tanh` (bool): Whether to use tanh activation
- `norm_constant` (float): Normalization constant
- `inv_sublayers` (int): Number of invariant sublayers
- `sin_embedding` (bool): Whether to use sinusoidal embedding
- `normalization_factor` (float): Normalization factor
- `aggregation_method` (str): Aggregation method ('sum', 'mean')
- `norm_diff` (bool): Whether to normalize differences
- `out_node_nf` (int): Number of output node features
- `coords_range` (float): Coordinate range for sinusoidal embedding
- `normalization` (str): Normalization type ('batch_norm', None)

#### Methods

##### `forward(h, x, edge_index, ligand_diff)`
Forward pass through the EGNN.

**Parameters:**
- `h` (torch.Tensor): Node features [N, in_node_nf]
- `x` (torch.Tensor): Node coordinates [N, 3]
- `edge_index` (torch.Tensor): Edge indices [2, E]
- `ligand_diff` (torch.Tensor): Ligand diffusion mask [N]

**Returns:**
- `tuple`: (updated_h, updated_x)

**Example:**
```python
from src.egnn import EGNN

egnn = EGNN(
    in_node_nf=8,
    in_edge_nf=0,
    hidden_nf=64,
    n_layers=3,
    attention=True
)

h_out, x_out = egnn(h, x, edge_index, ligand_diff)
```

### `EquivariantBlock` Class

A single equivariant block in the EGNN architecture.

#### Constructor

```python
EquivariantBlock(
    hidden_nf, edge_feat_nf=2, device='cpu',
    activation='silu', n_layers=2, attention=True,
    norm_diff=True, tanh=False, coords_range=15,
    norm_constant=1, sin_embedding=None,
    normalization_factor=100, aggregation_method='sum',
    normalization=None
)
```

#### Methods

##### `forward(h, x, edge_index, ligand_diff=None, edge_attr=None)`
Forward pass through the equivariant block.

**Parameters:**
- `h` (torch.Tensor): Node features
- `x` (torch.Tensor): Node coordinates
- `edge_index` (torch.Tensor): Edge indices
- `ligand_diff` (torch.Tensor): Ligand diffusion mask
- `edge_attr` (torch.Tensor): Edge attributes

**Returns:**
- `tuple`: (updated_h, updated_x)

### `EquivariantUpdate` Class

Handles equivariant updates for coordinates.

#### Constructor

```python
EquivariantUpdate(
    hidden_nf, normalization_factor, aggregation_method,
    edges_in_d=1, activation='silu', tanh=False, coords_range=10.0
)
```

#### Methods

##### `forward(h, coord, edge_index, coord_diff, edge_attr=None, ligand_diff=None)`
Performs equivariant coordinate updates.

**Parameters:**
- `h` (torch.Tensor): Node features
- `coord` (torch.Tensor): Node coordinates
- `edge_index` (torch.Tensor): Edge indices
- `coord_diff` (torch.Tensor): Coordinate differences
- `edge_attr` (torch.Tensor): Edge attributes
- `ligand_diff` (torch.Tensor): Ligand diffusion mask

**Returns:**
- `torch.Tensor`: Updated coordinates

### `GCL` Class (Graph Convolutional Layer)

Graph convolutional layer with attention mechanism.

#### Constructor

```python
GCL(
    input_nf, output_nf, hidden_nf, edges_in_d=0,
    activation='silu', attention=False,
    normalization_factor=100, aggregation_method='sum',
    normalization=None
)
```

#### Methods

##### `forward(h, edge_index, edge_attr=None)`
Forward pass through the graph convolutional layer.

**Parameters:**
- `h` (torch.Tensor): Node features
- `edge_index` (torch.Tensor): Edge indices
- `edge_attr` (torch.Tensor): Edge attributes

**Returns:**
- `torch.Tensor`: Updated node features

---

## GVP Components

### `GVPNetwork` Class (src/gvp_model.py)

Graph Vector Perceptron network implementation.

#### Constructor

```python
GVPNetwork(
    in_dims, hidden_dims, out_dims,
    vector_dims, n_layers=3, dropout=0.1
)
```

**Parameters:**
- `in_dims` (tuple): Input scalar and vector dimensions
- `hidden_dims` (tuple): Hidden scalar and vector dimensions
- `out_dims` (tuple): Output scalar and vector dimensions
- `vector_dims` (int): Vector dimension
- `n_layers` (int): Number of layers
- `dropout` (float): Dropout rate

#### Methods

##### `forward(h_V, h_E, edge_index)`
Forward pass through the GVP network.

**Parameters:**
- `h_V` (torch.Tensor): Node features
- `h_E` (torch.Tensor): Edge features
- `edge_index` (torch.Tensor): Edge indices

**Returns:**
- `torch.Tensor`: Updated node features

### `GVP` Class (src/gvp.py)

Individual Graph Vector Perceptron layer.

#### Constructor

```python
GVP(
    in_dims, out_dims, h_dim=None,
    activations=(F.relu, None), vector_gate=False
)
```

**Parameters:**
- `in_dims` (tuple): Input dimensions (scalar, vector)
- `out_dims` (tuple): Output dimensions (scalar, vector)
- `h_dim` (int): Hidden dimension
- `activations` (tuple): Activation functions
- `vector_gate` (bool): Whether to use vector gating

#### Methods

##### `forward(h_V, h_E)`
Forward pass through the GVP layer.

**Parameters:**
- `h_V` (torch.Tensor): Node features
- `h_E` (torch.Tensor): Edge features

**Returns:**
- `torch.Tensor`: Updated features

---

## Activation Functions

### `ScaledSiLU` Class

Scaled Swish/SiLU activation function.

```python
class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor
```

### `SiQU` Class

SiQU activation function (x * SiLU(x)).

```python
class SiQU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._activation = nn.SiLU()

    def forward(self, x):
        return x * self._activation(x)
```

### `DenseLayer` Class

Dense layer with configurable activation and initialization.

#### Constructor

```python
DenseLayer(
    in_features, out_features, bias=True,
    activation=None, weight_init=kaiming_uniform_,
    bias_init=zeros_
)
```

**Parameters:**
- `in_features` (int): Input features
- `out_features` (int): Output features
- `bias` (bool): Whether to include bias
- `activation` (str): Activation function
- `weight_init` (callable): Weight initialization function
- `bias_init` (callable): Bias initialization function

#### Methods

##### `reset_parameters()`
Resets layer parameters with specified initialization.

---

## Layer Components

### `LayerNorm` Class (src/layer_norm.py)

Layer normalization implementation.

#### Constructor

```python
LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)
```

**Parameters:**
- `normalized_shape` (int or tuple): Shape to normalize
- `eps` (float): Epsilon for numerical stability
- `elementwise_affine` (bool): Whether to use learnable parameters

#### Methods

##### `forward(x)`
Applies layer normalization.

**Parameters:**
- `x` (torch.Tensor): Input tensor

**Returns:**
- `torch.Tensor`: Normalized tensor

### `Dropout` Class (src/dropout.py)

Dropout layer implementation.

#### Constructor

```python
Dropout(p=0.5, inplace=False)
```

**Parameters:**
- `p` (float): Dropout probability
- `inplace` (bool): Whether to perform in-place operation

#### Methods

##### `forward(x)`
Applies dropout.

**Parameters:**
- `x` (torch.Tensor): Input tensor

**Returns:**
- `torch.Tensor`: Tensor with dropout applied

---

## Noise Schedules

### `GammaNetwork` Class (src/noise.py)

Learned noise schedule network.

#### Constructor

```python
GammaNetwork()
```

#### Methods

##### `forward(t)`
Computes gamma values for given timesteps.

**Parameters:**
- `t` (torch.Tensor): Timesteps [0, 1]

**Returns:**
- `torch.Tensor`: Gamma values

### `PredefinedNoiseSchedule` Class

Predefined noise schedule implementation.

#### Constructor

```python
PredefinedNoiseSchedule(
    schedule_type, timesteps=1000, precision=1e-4
)
```

**Parameters:**
- `schedule_type` (str): Schedule type ('cosine', 'linear', etc.)
- `timesteps` (int): Number of timesteps
- `precision` (float): Precision for gamma computation

#### Methods

##### `forward(t)`
Computes gamma values using predefined schedule.

**Parameters:**
- `t` (torch.Tensor): Timesteps [0, 1]

**Returns:**
- `torch.Tensor`: Gamma values

---

## Sinusoidal Embeddings

### `SinusoidsEmbeddingNew` Class

Sinusoidal embedding for time and coordinates.

#### Constructor

```python
SinusoidsEmbeddingNew(
    max_res=15., min_res=15. / 2000., div_factor=4
)
```

**Parameters:**
- `max_res` (float): Maximum resolution
- `min_res` (float): Minimum resolution
- `div_factor` (int): Division factor

#### Methods

##### `forward(x)`
Computes sinusoidal embeddings.

**Parameters:**
- `x` (torch.Tensor): Input values

**Returns:**
- `torch.Tensor`: Sinusoidal embeddings

---

## Utility Functions

### `coord2diff(x, edge_index, norm_constant=1)`

Computes coordinate differences between connected nodes.

**Parameters:**
- `x` (torch.Tensor): Node coordinates [N, 3]
- `edge_index` (torch.Tensor): Edge indices [2, E]
- `norm_constant` (float): Normalization constant

**Returns:**
- `torch.Tensor`: Coordinate differences [E, 3]

**Example:**
```python
from src.egnn import coord2diff

diff = coord2diff(x, edge_index)
```

---

## Architecture Diagrams

### EGNN Architecture Flow

```
Input: (h, x, edge_index, ligand_diff)
    ↓
[GCL Layer] → Node feature update
    ↓
[EquivariantUpdate] → Coordinate update
    ↓
[Attention (optional)] → Attention weights
    ↓
[Layer Normalization] → Normalized features
    ↓
Output: (h_out, x_out)
```

### GVP Architecture Flow

```
Input: (h_V, h_E, edge_index)
    ↓
[GVP Layer] → Scalar/vector feature update
    ↓
[Activation] → Non-linear transformation
    ↓
[Dropout] → Regularization
    ↓
Output: Updated features
```

---

## Performance Optimizations

### 1. Memory Efficiency

- Use `torch.no_grad()` for inference
- Implement gradient checkpointing for large models
- Use mixed precision training with `torch.cuda.amp`

### 2. Computational Efficiency

- Vectorized operations for coordinate updates
- Efficient edge indexing with `torch_geometric`
- Parallel processing of graph convolutions

### 3. Numerical Stability

- Layer normalization for feature stability
- Gradient clipping to prevent exploding gradients
- Careful initialization of weights

---

## Usage Examples

### Building a Custom EGNN

```python
from src.egnn import EGNN, Dynamics

# Create EGNN for dynamics prediction
dynamics = Dynamics(
    in_node_nf=8,
    n_dims=3,
    ligand_group_node_nf=6,
    hidden_nf=64,
    n_layers=4,
    attention=True,
    activation='silu'
)

# Create EGNN for general use
egnn = EGNN(
    in_node_nf=8,
    in_edge_nf=0,
    hidden_nf=64,
    n_layers=3,
    attention=True,
    activation='silu'
)
```

### Custom Activation Functions

```python
from src.egnn import ScaledSiLU, SiQU

# Use scaled SiLU
activation = ScaledSiLU()

# Use SiQU
activation = SiQU()
```

### Noise Schedule Configuration

```python
from src.noise import GammaNetwork, PredefinedNoiseSchedule

# Learned noise schedule
gamma_net = GammaNetwork()

# Predefined cosine schedule
gamma_net = PredefinedNoiseSchedule('cosine', timesteps=1000)
```

---

## Troubleshooting

### Common Issues

1. **NaN Values**: Check for numerical instability in coordinate updates
2. **Memory Issues**: Reduce batch size or use gradient checkpointing
3. **Slow Training**: Use GPU acceleration and optimize data loading
4. **Poor Convergence**: Adjust learning rate and normalization parameters

### Debugging Tips

1. Monitor gradient norms during training
2. Check for NaN values in forward pass
3. Verify coordinate updates maintain equivariance
4. Ensure proper initialization of weights

---

This documentation provides comprehensive coverage of the neural network architecture components used in LigandDiff. For implementation details, refer to the source code in the respective modules.