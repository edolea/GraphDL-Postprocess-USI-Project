# MeteoSwiss Graph Deep Learning Project - Complete Documentation

**Project**: Spatiotemporal Postprocessing of Wind NWP Forecasts using Graph Neural Networks
**Institution**: Università della Svizzera italiana (USI) - Graph Deep Learning Course
**Copyright**: 2025, MeteoSwiss
**License**: BSD 3-Clause

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Project Architecture](#project-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Model Architectures](#model-architectures)
5. [Training Pipeline](#training-pipeline)
6. [Loss Functions](#loss-functions)
7. [Probabilistic Layers](#probabilistic-layers)
8. [Visualization & Logging](#visualization--logging)
9. [Configuration System](#configuration-system)
10. [Complete Execution Flow](#complete-execution-flow)

---

## 1. High-Level Overview

### 1.1 Project Purpose

This project addresses the challenge of **postprocessing Numerical Weather Prediction (NWP) forecasts** for wind speed across Switzerland. Raw NWP forecasts from ensemble models often contain systematic biases and don't fully capture local effects. This project uses **Graph Neural Networks (GNNs)** combined with **temporal models** to:

1. **Learn spatial dependencies** between weather stations using graph structures
2. **Capture temporal patterns** in forecast errors over lead time
3. **Provide probabilistic forecasts** with uncertainty quantification
4. **Improve forecast accuracy** by learning from historical forecast-observation pairs

### 1.2 Key Innovation

The project combines three critical elements:

- **Graph Neural Networks**: Model spatial correlations between weather stations (stations close together have similar weather patterns)
- **Temporal Sequence Modeling**: RNNs, TCNs, or WaveNet to capture how forecast errors evolve over lead time (0-96 hours)
- **Probabilistic Forecasting**: Output full probability distributions (LogNormal/Normal) instead of point forecasts, scored using CRPS (Continuous Ranked Probability Score)

### 1.3 Problem Formulation

**Input**:
- Ensemble NWP forecasts for N weather stations over T lead times (0-96 hours)
- Static terrain features (elevation, topographic indices)
- Temporal encodings (hour of day, day of year)
- Shape: `[batch, lead_time, stations, features]` = `[B, T, N, F]`

**Output**:
- Probability distribution over wind speed for each station and lead time
- Shape: `[B, T, N, 1]` → Distribution parameters (μ, σ)

**Graph Structure**:
- Nodes: Weather stations (N ≈ number of stations in Switzerland)
- Edges: Spatial proximity weighted by Gaussian kernel of distance
- k-NN graph ensures computational efficiency

---

## 2. Project Architecture

### 2.1 Directory Structure

```
MeteoSwiss/
├── GraphDL-Postprocess-USI-Project/        # Main project
│   ├── spatiotemporal_postprocessing/      # Core Python package
│   │   ├── configs/                        # YAML configuration files
│   │   │   ├── default_training_conf.yaml
│   │   │   ├── bidirectional_rnn_training_conf.yaml
│   │   │   ├── mlp_training_conf.yaml
│   │   │   └── wavenet_training_conf.yaml
│   │   ├── datasets/                       # Data loading & graph construction
│   │   │   ├── __init__.py
│   │   │   └── datasets.py
│   │   ├── losses/                         # Loss functions
│   │   │   ├── __init__.py
│   │   │   ├── deterministic.py
│   │   │   └── probabilistic.py
│   │   ├── nn/                             # Neural network models
│   │   │   ├── __init__.py
│   │   │   ├── models.py
│   │   │   ├── prototypes.py
│   │   │   └── probabilistic_layers.py
│   │   ├── train.py                        # Main training script
│   │   └── utils.py                        # Visualization utilities
│   ├── imgs/
│   │   └── wind_stations.png               # Station network visualization
│   ├── pyproject.toml                      # Poetry dependencies
│   ├── poetry.lock
│   ├── README.md
│   └── LICENSE
├── spatiotemporal_pp_dataset/              # NetCDF data files (5.8 GB)
│   ├── features.nc                         # NWP forecasts & predictors
│   └── targets.nc                          # Observed wind speeds
└── papers/                                 # Reference documentation
    ├── GNN for Ensemble Weather Forecasts.pdf
    ├── MeteoSwissProjectDescription.pdf
    └── instructions.pages
```

### 2.2 Component Overview

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **train.py** | Main entry point | Hydra config, training loop, MLflow logging |
| **datasets.py** | Data pipeline | NetCDF loading, normalization, graph construction |
| **models.py** | Neural architectures | BiDirectionalSTGNN, MLP, WaveNet, LayeredGraphRNN |
| **probabilistic.py** | Loss functions | CRPS for Normal, LogNormal, Ensemble |
| **probabilistic_layers.py** | Output layers | Distribution parameter prediction |
| **utils.py** | Visualization | Quantile plots, MLflow artifacts |

---

## 3. Data Pipeline

### 3.1 Data Format

**Features File** ([spatiotemporal_pp_dataset/features.nc](spatiotemporal_pp_dataset/features.nc))
- **Format**: NetCDF (64-bit offset)
- **Size**: 5.6 GB
- **Dimensions**: `[forecast_reference_time, lead_time, station, features]`
- **Contains**: 18 predictor variables

**Predictors (18 features)**:
1. `ch2_ensemble_mean:wind_speed` - Ensemble mean wind speed forecast
2. `ch2_ensemble_std:wind_speed` - Ensemble standard deviation
3. `ch2_mean:mean_sea_level_pressure_diff` - Pressure gradient (mean)
4. `ch2_mean:pressure_100m_diff` - 100m pressure difference
5. `ch2_mean:pressure_650m_diff` - 650m pressure difference
6. `ch2_mean:surface_pressure_diff` - Surface pressure gradient
7. `elevation` - Station elevation (static)
8. `distance_alpine_ridge` - Distance to alpine ridge
9. `tpi` - Topographic Position Index
10. `std` - Terrain standard deviation
11. `valley_norm` - Valley normalization
12. `dzdx` - Terrain slope (x-direction)
13. `dzdy` - Terrain slope (y-direction)
14. `d2zdx2` - Terrain curvature (x)
15. `d2zdy2` - Terrain curvature (y)
16. `d2zdxdy` - Cross-derivative
17. `sin_hour` - Cyclical hour encoding
18. `cos_hour` - Cyclical day of year encoding

**Targets File** ([spatiotemporal_pp_dataset/targets.nc](spatiotemporal_pp_dataset/targets.nc))
- **Format**: NetCDF (64-bit offset)
- **Size**: 272 MB
- **Dimensions**: Same as features
- **Contains**: Single variable `obs:wind_speed` (ground truth observations)

### 3.2 XarrayDataset Class

**Location**: [datasets/datasets.py:11-70](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/datasets/datasets.py#L11-L70)

```python
class XarrayDataset(Dataset):
    """Custom PyTorch Dataset for xarray-based NetCDF data"""
```

**Initialization Process**:
1. **Load data**: Transpose from xarray to numpy `[time, lead_time, stations, features]`
2. **Normalize inputs**: Standardize features (zero mean, unit variance)
3. **Keep targets unnormalized**: Required for CRPS-LogNormal compatibility
4. **Store denormalizers**: Closures to reverse normalization

**Key Methods**:

```python
def normalize(self, data):
    """
    Standardization: (x - μ) / σ

    Returns:
        - standardized_data: Normalized array
        - denormalizer: Function to reverse normalization
    """
    data_mean = np.nanmean(data, axis=(0,1,2), keepdims=True)  # Across time, lead_time, stations
    data_std = np.nanstd(data, axis=(0,1,2), keepdims=True)
    standardized_data = (data - data_mean) / data_std

    def denormalizer(x):
        # Closure captures data_mean and data_std
        if isinstance(x, torch.Tensor):
            return (x * torch.Tensor(data_std).to(x.device)) + torch.Tensor(data_mean).to(x.device)
        return (x * data_std) + data_mean

    return standardized_data, denormalizer
```

**Why no target normalization?**
- CRPS for LogNormal distribution requires targets in original range (positive values)
- Standardization can produce negative values, incompatible with LogNormal

**Dataset Properties**:
- `__len__()`: Returns number of forecast reference times (time dimension)
- `__getitem__(idx)`: Returns `(x, y)` tensors for a specific time
- `stations`, `forecasting_times`, `lead_times`, `features`: Dimension accessors

### 3.3 Graph Construction

**Location**: [datasets/datasets.py:73-115](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/datasets/datasets.py#L73-L115)

```python
def get_graph(lat, lon, knn=10, threshold=None):
    """
    Constructs spatial graph from station coordinates

    Args:
        lat: Latitude array [N stations]
        lon: Longitude array [N stations]
        knn: Number of nearest neighbors to connect
        threshold: Minimum edge weight (prune weak connections)

    Returns:
        adj: Adjacency matrix [N, N] with edge weights
    """
```

**Graph Construction Algorithm**:

1. **Compute Pairwise Distances (Haversine)**:
   ```python
   def haversine(lat1, lon1, lat2, lon2, radius=6371):
       """Great-circle distance in kilometers"""
       lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
       delta_lat = lat2 - lat1
       delta_lon = lon2 - lon1
       a = sin(Δlat/2)² + cos(lat1) * cos(lat2) * sin(Δlon/2)²
       c = 2 * atan2(√a, √(1-a))
       return radius * c
   ```
   - Result: Distance matrix `D[i,j]` = km between station i and j

2. **Apply Gaussian Kernel**:
   ```python
   def gaussian_kernel(x, theta=None):
       if theta is None:
           theta = x.std()  # Bandwidth = std of distances
       weights = np.exp(-np.square(x / theta))  # w_ij = exp(-(d_ij/θ)²)
       return weights
   ```
   - Converts distances to similarity weights (0 to 1)
   - Closer stations → higher weights

3. **k-Nearest Neighbors Selection**:
   ```python
   adj = top_k(adj, knn, include_self=True, keep_values=True)
   ```
   - For each station, keep only k strongest connections
   - `include_self=True`: Self-loops (station influences itself)
   - Reduces edge count from N² to N*k for efficiency

4. **Threshold Pruning** (optional):
   ```python
   if threshold is not None:
       adj[adj < threshold] = 0  # Remove weak edges
   ```

**Default Configuration**:
- k-NN: 5 neighbors
- Threshold: 0.6
- Result: Sparse adjacency matrix with local connectivity

### 3.4 PostprocessDatamodule

**Location**: [datasets/datasets.py:117-140](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/datasets/datasets.py#L117-L140)

```python
class PostprocessDatamodule():
    """Container for train/val/test datasets and graph"""

    def __init__(self, train_dataset, val_dataset, test_dataset, adj_matrix=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.adj_matrix = adj_matrix
        self.num_edges = (adj_matrix != 0).sum() if adj_matrix is not None else 0
```

### 3.5 Data Loading Function

**Location**: [datasets/datasets.py:141-212](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/datasets/datasets.py#L141-L212)

```python
def get_datamodule(ds, ds_targets, predictors, lead_time_hours,
                   val_split, target_var, test_start_date,
                   train_val_end_date=None, return_graph=True,
                   graph_kwargs=None):
    """
    Complete data pipeline: split, normalize, create datasets and graph
    """
```

**Execution Flow**:

1. **Temporal Splitting**:
   ```python
   test_datetime = np.datetime64(test_start_date)  # e.g., '2024-05-16'
   train_val_datetime = np.datetime64(train_val_end_date)  # e.g., '2023-09-30'

   # Gap prevents data leakage (test uses future forecasts)
   input_data_train_val = input_data.sel(forecast_reference_time=slice(None, train_val_datetime))
   test_input_data = input_data.sel(forecast_reference_time=slice(test_datetime, None))
   ```

2. **Train/Val Split**:
   ```python
   train_val_rtimes = len(input_data_train_val['forecast_reference_time'])
   split_index = int(train_val_rtimes * (1.0 - val_split))  # e.g., 80/20

   train_input_data = input_data_train_val.isel(forecast_reference_time=slice(0, split_index))
   val_input_data = input_data_train_val.isel(forecast_reference_time=slice(split_index, None))
   ```

3. **Lead Time Filtering**:
   ```python
   input_data = input_data.sel(lead_time=slice(None, np.timedelta64(lead_time_hours, 'h')))
   ```
   - Default: 96 hours (4 days forecast horizon)

4. **Predictor/Target Selection**:
   ```python
   input_data = ds[predictors]  # Select 18 predictor variables
   target_data = ds_targets[[target_var]]  # Select 'obs:wind_speed'
   ```

5. **Graph Construction**:
   ```python
   lat = ds.latitude.data
   lon = ds.longitude.data
   adj_matrix = get_graph(lat=lat, lon=lon, **graph_kwargs)
   ```

6. **Dataset Creation**:
   ```python
   return PostprocessDatamodule(
       train_dataset=XarrayDataset(train_input_data, train_target_data),
       val_dataset=XarrayDataset(val_input_data, val_target_data),
       test_dataset=XarrayDataset(test_input_data, test_target_data),
       adj_matrix=adj_matrix
   )
   ```

**Data Shapes at Each Stage**:
```
Raw NetCDF:     [forecast_reference_time, lead_time, station, features]
                ~[500 days, 97 hours, N stations, 18 features]

After filtering: [500, 97, N, 18]
Train/Val split: Train [400, 97, N, 18] | Val [100, 97, N, 18]
Test split:      Test [~200, 97, N, 18]

PyTorch batches: [B, T, N, F] where B = batch_size (e.g., 64)
```

---

## 4. Model Architectures

### 4.1 LayeredGraphRNN (Base Class)

**Location**: [nn/models.py:10-69](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/nn/models.py#L10-L69)

```python
class LayeredGraphRNN(nn.Module):
    """
    Recurrent Graph Neural Network with stacked message-passing layers
    Processes sequences forward or backward in time
    """
```

**Architecture**:
```
Input [B, T, N, F]
    ↓
Encoder: Linear(F → hidden_size)
    ↓
For each timestep t (forward or backward):
    For each layer l:
        state_l[t] = GatedGraphNetwork([state_l[t-1], x[t]], edge_index)
    state[t] = state[t] + state[t-1]  # Skip connection in time
    ↓
Output: states [B, T, N, hidden_size*n_layers]
```

**Key Components**:

1. **Input Encoder**:
   ```python
   self.input_encoder = nn.Linear(input_size, hidden_size)
   ```
   - Projects raw features to hidden dimension

2. **Stacked GatedGraphNetwork Layers**:
   ```python
   for _ in range(n_layers):
       layers_.append(GatedGraphNetwork(
           input_size=hidden_size*2,  # [state, input] concatenation
           output_size=hidden_size
       ))
   ```
   - From TSL library (torch-spatiotemporal)
   - Performs message passing on graph
   - Each layer updates node states based on neighbors

3. **Recurrent Processing**:
   ```python
   def iterate_layers(self, state, x, edge_index):
       """Process one timestep through all layers"""
       state_ = rearrange(state, "b n ... (h l) -> l b n ... h", l=self.n_layers)
       for l, layer in enumerate(self.mp_layers):
           state_in_ = state_[l]
           input_ = torch.cat([state_in_, x], dim=-1)  # Recurrent connection
           input_ = self.dropout(input_)
           x = layer(input_, edge_index)
       return torch.cat(output, dim=-1)
   ```

4. **Forward Pass**:
   ```python
   def forward(self, x, edge_index):
       batch_size, win_size, num_nodes, num_feats = x.size()
       state = torch.zeros(batch_size, num_nodes, self.state_size, device=x.device)

       # Direction control
       t0 = 0 if self.mode == 'forwards' else win_size - 1
       tn = win_size if self.mode == 'forwards' else -1
       step = 1 if self.mode == 'forwards' else -1

       for t in range(t0, tn, step):
           x_ = self.input_encoder(x[:,t])
           state_out = self.iterate_layers(state=state, x=x_, edge_index=edge_index)
           state = state_out + state  # Skip connection
           states.append(state)

       return torch.stack(states, dim=1)  # [B, T, N, hidden_size*n_layers]
   ```

**Why Two Modes (forwards/backwards)?**
- Bidirectional RNNs capture patterns from both directions
- Forward: t=0→96 captures how errors evolve
- Backward: t=96→0 captures dependencies in reverse
- Concatenating both provides richer representations

### 4.2 BiDirectionalSTGNN

**Location**: [nn/models.py:72-108](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/nn/models.py#L72-L108)

```python
class BiDirectionalSTGNN(nn.Module):
    """
    Bidirectional Spatiotemporal Graph Neural Network
    Combines forward and backward LayeredGraphRNN
    """
```

**Architecture Diagram**:
```
Input [B, T, N, F]
    ↓
Encoder: Linear(F → hidden_size)
    ↓
+ Station Embeddings [N, hidden_size]
    ↓
┌─────────────────────────────┐
│ Forward RNN                 │  →  states_fwd [B, T, N, H*L]
│ (t=0 → t=96)                │
└─────────────────────────────┘
┌─────────────────────────────┐
│ Backward RNN                │  →  states_bwd [B, T, N, H*L]
│ (t=96 → t=0)                │
└─────────────────────────────┘
    ↓
Concatenate: [states_fwd, states_bwd] → [B, T, N, 2*H*L]
    ↓
+ Skip Connection from input → [B, T, N, 2*H*L]
    ↓
Readout: Linear(2*H*L → H) + BatchNorm + SiLU + Dropout + Linear(H → H)
    ↓
Probabilistic Layer: Linear(H → 2) → (μ, σ) → Distribution
    ↓
Output: LogNormal/Normal Distribution [B, T, N, 1]
```

**Component Breakdown**:

1. **Station Embeddings**:
   ```python
   self.station_embeddings = NodeEmbedding(n_stations, hidden_size)
   ```
   - Learnable per-station embeddings
   - Captures station-specific characteristics (location, terrain, etc.)
   - Added to encoded input: `x = x + self.station_embeddings()`

2. **Bidirectional Processing**:
   ```python
   self.forward_model = LayeredGraphRNN(..., mode='forwards')
   self.backward_model = LayeredGraphRNN(..., mode='backwards')

   states_forwards = self.forward_model(x, edge_index)
   states_backwards = self.backward_model(x, edge_index)
   states = torch.cat([states_forwards, states_backwards], dim=-1)
   ```
   - Two separate RNNs process sequence in opposite directions
   - Doubled hidden dimension: `2 * hidden_size * n_layers`

3. **Skip Connection**:
   ```python
   self.skip_conn = nn.Linear(input_size, 2*hidden_size*n_layers)
   states = states + self.skip_conn(x0)  # x0 = original input
   ```
   - Preserves original input information
   - Mitigates vanishing gradients
   - Helps model learn residuals

4. **Readout Network**:
   ```python
   self.readout = nn.Sequential(
       nn.Linear(2*hidden_size*n_layers, hidden_size),
       BatchNorm(in_channels=hidden_size, track_running_stats=False),
       nn.SiLU(),  # Smooth activation (sigmoid * x)
       nn.Dropout(p=dropout_p),
       nn.Linear(hidden_size, hidden_size)
   )
   ```
   - Processes bidirectional states
   - BatchNorm normalizes activations
   - SiLU (Swish) activation: smooth, non-monotonic
   - Dropout for regularization

5. **Probabilistic Output Layer**:
   ```python
   self.output_distr = dist_to_layer[output_dist](input_size=hidden_size)
   return self.output_distr(output)  # Returns Distribution object
   ```
   - Converts hidden states to distribution parameters
   - See [Probabilistic Layers](#7-probabilistic-layers) section

### 4.3 MLP (Baseline)

**Location**: [nn/models.py:112-143](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/nn/models.py#L112-L143)

```python
class MLP(nn.Module):
    """
    Multi-Layer Perceptron baseline
    Ignores graph structure - processes each (time, station) independently
    """
```

**Architecture**:
```
Input [B, T, N, F]
    ↓
For each hidden layer:
    Linear(F → hidden_size)
    Activation (ReLU/Sigmoid/Tanh/LeakyReLU)
    Dropout
    ↓
Skip: Linear(F → final_hidden_size)
    ↓
Output = Layers + Skip
    ↓
Probabilistic Layer → Distribution
```

**Key Characteristics**:
- **No spatial modeling**: Treats each station independently
- **No temporal modeling**: Treats each lead time independently
- **Purpose**: Baseline to measure benefit of graph/temporal structure
- **Skip connection**: Direct path from input to output

**Activation Options**:
```python
activation_map = {
    "relu": nn.ReLU,           # Standard, fast
    "sigmoid": nn.Sigmoid,     # Bounded (0,1)
    "tanh": nn.Tanh,          # Bounded (-1,1)
    "leaky_relu": nn.LeakyReLU # Prevents dying ReLU
}
```

**Forward Pass**:
```python
def forward(self, x, **kwargs):
    # **kwargs ignores edge_index (for compatibility with other models)
    x_skip = self.skip_conn(x)
    x = self.layers(x)
    x = x + x_skip  # Residual connection
    return self.output_distr(x)
```

### 4.4 WaveNet

**Location**: [nn/models.py:146-164](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/nn/models.py#L146-L164)

```python
class WaveNet(nn.Module):
    """
    Graph WaveNet for spatiotemporal forecasting
    Uses dilated causal convolutions + graph convolutions
    """
```

**Architecture** (from TSL library):
```
Input [B, T, N, F]
    ↓
For each layer l (l=0 to n_layers-1):
    Temporal: Dilated Causal Conv1D (dilation = 2^l)
    Spatial: Graph Convolution on adjacency matrix
    Skip connection to output
    ↓
Sum all skip connections
    ↓
Probabilistic Layer → Distribution
```

**Key Parameters**:
- `temporal_kernel_size`: Size of temporal convolution window
- `spatial_kernel_size`: Size of spatial convolution
- `dilation`: Exponential increase (2^l) for larger receptive field
- `dilation_mod`: Resets dilation every N layers (prevents explosion)
- `horizon`: Forecast horizon (time_steps)

**Dilated Convolutions**:
```
Layer 0: dilation=1  →  receptive field = 3 timesteps
Layer 1: dilation=2  →  receptive field = 7 timesteps
Layer 2: dilation=4  →  receptive field = 15 timesteps
...
```
- Exponentially growing receptive field
- Captures long-range dependencies efficiently
- Causal: no future information leakage

**Implementation**:
```python
self.wavenet = GraphWaveNetModel(
    input_size=input_size,
    output_size=hidden_size,
    horizon=time_steps,  # Forecast horizon
    hidden_size=hidden_size,
    ff_size=ff_size,     # Feed-forward network size
    n_layers=n_layers,
    temporal_kernel_size=temporal_kernel_size,
    spatial_kernel_size=spatial_kernel_size,
    dilation=2,          # Base dilation factor
    dilation_mod=3,      # Reset every 3 layers
    n_nodes=n_stations
)
```

### 4.5 TCN_GNN (Prototype)

**Location**: [nn/prototypes.py:48-105](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/nn/prototypes.py#L48-L105)

**Warning**: Experimental model, may not perform optimally.

```python
class TCN_GNN(nn.Module):
    """
    Temporal Convolutional Network + Graph Neural Network
    Alternates between temporal and spatial processing
    """
```

**Architecture**:
```
Input [B, T, N, F]
    ↓
Encoder: Linear(F → hidden)
    ↓
+ Station Embeddings
    ↓
Reshape: [B, T, N, C] → [(B*N), C, T]  (for Conv1D)
    ↓
For each layer l:
    TCN Layer: Causal Conv1D with dilation 2^l
        → Residual + Skip outputs
    Reshape: [(B*N), C, T] → [B, T, N, C]  (for GNN)
    GNN Layer: Message passing on graph
    Reshape: [B, T, N, C] → [(B*N), C, T]  (back to Conv1D)
    ↓
Sum all skip connections + final output
    ↓
Reshape: [(B*N), C, T] → [B, T, N, C]
    ↓
Probabilistic Layer → Distribution
```

**TCN Layer** ([prototypes.py:21-45](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/nn/prototypes.py#L21-L45)):
```python
class TCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_p, causal_conv):
        # Causal convolution prevents future leakage
        self.cconv = CausalConv1d(...) or nn.Conv1d(...)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout1d(p=dropout_p)
        self.downsample = nn.Conv1d(...)  # Match dimensions for residual

    def forward(self, x):
        residual = x
        x = self.cconv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        residual = self.downsample(residual)
        return residual + x, x  # (residual output, skip output)
```

**CausalConv1d** ([prototypes.py:11-19](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/nn/prototypes.py#L11-L19)):
```python
class CausalConv1d(nn.Conv1d):
    """Ensures no information leakage from future to past"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding=0, dilation=dilation, **kwargs)
        self._padding = (kernel_size - 1) * dilation  # Left-pad only

    def forward(self, x):
        x = nn.functional.pad(x, (self._padding, 0))  # Pad left, not right
        return super().forward(x)
```

**Alternating Temporal/Spatial Processing**:
```python
for tcn_l, gnn_l, norm_l in zip(self.tcn_layers, self.gnn_layers, self.norm_layers):
    x, skip = tcn_l(x)              # Temporal: Conv1D on time dimension
    x = self.rearrange_for_gnn(x)   # [(B*N), C, T] → [B, T, N, C]
    x = gnn_l(x, edge_index)        # Spatial: Message passing on graph
    x = norm_l(x)                   # Normalization
    x = self.rearrange_for_tcn(x)   # [B, T, N, C] → [(B*N), C, T]
    skips.append(skip)
```

---

## 5. Training Pipeline

### 5.1 Main Training Script

**Location**: [train.py:1-150](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/train.py#L1-L150)

**Entry Point**:
```python
@hydra.main(version_base="1.1", config_path="./configs", config_name="default_training_conf")
def app(cfg: DictConfig):
    # Hydra automatically loads YAML config and handles overrides
```

**Execution Flow**:

#### Step 1: Configuration Processing
```python
# Hydra resolver for dynamic values
OmegaConf.register_new_resolver("add_one", lambda x: int(x) + 1)

# Parse string representations to Python objects
if OmegaConf.select(cfg, "training.optim.kwargs.betas") is not None:
    cfg.training.optim.kwargs.betas = eval(cfg.training.optim.kwargs.betas)  # "(0.9, 0.999)" → (0.9, 0.999)

if 'hidden_sizes' in cfg.model.kwargs:
    cfg.model.kwargs.hidden_sizes = eval(cfg.model.kwargs.hidden_sizes)  # "[128, 64, 64]" → [128, 64, 64]
```

#### Step 2: Data Loading
```python
# Load NetCDF files
ds = xr.open_dataset(cfg.dataset.features_pth)  # Features
ds_targets = xr.open_dataset(cfg.dataset.targets_pth)  # Targets

# Create datamodule with splits and graph
dm = get_datamodule(
    ds=ds,
    ds_targets=ds_targets,
    val_split=cfg.dataset.val_split,
    test_start_date=cfg.dataset.test_start,
    train_val_end_date=cfg.dataset.train_val_end,
    lead_time_hours=cfg.dataset.hours_leadtime,
    predictors=cfg.dataset.predictors,
    target_var=cfg.dataset.target_var,
    return_graph=True,
    graph_kwargs=cfg.graph_kwargs
)
```

#### Step 3: Graph Preprocessing
```python
# Convert adjacency matrix to edge_index format (PyG format)
adj_matrix = dm.adj_matrix
edge_index, edge_weight = adj_to_edge_index(adj=torch.tensor(adj_matrix))
# edge_index: [2, num_edges] - COO format (source, target indices)
# edge_weight: [num_edges] - Edge weights (currently unused)
```

#### Step 4: Create DataLoaders
```python
batch_size = cfg.training.batch_size  # e.g., 64
train_dataloader = DataLoader(dm.train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dm.val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dm.test_dataset, batch_size=batch_size, shuffle=True)

# Sanity check: all datasets have same number of stations
assert dm.train_dataset.stations == dm.val_dataset.stations == dm.test_dataset.stations
```

#### Step 5: Model Initialization
```python
model_kwargs = {
    'input_size': dm.train_dataset.f,  # Number of features (18)
    'n_stations': dm.train_dataset.stations,  # Number of stations
    **cfg.model.kwargs  # Additional model-specific params
}
model = get_model(model_type=cfg.model.type, **model_kwargs)
# get_model is a factory function that returns the appropriate model class
```

#### Step 6: Training Components Setup
```python
epochs = cfg.training.epochs
criterion = get_loss(cfg.training.loss)  # e.g., MaskedCRPSLogNormal

# Dynamic optimizer creation (e.g., Adam, RMSprop, SGD)
optimizer = getattr(torch.optim, cfg.training.optim.algo)(
    model.parameters(),
    **cfg.training.optim.kwargs
)

# Dynamic LR scheduler creation (e.g., CosineAnnealingWarmRestarts)
lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.training.scheduler.algo)(
    optimizer,
    **cfg.training.scheduler.kwargs
)

gradient_clip_value = cfg.training.gradient_clip_value  # e.g., 1.0
```

#### Step 7: Device Setup
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
edge_index = edge_index.to(device)

if device_type == 'cpu':
    torch.set_num_threads(16)  # Optimize CPU performance
```

#### Step 8: MLflow Setup
```python
mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)  # Local or remote URI
mlflow.set_experiment(experiment_name=cfg.logging.experiment_id)

with mlflow.start_run():
    # Log metadata
    mlflow.log_param("device_type", device_type)
    mlflow.log_param("optimizer", type(optimizer).__name__)
    mlflow.log_param("criterion", type(criterion).__name__)
    mlflow.log_param("num_params", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Log complete config
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_dict(cfg_dict, 'training_config.json')
    mlflow.log_params(cfg_dict)  # Flatten and log as params
```

### 5.2 Training Loop

```python
total_iter = 0
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    # TRAINING PHASE
    for batch_idx, (x_batch, y_batch) in enumerate(train_dataloader):
        x_batch = x_batch.to(device)  # [B, T, N, F]
        y_batch = y_batch.to(device)  # [B, T, N, 1]

        optimizer.zero_grad()

        # Forward pass
        predictions = model(x_batch, edge_index=edge_index)  # Returns Distribution

        # Compute loss
        loss = criterion(predictions, y_batch)

        # Backward pass
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()
        mlflow.log_metric("loss", loss.item(), step=total_iter)
        total_iter += 1

    # Log epoch-level metrics
    avg_loss = total_loss / len(train_dataloader)
    mlflow.log_metric("train_loss", avg_loss, step=epoch)
```

**Key Training Details**:

1. **Gradient Clipping**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
   ```
   - Prevents exploding gradients (common in RNNs)
   - Clips gradient norm to maximum value (e.g., 1.0)
   - Formula: `g_clipped = g * clip_value / max(||g||, clip_value)`

2. **Loss Computation**:
   ```python
   loss = criterion(predictions, y_batch)
   ```
   - `predictions`: Distribution object (LogNormal/Normal)
   - `y_batch`: Ground truth observations
   - Criterion computes CRPS (see [Loss Functions](#6-loss-functions))

3. **Learning Rate Scheduling**:
   ```python
   lr_scheduler.step(epoch=epoch)
   for group, lr in enumerate(lr_scheduler.get_last_lr()):
       mlflow.log_metric(f'lr_{group}', lr, step=epoch)
   ```
   - Updates learning rate based on schedule
   - Logs current LR for monitoring

### 5.3 Validation Loop

```python
# VALIDATION LOOP
val_loss = 0
val_loss_original_range = 0
model.eval()

with torch.no_grad():
    tgt_denormalizer = dm.val_dataset.target_denormalizer

    for batch_idx, (x_batch, y_batch) in enumerate(val_dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        predictions = model(x_batch, edge_index=edge_index)

        # Loss on normalized targets
        val_loss += criterion(predictions, y_batch).item()

        # Loss on original range (denormalized)
        val_loss_original_range += criterion(
            tgt_denormalizer(predictions),
            tgt_denormalizer(y_batch)
        ).item()

avg_val_loss = val_loss / len(val_dataloader)
avg_val_loss_or = val_loss_original_range / len(val_dataloader)

mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
mlflow.log_metric("val_loss_original_range", avg_val_loss_or, step=epoch)
```

**Why Two Validation Metrics?**
- **Normalized**: Consistent scale for training stability
- **Original range**: True performance on physical units (m/s)
- Note: Targets are NOT normalized (for CRPS-LogNormal), but this pattern is preserved for consistency

### 5.4 Prediction Visualization

```python
# Optional plotting every 10 epochs
if epoch % 10 == 0:
    with torch.no_grad():
        x_val_batch, y_val_batch = next(iter(val_dataloader))
        x_val_batch = x_val_batch.to(device)
        y_val_batch = y_val_batch.to(device)
        val_predictions = model(x_val_batch, edge_index=edge_index)

        log_prediction_plots(
            x=x_val_batch,
            y=y_val_batch,
            pred_dist=val_predictions,
            example_indices=[0, 0, 0, 0],  # All from first batch element
            stations=[1, 2, 3, 4],         # Stations to plot
            epoch=epoch,
            input_denormalizer=dm.val_dataset.input_denormalizer
        )
```

---

## 6. Loss Functions

### 6.1 CRPS (Continuous Ranked Probability Score)

**Mathematical Definition**:

For a probabilistic forecast F and observation y:
```
CRPS(F, y) = ∫_{-∞}^{∞} [F(x) - 1{y ≤ x}]² dx
```

**Interpretation**:
- Proper scoring rule for probabilistic forecasts
- Generalizes MAE to distributions
- Lower is better
- Units: same as target variable (m/s for wind speed)
- Penalizes both:
  - **Bias**: Mean error (sharpness)
  - **Spread**: Variance error (calibration)

### 6.2 MaskedCRPSNormal

**Location**: [losses/probabilistic.py:6-26](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/losses/probabilistic.py#L6-L26)

```python
class MaskedCRPSNormal(nn.Module):
    """CRPS for Normal distribution with NaN masking"""

    def forward(self, pred, y):
        # Handle missing observations (NaN)
        mask = ~torch.isnan(y)
        y = y[mask]
        mu = pred.loc[mask].flatten()      # Mean
        sigma = pred.scale[mask].flatten() # Std deviation
```

**Analytical Formula** (Gneiting & Raftery, 2007):

For Y ~ Normal(μ, σ), observed value y:
```
CRPS(Normal(μ, σ), y) = σ * [ω * (2Φ(ω) - 1) + 2φ(ω) - 1/√π]

where:
    ω = (y - μ) / σ           (standardized residual)
    Φ(ω) = CDF of standard normal at ω
    φ(ω) = PDF of standard normal at ω
```

**Implementation**:
```python
normal = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))

scaled = (y - mu) / sigma  # ω

Phi = normal.cdf(scaled)   # Φ(ω)
phi = torch.exp(normal.log_prob(scaled))  # φ(ω)

crps = sigma * (
    scaled * (2 * Phi - 1) +
    2 * phi -
    (1 / torch.sqrt(torch.tensor(torch.pi, device=sigma.device)))
)

return crps.mean()
```

**Masking Behavior**:
- Weather observations often have missing data (sensor failures, etc.)
- `torch.isnan(y)` identifies missing values
- Only valid observations contribute to loss
- Prevents gradient corruption from NaNs

### 6.3 MaskedCRPSLogNormal

**Location**: [losses/probabilistic.py:28-63](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/losses/probabilistic.py#L28-L63)

```python
class MaskedCRPSLogNormal(nn.Module):
    """CRPS for LogNormal distribution with NaN masking"""

    def __init__(self):
        super().__init__()
        self.i = 0  # Iteration counter for debugging

    def forward(self, pred, y):
        mask = ~torch.isnan(y)
        y = y[mask]

        eps = 1e-5
        y += eps  # Avoid y=0 (LogNormal PDF undefined at 0)

        mu = pred.loc[mask].flatten()      # Log-space mean
        sigma = pred.scale[mask].flatten() # Log-space std
```

**Formula** (Baran & Lerch, 2015):

For Y ~ LogNormal(μ, σ), observed value y:
```
CRPS(LogNormal(μ, σ), y) = y * (2Φ(ω) - 1) - 2exp(μ + σ²/2) * [Φ(ω - σ) + Φ(σ/√2) - 1]

where:
    ω = (log(y) - μ) / σ
    Φ = CDF of standard normal
```

**Implementation**:
```python
normal = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))

omega = (torch.log(y) - mu) / sigma

ex_input = mu + (sigma**2) / 2

# Numerical stability: clamp to prevent overflow
# exp(15) ≈ 3,269,017 (sufficient for wind speed in m/s)
ex_input = torch.clamp(ex_input, max=15)
mlflow.log_metric('exp_input_debug', ex_input.max(), step=self.i)
self.i += 1

ex = 2 * torch.exp(ex_input)

crps = y * (2 * normal.cdf(omega) - 1.0) - \
       ex * (normal.cdf(omega - sigma) + normal.cdf(sigma / (2**0.5)) - 1.0)

return crps.mean()
```

**Why LogNormal for Wind Speed?**
1. **Non-negativity**: Wind speed ≥ 0 (LogNormal guarantees this)
2. **Right-skewed**: Calm winds common, extreme winds rare
3. **Multiplicative errors**: Forecast errors often proportional to magnitude
4. **Physical validity**: Negative predictions are impossible

**Numerical Stability**:
- **Exponential clamping**: Prevents overflow in `exp(μ + σ²/2)`
  - Mean of LogNormal: E[Y] = exp(μ + σ²/2)
  - Without clamping, large μ or σ → overflow
  - Clamp at 15: allows predictions up to ~3M m/s (absurdly high for wind)
- **Small epsilon**: `y += 1e-5` prevents log(0) when y=0
- **Debug logging**: Tracks maximum exponential input for monitoring

### 6.4 MaskedCRPSEnsemble

**Location**: [losses/probabilistic.py:66-79](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/losses/probabilistic.py#L66-L79)

```python
class MaskedCRPSEnsemble(nn.Module):
    """CRPS for ensemble forecasts using scoringrules library"""

    def forward(self, samples, y):
        # samples: [B, T, N, num_samples] - Monte Carlo samples from distribution
        # y: [B, T, N, 1] - Observations

        mask = ~torch.isnan(y)

        losses = sr.crps_ensemble(y.squeeze(-1), samples.squeeze(-1))
        # sr.crps_ensemble from scoringrules library

        return losses[mask.squeeze(1)].mean()
```

**Use Case**:
- When model outputs samples instead of parametric distribution
- Uses empirical CDF from samples
- More flexible but computationally expensive

**CRPS Ensemble Formula**:
```
CRPS(ensemble, y) = (1/M) Σ|x_i - y| - (1/2M²) ΣΣ|x_i - x_j|

where:
    x_i: ensemble members (samples)
    M: ensemble size
    y: observation
```

### 6.5 Deterministic Loss

**Location**: [losses/deterministic.py](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/losses/deterministic.py)

```python
class MaskedL1Loss(nn.Module):
    """Mean Absolute Error with NaN masking"""

    def forward(self, pred, y):
        mask = ~torch.isnan(y)
        return torch.abs(pred[mask] - y[mask]).mean()
```

**Use**: Baseline for deterministic models (not used in main configs)

---

## 7. Probabilistic Layers

### 7.1 DistributionLayer (Abstract Base)

**Location**: [nn/probabilistic_layers.py:18-46](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/nn/probabilistic_layers.py#L18-L46)

```python
class DistributionLayer(nn.Module, ABC):
    """
    Abstract base class for probabilistic output layers
    Converts hidden states to distribution parameters
    """

    def __init__(self, input_size):
        super().__init__()

        # Get distribution class from torch.distributions
        self.distribution = getattr(torch.distributions, self.name)

        # Linear layer to predict distribution parameters
        self.encoder = nn.Linear(input_size, self.num_parameters)

    @property
    @abstractmethod
    def num_parameters(self):
        """Number of parameters (e.g., 2 for Normal: μ, σ)"""
        pass

    @property
    @abstractmethod
    def name(self):
        """Distribution name (e.g., 'Normal', 'LogNormal')"""
        pass

    @abstractmethod
    def process_params(self, x):
        """Process raw parameters to valid distribution params"""
        pass

    def forward(self, x, return_type='distribution', reparametrized=True, num_samples=1):
        params = self.encoder(x)  # [B, T, N, num_parameters]
        distribution = self.process_params(params)

        if return_type == 'distribution':
            return distribution  # Return Distribution object

        # Return samples
        if reparametrized:
            return distribution.rsample((num_samples,))  # Reparameterization trick
        else:
            return distribution.sample((num_samples,))   # Non-differentiable
```

**Design Pattern**:
- Subclasses implement specific distributions (Normal, LogNormal, etc.)
- Encoder predicts raw parameters
- `process_params` applies constraints (e.g., σ > 0)
- Returns PyTorch Distribution object with `.loc`, `.scale`, `.cdf`, `.icdf`, etc.

### 7.2 SoftplusWithEps

**Location**: [nn/probabilistic_layers.py:8-15](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/nn/probabilistic_layers.py#L8-L15)

```python
class SoftplusWithEps(nn.Module):
    """Ensures positive outputs with minimum threshold"""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.softplus = nn.Softplus()  # softplus(x) = log(1 + exp(x))
        self.eps = eps

    def forward(self, x):
        return self.softplus(x) + self.eps
```

**Purpose**:
- Standard deviations must be strictly positive: σ > 0
- Softplus is smooth and always positive: softplus(x) ≥ 0
- Adding epsilon ensures σ ≥ ε (prevents division by zero)

**Why Softplus?**
```
Softplus(x) = log(1 + exp(x))

Properties:
- Always positive: softplus(x) ≥ 0
- Smooth (differentiable everywhere)
- Approximates ReLU but smoother
- As x → ∞: softplus(x) → x
- As x → -∞: softplus(x) → 0
```

### 7.3 LogNormalLayer

**Location**: [nn/probabilistic_layers.py:49-68](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/nn/probabilistic_layers.py#L49-L68)

```python
class LogNormalLayer(DistributionLayer):
    """Outputs LogNormal distribution parameters"""

    _name = 'LogNormal'

    def __init__(self, input_size):
        super().__init__(input_size=input_size)
        self.get_positive_std = SoftplusWithEps()

    @property
    def name(self):
        return self._name

    @property
    def num_parameters(self):
        return 2  # μ (log-space mean), σ (log-space std)

    def process_params(self, x):
        """
        x: [B, T, N, 2] - Raw predicted parameters
        Returns: LogNormal distribution
        """
        new_moments = x.clone()

        # Ensure σ > 0
        new_moments[..., 1] = self.get_positive_std(x[..., 1])

        # Create LogNormal distribution
        log_normal_dist = self.distribution(
            new_moments[..., 0:1],  # μ (unconstrained)
            new_moments[..., 1:2]   # σ (constrained > 0)
        )

        return log_normal_dist
```

**LogNormal Distribution**:
```
If Y ~ LogNormal(μ, σ), then log(Y) ~ Normal(μ, σ)

Parameters:
    μ: Mean of log(Y) (can be any real number)
    σ: Std of log(Y) (must be > 0)

Properties:
    E[Y] = exp(μ + σ²/2)          (expected value)
    Mode[Y] = exp(μ - σ²)          (most likely value)
    Var[Y] = [exp(σ²) - 1] * exp(2μ + σ²)
    Y ≥ 0 always                   (guaranteed positive)
```

**Parameter Constraints**:
- **μ (mean)**: No constraint, can be any real number
- **σ (std)**: Must be positive → SoftplusWithEps(raw_σ)

### 7.4 NormalLayer

**Location**: [nn/probabilistic_layers.py:71-90](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/nn/probabilistic_layers.py#L71-L90)

```python
class NormalLayer(DistributionLayer):
    """Outputs Normal (Gaussian) distribution parameters"""

    _name = 'Normal'

    def __init__(self, input_size):
        super().__init__(input_size=input_size)
        self.get_positive_std = SoftplusWithEps()

    @property
    def name(self):
        return self._name

    @property
    def num_parameters(self):
        return 2  # μ (mean), σ (std)

    def process_params(self, x):
        new_moments = x.clone()
        new_moments[..., 1] = self.get_positive_std(x[..., 1])

        normal_dist = self.distribution(
            new_moments[..., 0:1],  # μ
            new_moments[..., 1:2]   # σ
        )

        return normal_dist
```

**Normal Distribution**:
```
Y ~ Normal(μ, σ)

Parameters:
    μ: Mean (can be any real number)
    σ: Standard deviation (must be > 0)

Properties:
    E[Y] = μ
    Var[Y] = σ²
    Supports: (-∞, ∞)  (can predict negative values!)
```

**When to Use**:
- Normal: Variables that can be negative (temperature, pressure anomalies)
- LogNormal: Variables that must be positive (wind speed, precipitation)

### 7.5 Distribution Factory

**Location**: [nn/probabilistic_layers.py:93-96](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/nn/probabilistic_layers.py#L93-L96)

```python
# Automatically discover all DistributionLayer subclasses
prob_layers = [
    obj[1] for obj in getmembers(sys.modules[__name__], isclass)
    if issubclass(obj[1], DistributionLayer) and obj[0] != 'DistributionLayer'
]

# Create mapping: distribution name → class
dist_to_layer = {
    l._name: l for l in prob_layers
}

# Usage:
# dist_to_layer['LogNormal'](input_size=64) → LogNormalLayer instance
# dist_to_layer['Normal'](input_size=64) → NormalLayer instance
```

**Extensibility**:
- To add new distribution: create subclass of DistributionLayer
- Automatically registered in `dist_to_layer`
- No need to modify factory code

---

## 8. Visualization & Logging

### 8.1 Prediction Plotting

**Location**: [utils.py:6-54](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/utils.py#L6-L54)

```python
def log_prediction_plots(x, y, pred_dist, example_indices, stations, epoch, input_denormalizer):
    """
    Creates 2x2 subplot visualization of predictions vs observations

    Args:
        x: Input features [B, T, N, F]
        y: Observations [B, T, N, 1]
        pred_dist: Predicted distribution
        example_indices: Which batch elements to plot (length 4)
        stations: Which stations to plot (length 4)
        epoch: Current epoch number
        input_denormalizer: Function to denormalize inputs
    """
```

**Execution Flow**:

1. **Denormalize Inputs**:
   ```python
   x = input_denormalizer(x)  # Bring to original range (m/s)
   x = x.detach().cpu().numpy()
   y = y.detach().cpu().numpy()
   ```

2. **Compute Quantiles**:
   ```python
   quantile_levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
   quantile_levels = quantile_levels.repeat(*y.shape).to(pred_dist.mean.device)
   quantiles = pred_dist.icdf(quantile_levels).detach().cpu().numpy()
   # icdf = Inverse CDF (quantile function)
   ```

   **Result**: `quantiles[b, t, n, q]` where q ∈ {5%, 25%, 50%, 75%, 95%}

3. **Create 2x2 Subplots**:
   ```python
   fig, axs = plt.subplots(2, 2, figsize=(15, 8))
   axs = axs.flatten()  # [ax0, ax1, ax2, ax3]
   ```

4. **Plot for Each Station**:
   ```python
   for i, (b_idx, station) in enumerate(zip(example_indices, stations)):
       ax = axs[i]
       time = np.arange(x.shape[1])  # Lead times [0, 1, ..., 96]

       # Ensemble mean (first feature)
       ax.plot(x[b_idx, :, station, 0], label='ens_mean', color='forestgreen')

       # 5%-95% interval (outer)
       ax.fill_between(time, quantiles[b_idx, :, station, 0], quantiles[b_idx, :, station, 4],
                       alpha=0.15, color="blue", label="5%-95%")

       # 25%-75% interval (inner, darker)
       ax.fill_between(time, quantiles[b_idx, :, station, 1], quantiles[b_idx, :, station, 3],
                       alpha=0.35, color="blue", label="25%-75%")

       # Median (50% quantile)
       ax.plot(time, quantiles[b_idx, :, station, 2], color="black", linestyle="--", label="Median (50%)")

       # Ground truth observation
       ax.plot(y[b_idx, :, station, 0], label='observed', color='mediumvioletred')

       ax.set_title(f'Station {station} at batch element {b_idx}')
       ax.set_xlabel("Lead time")
       ax.set_ylabel("Wind speed")
   ```

**Visual Elements**:
- **Green line**: Ensemble mean (raw NWP forecast)
- **Black dashed line**: Model median prediction (50% quantile)
- **Light blue**: 90% prediction interval (5%-95%)
- **Dark blue**: 50% prediction interval (25%-75%)
- **Pink line**: Actual observed wind speed

5. **Save and Log to MLflow**:
   ```python
   plt.suptitle(f'Predictions at Epoch {epoch}')
   plt.tight_layout()

   plot_filename = f"predictions_epoch_{epoch}.png"
   plt.savefig(plot_filename)
   plt.close(fig)

   mlflow.log_artifact(plot_filename)  # Upload to MLflow
   ```

**Interpretation**:
- **Good calibration**: ~90% of observations fall in 90% interval
- **Sharpness**: Narrower intervals = more confident predictions
- **Bias**: Median should track observations closely

### 8.2 MLflow Logging

**Logged Metrics**:
```python
# Per-iteration (batch-level)
mlflow.log_metric("loss", loss.item(), step=total_iter)

# Per-epoch (training)
mlflow.log_metric("train_loss", avg_loss, step=epoch)
mlflow.log_metric(f'lr_{group}', lr, step=epoch)

# Per-epoch (validation)
mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
mlflow.log_metric("val_loss_original_range", avg_val_loss_or, step=epoch)

# Debugging (from MaskedCRPSLogNormal)
mlflow.log_metric('exp_input_debug', ex_input.max(), step=iteration)
```

**Logged Parameters**:
```python
mlflow.log_param("device_type", device_type)
mlflow.log_param("optimizer", type(optimizer).__name__)
mlflow.log_param("criterion", type(criterion).__name__)
mlflow.log_param("num_params", num_trainable_params)
# + all config parameters from cfg_dict
```

**Logged Artifacts**:
```python
mlflow.log_dict(cfg_dict, 'training_config.json')  # Complete configuration
mlflow.log_artifact(f"predictions_epoch_{epoch}.png")  # Plots every 10 epochs
```

**Accessing MLflow UI**:
```bash
mlflow ui --port 5000
# Navigate to http://localhost:5000
```

---

## 9. Configuration System

### 9.1 Hydra Framework

**Benefits**:
- **Hierarchical configs**: Compose from multiple YAML files
- **Command-line overrides**: `++param.name=value`
- **Config inheritance**: Base config + overrides
- **Type safety**: OmegaConf validates types
- **Dynamic values**: Resolvers for computed values

### 9.2 Default Configuration

**Location**: [configs/default_training_conf.yaml](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/configs/default_training_conf.yaml)

```yaml
dataset:
  # Data paths (from environment variable)
  features_pth: ${oc.env:DATA_BASE_FOLDER}/features.nc
  targets_pth: ${oc.env:DATA_BASE_FOLDER}/targets.nc

  # NWP model and forecast window
  model_name: ch2
  hours_leadtime: 96  # 4-day forecast

  # Temporal splits
  val_split: 0.2  # 80% train, 20% val
  test_start: "2024-05-16"
  train_val_end: "2023-09-30"  # Gap prevents data leakage

  # Predictors (18 features)
  predictors:
    - ch2_ensemble_mean:wind_speed
    - ch2_ensemble_std:wind_speed
    # ... (see full list in section 3.1)

  # Target variable
  target_var: obs:wind_speed

graph_kwargs:
  knn: 5          # k-nearest neighbors
  threshold: 0.6  # Prune edges below this weight

model:
  type: TCN_GNN
  kwargs:
    num_layers: 4
    hidden_channels: 64
    kernel_size: 3
    dropout_p: 0.2
    causal_conv: False
    output_dist: LogNormal

training:
  batch_size: 64
  epochs: 100
  gradient_clip_value: 1.0
  loss: MaskedCRPSLogNormal

  optim:
    algo: Adam
    kwargs:
      lr: 0.0001
      betas: "(0.9, 0.999)"  # String, parsed by eval()

  scheduler:
    algo: CosineAnnealingWarmRestarts
    kwargs:
      T_0: 10      # Initial restart period
      T_mult: 2    # Period multiplier after restart

logging:
  mlflow_tracking_uri: ${oc.env:MLFLOW_TRACKING_URI,mlruns}  # Default: local 'mlruns' folder
  experiment_id: stgnn_wind_postproc
```

### 9.3 Model-Specific Configurations

**Bidirectional RNN** ([configs/bidirectional_rnn_training_conf.yaml](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/configs/bidirectional_rnn_training_conf.yaml)):
```yaml
defaults:
  - default_training_conf

model:
  type: BiDirectionalSTGNN
  kwargs:
    hidden_size: 64
    n_layers: 1
    dropout_p: 0.2
    output_dist: LogNormal

training:
  optim:
    algo: RMSprop
    kwargs:
      lr: 0.0001
      momentum: 0.5
```

**MLP Baseline** ([configs/mlp_training_conf.yaml](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/configs/mlp_training_conf.yaml)):
```yaml
defaults:
  - default_training_conf

model:
  type: MLP
  kwargs:
    hidden_sizes: "[128, 64, 64]"  # String, parsed by eval()
    dropout_p: 0.2
    output_dist: LogNormal
```

**WaveNet** ([configs/wavenet_training_conf.yaml](GraphDL-Postprocess-USI-Project/spatiotemporal_postprocessing/configs/wavenet_training_conf.yaml)):
```yaml
defaults:
  - default_training_conf

model:
  type: WaveNet
  kwargs:
    time_steps: ${add_one:${dataset.hours_leadtime}}  # Resolver: 96 + 1 = 97
    hidden_size: 64
    ff_size: 64
    n_layers: 4
    temporal_kernel_size: 3
    spatial_kernel_size: 3
    output_dist: LogNormal
```

### 9.4 Command-Line Usage

**Basic Training**:
```bash
python train.py
```

**Select Config**:
```bash
python train.py --config-name bidirectional_rnn_training_conf
```

**Override Parameters**:
```bash
# Change optimizer
python train.py ++training.optim.algo=SGD

# Change learning rate
python train.py ++training.optim.kwargs.lr=0.001

# Change batch size
python train.py ++training.batch_size=128

# Multiple overrides
python train.py ++training.epochs=200 ++model.kwargs.hidden_size=128
```

**Environment Variables**:
```bash
export DATA_BASE_FOLDER=/path/to/spatiotemporal_pp_dataset
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
python train.py
```

---

## 10. Complete Execution Flow

### 10.1 End-to-End Workflow

```
1. USER COMMAND
   └─> python train.py --config-name bidirectional_rnn_training_conf ++training.epochs=200

2. HYDRA LOADS CONFIG
   ├─> Loads base: default_training_conf.yaml
   ├─> Overrides with: bidirectional_rnn_training_conf.yaml
   ├─> Applies CLI override: training.epochs=200
   └─> Resolves env vars: DATA_BASE_FOLDER, MLFLOW_TRACKING_URI

3. DATA LOADING
   ├─> Open NetCDF: features.nc, targets.nc (xarray)
   ├─> Select predictors: 18 features
   ├─> Select target: obs:wind_speed
   ├─> Filter lead time: 0-96 hours
   ├─> Temporal split: Train/Val (80/20), Test (separate period with gap)
   ├─> Normalization: Standardize inputs (targets unnormalized)
   └─> Create PyTorch datasets: XarrayDataset

4. GRAPH CONSTRUCTION
   ├─> Extract station coordinates: lat, lon
   ├─> Compute distances: Haversine formula
   ├─> Apply Gaussian kernel: exp(-(d/θ)²)
   ├─> k-NN selection: Keep 5 neighbors per station
   ├─> Threshold pruning: Remove edges < 0.6
   └─> Convert to edge_index: [2, num_edges] (PyG format)

5. MODEL INITIALIZATION
   ├─> Select model: BiDirectionalSTGNN
   ├─> Initialize layers:
   │   ├─> Encoder: Linear(18 → 64)
   │   ├─> Station embeddings: [N, 64]
   │   ├─> Forward RNN: LayeredGraphRNN (mode='forwards')
   │   ├─> Backward RNN: LayeredGraphRNN (mode='backwards')
   │   ├─> Readout: MLP(128 → 64 → 64)
   │   └─> Output: LogNormalLayer(64 → 2)
   └─> Move to device: GPU/CPU

6. TRAINING SETUP
   ├─> Loss: MaskedCRPSLogNormal
   ├─> Optimizer: RMSprop(lr=0.0001, momentum=0.5)
   ├─> LR Scheduler: CosineAnnealingWarmRestarts
   ├─> Gradient clipping: 1.0
   └─> MLflow: Start run, log config

7. TRAINING LOOP (200 epochs)
   For each epoch:
     ├─> TRAINING PHASE
     │   For each batch [64, 97, N, 18]:
     │     ├─> Forward pass:
     │     │   ├─> Encode: x [64,97,N,18] → [64,97,N,64]
     │     │   ├─> Add embeddings: +station_emb [N,64]
     │     │   ├─> Forward RNN: [64,97,N,64] → states_fwd [64,97,N,64]
     │     │   ├─> Backward RNN: [64,97,N,64] → states_bwd [64,97,N,64]
     │     │   ├─> Concat: [states_fwd, states_bwd] → [64,97,N,128]
     │     │   ├─> Skip: +linear(x0) [64,97,N,128]
     │     │   ├─> Readout: [64,97,N,128] → [64,97,N,64]
     │     │   └─> Output: [64,97,N,64] → LogNormal(μ,σ) [64,97,N,1]
     │     ├─> Compute CRPS: MaskedCRPSLogNormal(pred_dist, y)
     │     ├─> Backward: loss.backward()
     │     ├─> Clip gradients: max_norm=1.0
     │     ├─> Optimizer step: Update parameters
     │     └─> Log: mlflow.log_metric("loss", loss, step=iter)
     │
     ├─> VALIDATION PHASE
     │   For each batch:
     │     ├─> Forward pass (no_grad)
     │     ├─> Compute CRPS (normalized + denormalized)
     │     └─> Log: mlflow.log_metric("val_loss", avg_loss, step=epoch)
     │
     ├─> LR SCHEDULE UPDATE
     │   └─> scheduler.step() → Update learning rate
     │
     └─> VISUALIZATION (every 10 epochs)
         ├─> Compute quantiles: 5%, 25%, 50%, 75%, 95%
         ├─> Create 2x2 plot: predictions vs observations
         ├─> Save: predictions_epoch_{epoch}.png
         └─> Log: mlflow.log_artifact(plot)

8. RESULTS
   ├─> MLflow UI: Metrics, parameters, artifacts
   ├─> Best model: Based on val_loss
   └─> Test evaluation: (manual, using test_dataloader)
```

### 10.2 Key Data Transformations

```
SHAPE TRANSFORMATIONS:

NetCDF (xarray):
    features.nc: [forecast_reference_time, lead_time, station, variable]
                 ~[500 days, 97 hours, N stations, 18 features]

After transpose:
    numpy: [time, lead_time, stations, features]
           [500, 97, N, 18]

After normalization (XarrayDataset):
    normalized: (x - μ) / σ
    targets: unchanged (for CRPS-LogNormal)

After DataLoader:
    batch: [B, T, N, F]
           [64, 97, N, 18]

Model forward pass:
    Encoder: [64, 97, N, 18] → [64, 97, N, 64]
    RNN: [64, 97, N, 64] → [64, 97, N, 64] (per direction)
    Concat: [64, 97, N, 128]
    Readout: [64, 97, N, 128] → [64, 97, N, 64]
    Output: [64, 97, N, 64] → LogNormal(μ,σ) where μ,σ: [64, 97, N, 1]

Loss computation:
    pred_dist: LogNormal([64, 97, N, 1])
    y: [64, 97, N, 1]
    CRPS: scalar (mean over all valid observations)
```

### 10.3 Memory and Computational Considerations

**Memory Footprint**:
```
Batch size: 64
Sequence length: 97
Stations: ~N (e.g., 100)
Features: 18
Hidden size: 64

Input: 64 * 97 * 100 * 18 * 4 bytes = ~44 MB
Hidden states: 64 * 97 * 100 * 64 * 4 bytes = ~160 MB
Gradients: ~2x model size

Total: ~500 MB - 2 GB per batch (forward + backward)
```

**Computational Bottlenecks**:
1. **Graph message passing**: O(|E| * H) per timestep
   - |E| = num_edges ≈ N * k (k-NN)
   - H = hidden_size
   - T = timesteps
   - Total: O(N * k * H * T)

2. **Haversine distance**: O(N²) for graph construction
   - One-time cost (only during data loading)
   - Can be precomputed and cached

3. **CRPS computation**: O(B * T * N)
   - Analytical formula (fast)
   - No Monte Carlo sampling needed

**Optimization Strategies**:
- **Gradient clipping**: Prevents exploding gradients
- **Batch normalization**: Stabilizes training
- **Dropout**: Prevents overfitting
- **LR scheduling**: Improves convergence
- **Mixed precision**: (not implemented, potential speedup)

---

## 11. Project Strengths and Design Decisions

### 11.1 Why This Architecture Works

1. **Graph Structure**:
   - Captures spatial correlations (nearby stations have similar weather)
   - Gaussian kernel weights preserve locality
   - k-NN reduces complexity from O(N²) to O(N*k)

2. **Bidirectional RNN**:
   - Forward pass: Learn how errors evolve over lead time
   - Backward pass: Capture dependencies in reverse
   - Richer representations than unidirectional

3. **Probabilistic Outputs**:
   - LogNormal ensures non-negative predictions
   - CRPS rewards both accuracy and calibration
   - Quantifies uncertainty (critical for decision-making)

4. **Skip Connections**:
   - Preserve input information
   - Mitigate vanishing gradients
   - Help model learn residuals (corrections to NWP)

5. **No Target Normalization**:
   - CRPS-LogNormal requires positive targets
   - Standardization can produce negative values
   - Trade-off: training stability vs. loss compatibility

### 11.2 Potential Improvements

**Not Yet Implemented** (for future work):

1. **Attention Mechanisms**:
   - Graph Attention Networks (GATs) for learnable edge weights
   - Temporal attention for adaptive lead time weighting

2. **Ensemble Forecasting**:
   - Use full ensemble members (not just mean/std)
   - Mixture of distributions for multimodal forecasts

3. **Transfer Learning**:
   - Pre-train on global NWP data
   - Fine-tune on Swiss stations

4. **Hyperparameter Optimization**:
   - Automated tuning (Optuna, Ray Tune)
   - Architecture search

5. **Test Set Evaluation**:
   - Currently trains/validates, but test loop not in script
   - Requires manual evaluation

---

## 12. Running the Project

### 12.1 Installation

```bash
cd GraphDL-Postprocess-USI-Project
poetry install
```

**Dependencies**:
- Python 3.11-3.12
- PyTorch 2.5.1
- PyTorch Geometric 2.6.1
- TSL (torch-spatiotemporal) 0.9.5
- Hydra 1.3.2
- MLflow 2.19.0
- xarray, scoringrules, etc.

### 12.2 Data Setup

```bash
export DATA_BASE_FOLDER=/Users/matteovitali/Desktop/MeteoSwiss/spatiotemporal_pp_dataset
```

**Data Files**:
- `features.nc`: 5.6 GB (NWP forecasts + terrain)
- `targets.nc`: 272 MB (observations)

### 12.3 Training

```bash
cd spatiotemporal_postprocessing
poetry shell

# Default config (TCN_GNN)
python train.py

# Bidirectional RNN
python train.py --config-name bidirectional_rnn_training_conf

# MLP baseline
python train.py --config-name mlp_training_conf

# WaveNet
python train.py --config-name wavenet_training_conf

# Custom overrides
python train.py ++training.epochs=200 ++training.batch_size=128
```

### 12.4 Monitoring

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

**Available Metrics**:
- `loss`: Batch-level training loss
- `train_loss`: Epoch-level training loss
- `val_loss`: Validation loss (normalized)
- `val_loss_original_range`: Validation loss (denormalized)
- `lr_0`: Learning rate
- `exp_input_debug`: Debug metric from CRPS-LogNormal

**Available Artifacts**:
- `training_config.json`: Complete configuration
- `predictions_epoch_*.png`: Visualizations every 10 epochs

---

## 13. Summary

This project implements a sophisticated **spatiotemporal postprocessing system** for wind speed forecasts using:

1. **Graph Neural Networks**: Model spatial dependencies between stations
2. **Recurrent/Convolutional Models**: Capture temporal patterns in forecast errors
3. **Probabilistic Forecasting**: Output full distributions with LogNormal guarantees
4. **Proper Scoring**: CRPS loss for probabilistic evaluation
5. **Modular Architecture**: Easily swap models, losses, distributions
6. **Experiment Tracking**: MLflow for reproducibility
7. **Configuration Management**: Hydra for flexible experimentation

**Key Components**:
- **Data Pipeline**: NetCDF → XarrayDataset → Graph → DataLoader
- **Models**: BiDirectionalSTGNN, MLP, WaveNet, TCN_GNN
- **Losses**: CRPS for Normal, LogNormal, Ensemble
- **Output Layers**: LogNormal/Normal distribution parameters
- **Training**: Gradient clipping, LR scheduling, validation, visualization

**Design Philosophy**:
- **Modularity**: Easy to add new models, losses, distributions
- **Reproducibility**: All configs logged to MLflow
- **Flexibility**: Hydra allows easy experimentation
- **Robustness**: NaN masking, gradient clipping, numerical stability

This documentation should provide a complete understanding of every component in the project. For specific implementation details, refer to the source code with the line number references provided throughout.
