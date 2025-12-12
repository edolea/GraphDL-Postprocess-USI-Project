# Spatiotemporal Postprocessing with Graph Neural Networks

A deep learning project for postprocessing wind Numerical Weather Prediction (NWP) forecasts using graph neural networks and spatiotemporal models. Developed for the [Graph Deep Learning](https://search.usi.ch/en/courses/35270698/graph-deep-learning) course at Università della Svizzera italiana (USI).

<img src="./imgs/wind_stations.png" alt="Wind Station Network" width="600">

## Overview

This project implements probabilistic postprocessing of ensemble weather forecasts using various neural network architectures.
The models predict corrected wind speed forecasts with uncertainty quantification using probabilistic outputs (mean and variance), evaluated using the Continuous Ranked Probability Score (CRPS) and Mean Absolute Error (MAE).

## Features

- 🌐 **Graph-based spatial modeling** with PyTorch Geometric
- ⏱️ **Temporal sequence modeling** with various architectures
- 📊 **Probabilistic forecasting** with uncertainty quantification
- 🔧 **Automated hyperparameter tuning** with Optuna
- 📈 **Experiment tracking** with MLflow
- ⚙️ **Configuration management** with Hydra
- 🧪 **Multi-seed experiments** for robust evaluation

## Installation

### Prerequisites
- Python 3.11 or 3.12
- CUDA 12.1+ (for GPU support)
- Poetry (dependency management)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd GraphDL-Postprocess-USI-Project
```

2. **Install dependencies:**
```bash
poetry install
```

3. **Activate the environment:**
```bash
poetry shell
cd spatiotemporal_postprocessing
```

## Quick Start

### Environment Configuration

Set the data folder path:
```bash
export DATA_BASE_FOLDER=<path-to-your-data>
```

Optionally, configure MLflow tracking URI (defaults to `./mlruns`):
```bash
export MLFLOW_TRACKING_URI=<uri>
```

### Training

**Train with default configuration:**
```bash
python train.py
```

**Train with a specific model configuration:**
```bash
python train.py --config-name stgnn2_best
```

**Available configurations:**
- `default_training_conf` - Baseline configuration
- `model0_best` - Optimized baseline model
- `stgnn1_best` - STGNN variant 1 (tuned)
- `stgnn2_best` - STGNN variant 2 (tuned, best performance)
- `bidirectional_rnn_best` - Bidirectional RNN (tuned)
- `mlp_best` - MLP baseline (tuned)
- `wavenet_best` - WaveNet architecture (tuned)

## Hyperparameter Tuning

### tune.sh

The `tune.sh` script automates the hyperparameter optimization process using Optuna. It performs the following steps for each model:

1. **Hyperparameter search**: Runs 15 trials (configurable) using Tree-structured Parzen Estimator (TPE) sampling to find optimal hyperparameters including:
   - Learning rate
   - Hidden channels
   - Number of layers
   - Dropout rate
   - Batch size (with automatic OOM handling)

2. **Configuration export**: Saves the best configuration as `<model>_best.yaml`

3. **Multi-seed training**: Trains the best configuration with 5 different random seeds for robust evaluation

**Usage:**
```bash
bash tune.sh
```

The script processes multiple models sequentially: `model0`, `bidirectional_rnn`, `stgnn1`, `stgnn2`, and `default_training_config`. If a `*_best.yaml` configuration already exists, tuning is skipped.

**Manual tuning:**
```bash
python tune.py --base-config configs/stgnn2.yaml
```

If needed, check argparse of tune.py for more options

## Batch Experiments

### run_all_experiments.sh

The `run_all_experiments.sh` script orchestrates large-scale training experiments across multiple model architectures and random seeds. It:

1. **Systematic evaluation**: Trains each model configuration with 5 different random seeds (42, 123, 456, 789, 2024) to ensure statistical robustness

2. **Progress tracking**: Displays detailed progress indicators showing current run, total runs, and completion status

3. **Error handling**: Exits immediately if any training run fails

4. **Experiment variants**: Supports running additional experiments with graph augmentation techniques

**Usage:**
```bash
bash run_all_experiments.sh
```

This executes training for all optimized configurations:
- `model0_best` (5 seeds)
- `bidirectional_rnn_best` (5 seeds)
- `stgnn1_best` (5 seeds)
- `default_best` (5 seeds)
- `stgnn2_best` (5 seeds)

**Total**: 25+ training runs with comprehensive experiment tracking via MLflow. Optionally also the 5 runs for the graph augementation can be run.

The script automatically logs each run with a unique name (`<config>_seed<N>`) for easy identification and comparison in MLflow.

## Monitoring & Results

### MLflow UI

Launch the MLflow tracking UI to visualize experiments:
```bash
mlflow ui --port 5000 --backend-store-uri ./spatiotemporal_postprocessing/mlruns
```

Then open `http://localhost:5000` in your browser.

### Model Performance

Best performing models (averaged over 5 seeds):

| Model | Overall CRPS | Overall MAE |
|-------|--------------|-------------|
| **STGNN2 (augmented)** | **0.6173 ± 0.0044** | **0.9002 ± 0.0105** |
| STGNN2 | 0.6209 ± 0.0053 | 0.9077 ± 0.0129 |
| STGNN1 | 0.6422 ± 0.0046 | 0.9456 ± 0.0129 |
| Default | 0.6648 ± 0.0031 | 0.9776 ± 0.0080 |
| Model0 | 0.6715 ± 0.0011 | 0.9802 ± 0.0037 |

*Lower CRPS and MAE values indicate better performance.*

## Project Structure

```
spatiotemporal_postprocessing/
├── configs/              # Model and training configurations
│   ├── default_training_conf.yaml
│   ├── stgnn2_best.yaml
│   └── ...
├── datasets/             # Data loading and preprocessing
├── losses/               # Loss functions (deterministic & probabilistic)
├── nn/                   # Neural network models
│   ├── models.py        # Model implementations
│   └── probabilistic_layers.py
├── mlruns/              # MLflow experiment tracking
├── results/             # Training outputs
├── train.py             # Main training script
├── tune.py              # Hyperparameter tuning script
├── tune.sh              # Automated tuning pipeline
└── run_all_experiments.sh  # Batch experiment execution
```

## Configuration

Configuration files use Hydra/OmegaConf format. Key sections:

- `model`: Architecture and hyperparameters
- `training`: Optimizer, batch size, epochs, learning rate
- `data`: Dataset parameters and preprocessing
- `logging`: MLflow experiment tracking settings

Example configuration structure:
```yaml
model:
  name: STGNN2
  kwargs:
    hidden_channels: 64
    num_layers: 3
    dropout_p: 0.2

training:
  epochs: 50
  batch_size: 16
  optim:
    algo: Adam
    kwargs:
      lr: 0.0001
```

## Dependencies

Core libraries:
- PyTorch 2.5.1 (with CUDA 12.1 support)
- PyTorch Geometric 2.6.1
- torch-spatiotemporal 0.9.5
- MLflow 2.19.0
- Hydra-core 1.3.2
- Optuna (for hyperparameter tuning)
- xarray, zarr (for data handling)
- scoringrules (for CRPS evaluation)

See `pyproject.toml` for complete dependency list.

> **Note**: For CUDA 12.8 support with the latest PyTorch libraries and dependencies, copy `pyproject_cu128.txt` to `pyproject.toml`.

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Course: Graph Deep Learning @ USI