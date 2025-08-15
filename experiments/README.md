# Hydra + MLflow Experiment System

This directory contains a comprehensive experiment management system for Code of Kutulu RL experiments, combining Hydra for configuration management and MLflow for experiment tracking.

## Features

- **Hydra Configuration Management**: YAML-based configuration with composition and overrides
- **MLflow Experiment Tracking**: Automatic logging of parameters, metrics, and artifacts
- **Model Registry**: Automatic model versioning and registration
- **Reproducibility**: Complete experiment tracking with git commits and system info
- **Easy Parameter Sweeps**: Multi-run experiments with parameter grids
- **Backward Compatibility**: Works alongside existing notebook-based experiments

## Quick Start

### 1. Run a Single Experiment

```bash
# Run PPO single agent experiment
python experiments/run_experiment.py experiment=ppo_single

# Override specific parameters
python experiments/run_experiment.py experiment=ppo_single agent.lr=0.001 trainer.num_experiments=500
```

### 2. Parameter Sweeps

```bash
# Run multiple experiments with different learning rates
python experiments/run_experiment.py --multirun agent.lr=0.001,0.01,0.1

# Grid search over multiple parameters
python experiments/run_experiment.py --multirun agent.lr=0.001,0.01 agent.entropy_coef=0.1,0.4
```

### 3. View Results

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# List experiments
python experiments/experiment_manager.py list

# Find best run
python experiments/experiment_manager.py best --metric=final_win_rate
```

## Directory Structure

```
experiments/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config with defaults
│   ├── agent/                 # Agent configurations
│   │   ├── ppo.yaml          # PPO agent config
│   │   └── random.yaml       # Random agent config
│   ├── env/                   # Environment configurations
│   │   └── default.yaml      # Default environment config
│   ├── trainer/               # Trainer configurations
│   │   └── default.yaml      # Default trainer config
│   └── experiment/            # Experiment configurations
│       └── ppo_single.yaml   # Single PPO agent experiment
├── outputs/                   # Hydra outputs (auto-generated)
├── mlruns/                    # MLflow tracking store
├── mlflow_artifacts/          # MLflow artifact store
├── run_experiment.py          # Main experiment runner
├── experiment_manager.py      # Experiment management utilities
└── README.md                  # This file
```

## Configuration System

### Composition

The configuration system uses Hydra's composition feature. The main config (`config.yaml`) specifies defaults:

```yaml
defaults:
  - agent: ppo              # Use configs/agent/ppo.yaml
  - env: default           # Use configs/env/default.yaml
  - trainer: default       # Use configs/trainer/default.yaml
  - experiment: ppo_single # Use configs/experiment/ppo_single.yaml
```

### Overrides

You can override any configuration parameter from the command line:

```bash
# Override single parameters
python experiments/run_experiment.py agent.lr=0.001

# Override nested parameters
python experiments/run_experiment.py agent.model_params.fc_dim=32

# Override trainer settings
