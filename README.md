# Code of Kutulu - Reinforcement Learning Client

A comprehensive reinforcement learning framework for training intelligent agents to play the "Code of Kutulu" game - a competitive multi-agent survival game where explorers must navigate mazes while avoiding monsters and maintaining their sanity.

## What is Code of Kutulu?

Code of Kutulu is a competitive programming game where 2-4 players control explorers in a maze filled with wandering monsters (Wanderers and Slashers). Players must:
- Survive as long as possible while maintaining sanity
- Avoid or manage encounters with monsters
- Use special abilities (PLAN, LIGHT, YELL) strategically
- Compete against other explorers for the highest score

## Features

### Reinforcement Learning Algorithms
- **Deep Q-Networks (DQN)**: Multiple variants including standard, convolutional, and extended state representations
- **Policy Gradient Methods**: PPO (Proximal Policy Optimization), A2C (Advantage Actor-Critic), REINFORCE
- **Q-Learning**: Tabular Q-learning with various exploration strategies
- **Rule-based Baselines**: Dummy agents and epsilon-greedy strategies for benchmarking

### Training Capabilities
- **Multi-agent self-play**: Train agents against each other
- **League-based progression**: Gradually increase difficulty through league levels
- **Parallel environments**: Speed up training with multiple simultaneous games
- **Reward shaping**: Potential-based and custom reward functions
- **Experience replay**: Prioritized and standard replay buffers

### Experiment Management
- **Hydra configuration**: YAML-based configuration with composition and overrides
- **MLflow tracking**: Automatic logging of parameters, metrics, and artifacts
- **Hyperparameter optimization**: Optuna integration for automated tuning
- **Model versioning**: Automatic model registry and checkpointing

## Project Structure

```
code-of-kutulu-client/
├── src/
│   ├── envs/                      # Game environment and training
│   │   ├── agents/                # RL agent implementations
│   │   │   ├── ppo_agent.py      # PPO implementation
│   │   │   ├── dqn_agent*.py     # DQN variants
│   │   │   ├── a2c_agent.py      # A2C implementation
│   │   │   └── ...
│   │   ├── models/                # Neural network architectures
│   │   ├── league/                # League and ELO rating system
│   │   ├── kutulu_world.py        # Main game environment
│   │   ├── kutulu_observer.py     # State observation handlers
│   │   ├── reward_shaper.py       # Reward shaping utilities
│   │   └── trainer.py             # Training orchestration
│   └── game/
│       └── template.py            # Competition submission template
├── experiments/                    # Experiment management
│   ├── configs/                   # Hydra configuration files
│   │   ├── agent/                 # Agent configs (PPO, DQN, etc.)
│   │   ├── trainer/               # Training configs
│   │   └── experiment/            # Full experiment configs
│   ├── run_experiment.py          # Main experiment runner
│   ├── optimize_hyperparameters.py # Hyperparameter optimization
│   └── experiment_manager.py      # Experiment utilities
├── notebooks/                      # Jupyter notebooks for exploration
├── tests/                          # Unit tests
└── requirements.txt                # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd code-of-kutulu-client
```

2. Create and activate a virtual environment:
```bash
python -m venv kutulu-env
source kutulu-env/bin/activate  # On Windows: kutulu-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training a PPO Agent

```bash
# Train a PPO agent with default settings
python experiments/run_experiment.py experiment=ppo_single

# Train with custom parameters
python experiments/run_experiment.py \
    experiment=ppo_single \
    agent.lr=0.001 \
    trainer.num_experiments=1000 \
    trainer.num_envs=4
```

### Training a DQN Agent

```bash
# Train a convolutional DQN agent
python experiments/run_experiment.py \
    agent=qdn_conv \
    trainer.num_experiments=500
```

### Running Parameter Sweeps

```bash
# Test multiple learning rates
python experiments/run_experiment.py --multirun \
    agent.lr=0.001,0.01,0.1

# Grid search over multiple parameters
python experiments/run_experiment.py --multirun \
    agent.lr=0.001,0.01 \
    agent.entropy_coef=0.1,0.4
```

## Training Agents

### Configuration System

The project uses Hydra for configuration management. Configurations are organized hierarchically:

- **Agent configs** (`experiments/configs/agent/`): Define agent architecture and hyperparameters
- **Trainer configs** (`experiments/configs/trainer/`): Define training parameters
- **Experiment configs** (`experiments/configs/experiment/`): Combine agent, trainer, and competitor settings

### Available Agent Types

1. **PPO (Proximal Policy Optimization)**
   - State types: `conv`, `conv_ext`, `conv_ext_deep`
   - Best for: General-purpose training, stable learning

2. **DQN (Deep Q-Network)**
   - Variants: Standard, Convolutional, By-kind
   - State types: `ext`, `conv`, `conv_ext`
   - Best for: Value-based learning, discrete actions

3. **A2C (Advantage Actor-Critic)**
   - State types: `conv`
   - Best for: Fast training, on-policy learning

### Training Against Competitors

```bash
# Train against a PPO competitor
python experiments/run_experiment.py \
    experiment=ppo_single \
    competitor=ppo

# Train against a DQN competitor
python experiments/run_experiment.py \
    experiment=ppo_single \
    competitor=qdn_conv
```

## Experiment Tracking

### View Experiments with MLflow

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns --port 5002

# Open browser to http://localhost:5002
```

### View Training Logs with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir experiments/outputs --port 6006

# Open browser to http://localhost:6006
```

### Experiment Management

```bash
# List all experiments
python experiments/experiment_manager.py list

# Find best run by metric
python experiments/experiment_manager.py best --metric=final_win_rate

# Compare multiple runs
python experiments/experiment_manager.py compare --run-ids <id1> <id2>
```

## Hyperparameter Optimization

Use Optuna for automated hyperparameter tuning:

```bash
python experiments/optimize_hyperparameters.py \
    optimization.n_trials=100 \
    optimization.objective_metric=acc_weighted_full_avg100 \
    optimization.direction=maximize \
    optimization.study_name=ppo_study_1 \
    trainer.num_experiments=1000
```

## Model Submission

After training, models can be converted to standalone submission scripts:

1. **Locate your trained model** in the experiment outputs
2. **Use the template system** in `src/game/template.py`
3. **Convert model weights** to embedded format
4. **Submit** the generated Python file to the competition

The template supports multiple agent architectures and handles all game logic internally.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ppo_agent.py

# Run with coverage
pytest --cov=src tests/
```

## Key Concepts

### State Representations

- **Closest**: Simple state with distances to nearest entities
- **Extended (ext)**: Detailed entity features and relationships
- **Convolutional (conv)**: Grid-based spatial representation
- **Conv Extended (conv_ext)**: Enhanced convolutional with aggregated features

### Reward Shaping

The framework supports multiple reward shaping strategies:
- Survival rewards
- Sanity maintenance bonuses
- Strategic ability usage rewards
- Potential-based shaping for smoother learning

## Acknowledgments

This project is built for the Code of Kutulu competitive programming challenge, inspired by the Lovecraftian horror theme and strategic gameplay mechanics.
