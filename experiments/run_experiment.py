import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.pytorch
import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.trainer import Trainer, get_actions_by_league
from src.envs.agents.agent_factory import get_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_resolver(s):
    return eval(s)

OmegaConf.register_new_resolver("eval", eval_resolver)


def flatten_config(cfg: DictConfig, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested configuration for MLflow logging."""
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, DictConfig):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def setup_mlflow(cfg: DictConfig) -> None:
    """Setup MLflow tracking."""
    # Set tracking URI
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    
    # Set experiment
    experiment_name = cfg.mlflow.experiment_name
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=cfg.mlflow.artifact_location
            )
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        logger.warning(f"Could not setup MLflow experiment: {e}")
        experiment_id = None
    
    return experiment_id


def create_agents_info(cfg: DictConfig) -> list:
    """Create agents_info list from configuration."""
    agents_info = []
    
    # Create training agent
    training_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    agents_info.append(training_agent)

    # Create random agents for remaining slots
    random_agent_template = OmegaConf.to_container(cfg.competitor, resolve=True)
    
    # Fill remaining agent slots
    num_agents = cfg.experiment.get('num_agents', 4)
    for i in range(len(agents_info), num_agents):
        agents_info.append(dict(random_agent_template))

    return agents_info


def log_system_info():
    """Log system information to MLflow."""
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        mlflow.set_tag("git_commit", repo.head.commit.hexsha)
        mlflow.set_tag("git_branch", repo.active_branch.name)
        mlflow.set_tag("git_dirty", repo.is_dirty())
    except Exception as e:
        logger.warning(f"Could not get git info: {e}")
    
    mlflow.set_tag("python_version", sys.version)
    mlflow.set_tag("pytorch_version", torch.__version__)
    mlflow.set_tag("hostname", os.uname().nodename)


def calculate_final_metrics(results) -> dict:
    """Calculate final metrics from training results."""
    reward_list, model_dir_list, metrics_list = results
    
    if not reward_list:
        return {}
    
    import numpy as np
    
    # Calculate final reward statistics
    final_rewards = []
    for rewards in reward_list[-100:]:  # Last 100 episodes
        if len(rewards) > 0:
            final_rewards.append(np.sum(rewards[:, 0]))  # Sum rewards for training agent
    
    final_metrics = {}
    if final_rewards:
        final_metrics['final_mean_reward'] = float(np.mean(final_rewards))
        final_metrics['final_std_reward'] = float(np.std(final_rewards))
        final_metrics['final_max_reward'] = float(np.max(final_rewards))
        final_metrics['final_min_reward'] = float(np.min(final_rewards))
    
    # Add metrics from last metrics calculation
    if metrics_list:
        last_metrics = metrics_list[-1]
        if 'winner_list' in last_metrics:
            final_metrics['final_win_rate'] = float(last_metrics['winner_list'][0])
        if 'rewards' in last_metrics:
            final_metrics['final_avg_reward'] = float(last_metrics['rewards'][0])
    
    final_metrics['total_episodes'] = len(reward_list)
    final_metrics['total_model_saves'] = len(model_dir_list)
    
    return final_metrics


def log_model_artifacts(trainer: Trainer, run_id: str):
    """Log model artifacts to MLflow."""
    try:
        # Log final models
        for i, agent in enumerate(trainer.agents):
            if agent.train and hasattr(agent, 'save_agent'):
                # Create temporary directory for final model
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path = os.path.join(temp_dir, f"agent_{i}_final")
                    os.makedirs(model_path, exist_ok=True)
                    agent.save_agent(model_path)
                    
                    # Log as MLflow artifact
                    mlflow.log_artifacts(model_path, f"models/agent_{i}/final")
                    
                    # Register model in MLflow Model Registry
                    model_uri = f"runs:/{run_id}/models/agent_{i}/final"
                    try:
                        mlflow.register_model(
                            model_uri,
                            f"{trainer.exp_name or 'kutulu_agent'}_{i}",
                            tags={
                                "agent_type": type(agent).__name__,
                                "league_level": str(trainer.league_level),
                                "final_model": "true"
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Could not register model: {e}")
                        
    except Exception as e:
        logger.error(f"Error logging model artifacts: {e}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    """Main experiment runner."""
    logger.info("Starting experiment with configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Setup MLflow
    experiment_id = setup_mlflow(cfg)
    
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        try:
            # Log configuration parameters
            flat_config = flatten_config(cfg)
            mlflow.log_params(flat_config)
            
            # Log system information
            log_system_info()
            
            # Log Hydra output directory
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            output_dir = hydra_cfg.runtime.output_dir
            mlflow.set_tag("hydra_output_dir", output_dir)
            
            # Create agents configuration
            agents_info = create_agents_info(cfg)
            
            # Create trainer
            trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
            trainer_config['agents_info'] = agents_info
            # trainer_config['env_kwargs'] = OmegaConf.to_container(cfg.env, resolve=True)
            
            # Override paths to use Hydra output directory
            trainer_config['log_dir'] = os.path.join(output_dir, "tb")
            trainer_config['output_dir'] = os.path.join(output_dir, "models")
            trainer_config['exp_name'] = run.info.run_name
            
            logger.info(f"Creating trainer with {len(agents_info)} agents")
            trainer = Trainer(**trainer_config)
            
            # Log experiment metadata
            mlflow.set_tag("experiment_name", cfg.experiment.name)
            mlflow.set_tag("experiment_description", cfg.experiment.get('description', ''))
            mlflow.set_tag("num_agents", len(agents_info))
            mlflow.set_tag("training_agent_type", agents_info[0]['type'])
            
            # Run training
            logger.info("Starting training...")
            results = trainer.train()
            
            # Calculate and log final metrics
            final_metrics = calculate_final_metrics(results)
            if final_metrics:
                mlflow.log_metrics(final_metrics)
            
            # Log model artifacts
            log_model_artifacts(trainer, run.info.run_id)
            
            # Log Hydra config as artifact
            hydra_config_path = os.path.join(output_dir, ".hydra", "config.yaml")
            if os.path.exists(hydra_config_path):
                mlflow.log_artifact(hydra_config_path, "hydra_config")
            
            # Save experiment results
            results_path = os.path.join(output_dir, "results.pkl")
            import pickle
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            mlflow.log_artifact(results_path, "results")
            
            mlflow.set_tag("status", "completed")
            logger.info(f"Experiment completed successfully. MLflow run: {run.info.run_id}")
            
        except Exception as e:
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e))
            logger.error(f"Experiment failed: {e}")
            raise
        finally:
            # Clean up trainer resources
            if 'trainer' in locals():
                trainer.close()


if __name__ == "__main__":
    run_experiment()
