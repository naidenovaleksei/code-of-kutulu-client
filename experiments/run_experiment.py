import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.pytorch
import numpy as np
import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.trainer import Trainer, get_actions_by_league, DEFAULT_KUTULU_ACTIONS
from src.envs.agents.agent_factory import get_agent
from src.envs.league.agent_description import AgentDescription

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
    
    if cfg.use_master_agent:
        random_agent_template = OmegaConf.to_container(cfg.agent, resolve=True)
    else:
        # Create random agents for remaining slots
        random_agent_template = OmegaConf.to_container(cfg.competitor, resolve=True)
        
        if 'wrapper_params' in random_agent_template:
            if 'actions_mask' in random_agent_template['wrapper_params']:
                actions_mask = random_agent_template['wrapper_params']['actions_mask']
                if actions_mask is not None:
                    if actions_mask == 'default':
                        actions = DEFAULT_KUTULU_ACTIONS
                    elif actions_mask == 'no_yell':
                        actions = DEFAULT_KUTULU_ACTIONS + ["PLAN", "LIGHT"]
                    elif actions_mask == 'wait':
                        actions = ["WAIT"]
                    else:
                        raise ValueError()
                    random_agent_template['wrapper_params']['actions_mask'] = actions
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


def calculate_final_metrics(results, training_agent_id) -> dict:
    """Calculate final metrics from training results."""
    reward_list, model_dir_list, metrics_list = results
    
    final_metrics = {}

    if metrics_list:
        last_metrics = metrics_list[-1]
        final_metrics['final_win_rate'] = last_metrics['winner_list'][training_agent_id]
        final_metrics['final_avg_reward'] = last_metrics['rewards'][training_agent_id]
        final_metrics['acc_explorer'] = last_metrics['check_exp'][training_agent_id][0]
        final_metrics['acc_explorer_plan0'] = last_metrics['check_exp_normal_plan0'][training_agent_id][0]
        final_metrics['acc_explorer_plan1'] = last_metrics['check_exp_normal_plan1'][training_agent_id][0]
        final_metrics['acc_wanderer'] = last_metrics['check_wan'][training_agent_id][0]
        final_metrics['acc_slasher'] = last_metrics['check_slsh'][training_agent_id][0]
        final_metrics['acc_weighted'] = last_metrics['acc_weighted'][training_agent_id]
        final_metrics['acc_weighted_full'] = last_metrics['acc_weighted_full'][training_agent_id]
        final_metrics['loss'] = last_metrics['loss'][training_agent_id]
        final_metrics['policy_loss'] = last_metrics['policy_loss'][training_agent_id]
        final_metrics['value_loss'] = last_metrics['value_loss'][training_agent_id]
        final_metrics['entropy'] = last_metrics['entropy'][training_agent_id]
        final_metrics['acc_weighted_avg100'] = np.mean([
            m['acc_weighted'][training_agent_id] for m in metrics_list[-10:]
        ])
        final_metrics['acc_weighted_full_avg100'] = np.mean([
            m['acc_weighted_full'][training_agent_id] for m in metrics_list[-10:]
        ])

    final_metrics['total_episodes'] = len(reward_list)
    final_metrics['total_model_saves'] = len(model_dir_list)
    
    return final_metrics


def run_training(cfg: DictConfig, output_dir: str = None, exp_name: str = None, 
                 num_experiments_override: int = None) -> tuple:
    """
    Core training function that can be reused by both run_experiment and optimize_hyperparameters.
    
    Args:
        cfg: Configuration with agent, trainer, and experiment settings
        output_dir: Directory for outputs (defaults to Hydra output dir if available)
        exp_name: Experiment name override
        num_experiments_override: Override for number of training experiments
        
    Returns:
        tuple: (results, trainer, final_metrics)
    """
    # Get output directory
    if output_dir is None:
        try:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            output_dir = hydra_cfg.runtime.output_dir
        except Exception:
            output_dir = "./output"
    
    # Create agents configuration
    agents_info = create_agents_info(cfg)
    
    # Create trainer
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_config['agents_info'] = agents_info
    trainer_config['use_master_agent'] = cfg.get('use_master_agent', False)
    
    # Override paths to use output directory
    trainer_config['log_dir'] = os.path.join(output_dir, "tb")
    trainer_config['output_dir'] = os.path.join(output_dir, "models")
    trainer_config['exp_name'] = exp_name or cfg.experiment.name
    
    # Override number of experiments if specified
    if num_experiments_override is not None:
        trainer_config['num_experiments'] = num_experiments_override
    
    # Handle challenge mode if configured
    if cfg.get('use_challenge', False):
        competitors = {}
        for competitor_type, competitor_config, new_experiment, legacy_encoder in [
            ('ppo', '05abe073de06428e896fcd880c9f3eac', True, True),
            ('qdn_conv', '20250622-045641', False, False),
        ]:
            agent_info = get_agent_info(competitor_config, new_experiment=new_experiment)
            agent_info['legacy_encoder'] = legacy_encoder
            competitors[competitor_type] = agent_info
        trainer_config['competitors'] = competitors
    
    logger.info(f"Creating trainer with {len(agents_info)} agents")
    trainer = Trainer(**trainer_config)
    
    # Run training
    logger.info("Starting training...")
    results = trainer.train()
    
    # Calculate final metrics
    final_metrics = calculate_final_metrics(results, cfg.experiment.training_agent_id)
    
    return results, trainer, final_metrics


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


def _get_hydra_agent_info(experiment, train=False):
    config_path = f'../../kutulu_artifacts/mlflow_artifacts/{experiment}/artifacts/hydra_config'
    checkpoint_dir = f'/home/kutulu/projects/kutulu_artifacts/mlflow_artifacts/{experiment}/artifacts/models/agent_0/final'
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path=config_path):
        cfg = hydra.compose(config_name="config")
    info = OmegaConf.to_container(cfg.agent, resolve=True)
    # del info['type']
    info['train'] = train
    info['checkpoint_dir'] = checkpoint_dir
    return info


def get_agent_info(experiment, best_iter=1000, agent_id=0, new_experiment=True, output_dir='../../output'):
    if new_experiment:
        return _get_hydra_agent_info(experiment)
    else:
        return AgentDescription(experiment, best_iter, agent_id, output_dir=output_dir).agent_info


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
            
            # Run training using shared function
            results, trainer, final_metrics = run_training(cfg, output_dir, run.info.run_name)
            
            # Log experiment metadata
            mlflow.set_tag("experiment_name", cfg.experiment.name)
            mlflow.set_tag("experiment_description", cfg.experiment.get('description', ''))
            mlflow.set_tag("num_agents", len(trainer.agents))
            mlflow.set_tag("training_agent_type", trainer.agents_info[0]['type'])
            
            # Log final metrics
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
