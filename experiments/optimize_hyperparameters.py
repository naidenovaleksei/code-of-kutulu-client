import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import tempfile
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.pytorch
import optuna
from optuna.integration.mlflow import MLflowCallback
import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.trainer import Trainer, DEFAULT_KUTULU_ACTIONS
from experiments.run_experiment import (
    setup_mlflow, create_agents_info, log_system_info, 
    calculate_final_metrics, flatten_config
)

# Register eval resolver if not already registered
def eval_resolver(s):
    return eval(s)

try:
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval_resolver)
except Exception:
    # Resolver already registered, ignore
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress optuna logs to reduce noise
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_hyperparameters(trial: optuna.Trial, base_config: DictConfig) -> DictConfig:
    """Suggest hyperparameters using Optuna trial."""
    config = deepcopy(base_config)
    
    # env params
    # # config.trainer.env_kwargs.reward_params.sanity_coef = trial.suggest_float("sanity_coef", 0.01, 0.1)
    # config.trainer.env_kwargs.reward_params.reward_for_win = trial.suggest_float("reward_for_win", 0, 5)
    # config.trainer.env_kwargs.reward_params.reward_for_lose = trial.suggest_float("reward_for_lose", -5, 0)
    # config.trainer.env_kwargs.reward_params.step_bonus = trial.suggest_float("step_bonus", 0, 0.5)
    # config.trainer.env_kwargs.reward_params.madness_per_turn_coef = trial.suggest_float("madness_per_turn_coef", 0, 1)
    
    # # rl params
    # config.agent.gamma = trial.suggest_float("gamma", 0.0, 0.99)
    # config.agent.gae_lambda = trial.suggest_float("gae_lambda", 0.0, 0.99)

    # competitor params
    # config.competitor.epsilon_params.start = trial.suggest_float("epsilon_start", 0, 1)
    config.competitor.wrapper_params.epsilon = trial.suggest_float("epsilon", 0.0, 1.0)
    config.competitor.wrapper_params.explicit_change = trial.suggest_categorical("explicit_change", [True, False])
    config.competitor.wrapper_params.mode = trial.suggest_categorical("mode", ["sample", "random", "top2"])
    config.competitor.wrapper_params.actions_mask = trial.suggest_categorical(
        "actions_mask",
        [
            None,
            "default",
            "no_yell",
            "wait",
        ]
    )

    # training params
    # config.agent.entropy_coef = trial.suggest_float("entropy_coef", 0, 1)
    # config.agent.value_loss_coef = trial.suggest_float("value_loss_coef", 0, 0.5)
    # config.agent.clip_ratio = trial.suggest_float("clip_ratio", 0.01, 1)
    # config.agent.target_kl = trial.suggest_float("target_kl", 0.01, 1)
    # config.agent.max_grad_norm = trial.suggest_float("max_grad_norm", 0.01, 1)
    
    # # loss params
    # config.agent.lr = trial.suggest_float("lr", 0.0001, 0.1, log=True)
    # config.agent.mini_batch_size = trial.suggest_categorical("mini_batch_size", [8, 16, 32, 64])
    # config.agent.ppo_epochs = trial.suggest_int("ppo_epochs", 1, 10)

    # # reward params
    # # config.agent.reward_params.gamma = trial.suggest_float("gamma", 0.9, 0.99999, log=True)
    # config.agent.reward_params.lights_left_coef = trial.suggest_float("lights_left_coef", 0, 0.2)
    # config.agent.reward_params.light_potential_coef = trial.suggest_float("light_potential_coef", 0, 0.5)
    # config.agent.reward_params.plans_left_coef = trial.suggest_float("plans_left_coef", 0, 0.2)
    # config.agent.reward_params.plan_potential_coef = trial.suggest_float("plan_potential_coef", 0, 0.2)
    # config.agent.reward_params.yell_potential_coef = trial.suggest_float("yell_potential_coef", 0, 0.3)
    # # config.agent.reward_params.w_nearby_potential_coef = trial.suggest_float("w_nearby_potential_coef", 1e-4, 1.0, log=True)
    # config.agent.reward_params.e_nearby_potential_coef = trial.suggest_float("e_nearby_potential_coef", 0, 0.4)
    # config.agent.reward_params.s_nearby_potential_coef = trial.suggest_float("s_nearby_potential_coef", 0, 0.9)
    # config.agent.reward_params.sanity_loss_potential_coef = trial.suggest_float("sanity_loss_potential_coef", 0, 0.35)
    # config.agent.reward_params.no_move_reward_coef = trial.suggest_float("no_move_reward_coef", 0, 0.02)

    return config


def objective(trial: optuna.Trial, base_config: DictConfig, optimization_config: DictConfig) -> float:
    """Objective function for Optuna optimization."""
    
    # Get suggested hyperparameters
    config = suggest_hyperparameters(trial, base_config)
    
    # Create unique experiment name for this trial
    trial_name = f"{base_config.experiment.name}_trial"
    config.experiment.name = trial_name
    config.mlflow.experiment_name = trial_name
    
    # Setup MLflow for this trial
    experiment_id = setup_mlflow(config)
    
    # Start MLflow run for this trial
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"trial_{trial.number}") as run:
        try:
            # Log trial parameters
            flat_config = flatten_config(config)
            mlflow.log_params(flat_config)
            
            # Log system information
            log_system_info()
            
            # Log Optuna trial info
            mlflow.set_tag("optuna_trial_number", trial.number)
            mlflow.set_tag("optuna_study_name", trial.study.study_name)
                
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            output_dir = hydra_cfg.runtime.output_dir
            # Create agents configuration
            agents_info = create_agents_info(config)
            
            # Create trainer
            trainer_config = OmegaConf.to_container(config.trainer, resolve=True)
            trainer_config['agents_info'] = agents_info
            
            # Set paths to temporary directory
            trainer_config['log_dir'] = os.path.join(output_dir, "tb")
            trainer_config['output_dir'] = os.path.join(output_dir, "models")
            trainer_config['exp_name'] = f"trial_{trial.number}"
            
            # Reduce number of experiments for optimization (faster trials)
            if optimization_config.get('fast_trials', True):
                trainer_config['num_experiments'] = min(
                    trainer_config.get('num_experiments', 2000),
                    optimization_config.get('max_experiments_per_trial', 500)
                )
            
            logger.info(f"Trial {trial.number}: Starting training with {len(agents_info)} agents")
            trainer = Trainer(**trainer_config)
            
            # Log experiment metadata
            mlflow.set_tag("num_agents", len(agents_info))
            mlflow.set_tag("training_agent_type", agents_info[0]['type'])
            
            # Run training
            results = trainer.train()
            
            # Calculate final metrics
            final_metrics = calculate_final_metrics(results, base_config.experiment.training_agent_id)
            if final_metrics:
                mlflow.log_metrics(final_metrics)
            
            # Clean up trainer resources
            trainer.close()
            
            # Determine objective value based on optimization target
            objective_metric = optimization_config.get('objective_metric', 'final_mean_reward')
            objective_value = final_metrics.get(objective_metric, 0.0)
            
            # Log objective value
            mlflow.log_metric("objective_value", objective_value)
            mlflow.set_tag("status", "completed")
            
            logger.info(f"Trial {trial.number} completed. Objective: {objective_value}")
            
            # Report intermediate value for pruning
            trial.report(objective_value, step=trainer_config['num_experiments'])
            
            return objective_value
                
        except optuna.TrialPruned:
            mlflow.set_tag("status", "pruned")
            logger.info(f"Trial {trial.number} was pruned")
            raise
        except Exception as e:
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e))
            logger.error(f"Trial {trial.number} failed: {e}")
            raise


def create_study(optimization_config: DictConfig) -> optuna.Study:
    """Create or load Optuna study."""
    study_name = optimization_config.get('study_name', 'hyperparameter_optimization')
    storage_url = optimization_config.get('storage_url', None)
    
    # Determine optimization direction
    direction = optimization_config.get('direction', 'maximize')
    
    # Create sampler
    sampler_config = optimization_config.get('sampler', {})
    sampler_type = sampler_config.get('type', 'tpe')
    
    if sampler_type == 'tpe':
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=sampler_config.get('n_startup_trials', 10),
            seed=sampler_config.get('seed', 42)
        )
    elif sampler_type == 'random':
        sampler = optuna.samplers.RandomSampler(seed=sampler_config.get('seed', 42))
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction=direction,
        sampler=sampler
    )
    
    return study


@hydra.main(version_base=None, config_path="configs", config_name="optimization")
def optimize_hyperparameters(cfg: DictConfig) -> None:
    """Main hyperparameter optimization function."""
    logger.info("Starting hyperparameter optimization")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Extract base experiment config and optimization config
    base_config = cfg.base_experiment
    optimization_config = cfg.optimization
    
    # Create Optuna study
    study = create_study(optimization_config)
    
    # Setup MLflow callback for Optuna
    mlflow_callback = None
    if optimization_config.get('log_to_mlflow', True):
        # Setup parent MLflow experiment for the study
        study_mlflow_config = deepcopy(base_config.mlflow)
        study_mlflow_config.experiment_name = f"{base_config.experiment.name}_optimization_study"
        
        # Create temporary config for MLflow setup
        temp_config = DictConfig({'mlflow': study_mlflow_config})
        study_experiment_id = setup_mlflow(temp_config)
        
        mlflow_callback = MLflowCallback(
            tracking_uri=base_config.mlflow.tracking_uri,
            metric_name="objective_value"
        )
    
    # Run optimization
    n_trials = optimization_config.get('n_trials', 100)
    timeout = optimization_config.get('timeout', None)
    
    logger.info(f"Starting optimization with {n_trials} trials")
    
    try:
        study.optimize(
            lambda trial: objective(trial, base_config, optimization_config),
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[mlflow_callback] if mlflow_callback else None
        )
        
        # Log best results
        logger.info("Optimization completed!")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value}")
        logger.info(f"Best parameters: {study.best_params}")
        
        # Save study results
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        output_dir = hydra_cfg.runtime.output_dir
        
        # Save best parameters as YAML config
        best_config = suggest_hyperparameters(study.best_trial, base_config)
        best_config_path = os.path.join(output_dir, "best_config.yaml")
        with open(best_config_path, 'w') as f:
            OmegaConf.save(best_config, f)
        
        # Save study object
        study_path = os.path.join(output_dir, "study.pkl")
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        
        # Save optimization results summary
        results_summary = {
            'best_trial_number': study.best_trial.number,
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'study_name': study.study_name
        }
        
        results_path = os.path.join(output_dir, "optimization_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results_summary, f)
        
        logger.info(f"Results saved to {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        logger.info(f"Completed {len(study.trials)} trials")
        if study.trials:
            logger.info(f"Best trial so far: {study.best_trial.number}")
            logger.info(f"Best value so far: {study.best_value}")


if __name__ == "__main__":
    optimize_hyperparameters()
