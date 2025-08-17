# venv
source ~/kutulu-env/bin/activate

# mlflow
mlflow ui --backend-store-uri file:$HOME/projects/kutulu_artifacts/mlruns --port 5001

# tensorboard
tensorboard --logdir $HOME/projects/kutulu_artifacts/experiments/outputs --port 6007

# experiment
python experiments/run_experiment.py agent.lr=0.001 trainer.num_experiments=100 trainer.silent=false trainer.verbose=false trainer.metrics_int=10 trainer.num_envs=4 competitor=qdn_conv

python experiments/run_experiment.py \
    agent.lr=0.001 \
    trainer.num_experiments=1 \
    trainer.silent=true \
    trainer.verbose=true \
    trainer.metrics_int=10 \
    trainer.num_envs=4 \
    competitor=qdn_conv \
    competitor.epsilon_params.start=0.75 \
    trainer.env_kwargs.reward_params.sanity_coef=0.025 \
    trainer.env_kwargs.reward_params.reward_for_win=5 \
    trainer.env_kwargs.reward_params.reward_for_lose=-.5 \
    trainer.env_kwargs.reward_params.step_bonus=0.25 \
    trainer.env_kwargs.reward_params.madness_per_turn_coef=0.5

python experiments/run_experiment.py \
    trainer.num_experiments=1 \
    trainer.silent=true \
    trainer.verbose=true

# optimize_hyperparameters
python experiments/optimize_hyperparameters.py optimization.n_trials=100 optimization.objective_metric=acc_weighted \
    competitor=qdn_conv 
