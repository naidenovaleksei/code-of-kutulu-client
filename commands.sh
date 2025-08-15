# mlflow
mlflow ui --backend-store-uri file:./mlruns --port 5001

# tensorboard
tensorboard --logdir $HOME/projects/kutulu_artifacts/experiments/outputs

# experiment
python experiments/run_experiment.py agent.lr=0.001 trainer.num_experiments=100 trainer.silent=false trainer.verbose=false trainer.metrics_int=10
