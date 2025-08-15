#!/usr/bin/env python3
"""
Experiment management utilities for Code of Kutulu RL experiments.

This script provides utilities for:
- Listing and searching experiments
- Comparing experiment results
- Managing MLflow models
- Analyzing experiment performance

Usage:
    python experiments/experiment_manager.py list
    python experiments/experiment_manager.py find --agent.lr=0.001
    python experiments/experiment_manager.py best --metric=final_win_rate
    python experiments/experiment_manager.py compare run1 run2 run3
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


class ExperimentManager:
    """Manager for MLflow experiments and runs."""
    
    def __init__(self, tracking_uri: str = "file:./mlruns"):
        """Initialize experiment manager."""
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def list_experiments(self) -> pd.DataFrame:
        """List all experiments."""
        experiments = self.client.search_experiments()
        
        data = []
        for exp in experiments:
            runs = self.client.search_runs(exp.experiment_id)
            data.append({
                'experiment_id': exp.experiment_id,
                'experiment_name': exp.name,
                'num_runs': len(runs),
                'creation_time': exp.creation_time,
                'last_update_time': exp.last_update_time
            })
        
        return pd.DataFrame(data)
    
    def list_runs(self, experiment_name: Optional[str] = None, 
                  max_results: int = 100) -> pd.DataFrame:
        """List runs, optionally filtered by experiment."""
        if experiment_name:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                print(f"Experiment '{experiment_name}' not found")
                return pd.DataFrame()
            experiment_ids = [experiment.experiment_id]
        else:
            experiments = self.client.search_experiments()
            experiment_ids = [exp.experiment_id for exp in experiments]
        
        all_runs = []
        for exp_id in experiment_ids:
            runs = self.client.search_runs(exp_id, max_results=max_results)
            all_runs.extend(runs)
        
        data = []
        for run in all_runs:
            run_data = {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
            }
            
            # Add parameters
            for key, value in run.data.params.items():
                run_data[f'param_{key}'] = value
            
            # Add metrics
            for key, value in run.data.metrics.items():
                run_data[f'metric_{key}'] = value
            
            # Add tags
            for key, value in run.data.tags.items():
                run_data[f'tag_{key}'] = value
            
            data.append(run_data)
        
        return pd.DataFrame(data)
    
    def find_runs(self, filter_string: str = "", experiment_name: Optional[str] = None) -> pd.DataFrame:
        """Find runs matching filter criteria."""
        if experiment_name:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                print(f"Experiment '{experiment_name}' not found")
                return pd.DataFrame()
            experiment_ids = [experiment.experiment_id]
        else:
            experiments = self.client.search_experiments()
            experiment_ids = [exp.experiment_id for exp in experiments]
        
        all_runs = []
        for exp_id in experiment_ids:
            runs = self.client.search_runs(exp_id, filter_string=filter_string)
            all_runs.extend(runs)
        
        data = []
        for run in all_runs:
            run_data = {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
            }
            
            # Add key parameters and metrics
            for key, value in run.data.params.items():
                run_data[key] = value
            
            for key, value in run.data.metrics.items():
                run_data[key] = value
            
            data.append(run_data)
        
        return pd.DataFrame(data)
    
    def get_best_run(self, metric: str = "final_win_rate", 
                     experiment_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the best run by specified metric."""
        runs_df = self.list_runs(experiment_name)
        
        if runs_df.empty:
            return None
        
        metric_col = f'metric_{metric}'
        if metric_col not in runs_df.columns:
            print(f"Metric '{metric}' not found in runs")
            return None
        
        # Find run with highest metric value
        best_idx = runs_df[metric_col].idxmax()
        best_run = runs_df.iloc[best_idx]
        
        return best_run.to_dict()
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs."""
        data = []
        
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                run_data = {
                    'run_id': run_id,
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                }
                
                # Add parameters and metrics
                run_data.update(run.data.params)
                run_data.update(run.data.metrics)
                
                data.append(run_data)
                
            except Exception as e:
                print(f"Error getting run {run_id}: {e}")
        
        return pd.DataFrame(data)
    
    def get_run_artifacts(self, run_id: str) -> List[str]:
        """List artifacts for a run."""
        try:
            artifacts = self.client.list_artifacts(run_id)
            return [artifact.path for artifact in artifacts]
        except Exception as e:
            print(f"Error listing artifacts for run {run_id}: {e}")
            return []
    
    def download_artifacts(self, run_id: str, path: str = "", dst_path: str = "."):
        """Download artifacts from a run."""
        try:
            artifact_path = self.client.download_artifacts(run_id, path, dst_path)
            print(f"Downloaded artifacts to: {artifact_path}")
            return artifact_path
        except Exception as e:
            print(f"Error downloading artifacts: {e}")
            return None


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Experiment Manager")
    parser.add_argument("--tracking-uri", default="file:./mlruns", 
                       help="MLflow tracking URI")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List experiments or runs")
    list_parser.add_argument("--type", choices=["experiments", "runs"], 
                           default="experiments", help="What to list")
    list_parser.add_argument("--experiment", help="Experiment name (for runs)")
    list_parser.add_argument("--max-results", type=int, default=100, 
                           help="Maximum number of results")
    
    # Find command
    find_parser = subparsers.add_parser("find", help="Find runs by criteria")
    find_parser.add_argument("--filter", default="", help="MLflow filter string")
    find_parser.add_argument("--experiment", help="Experiment name")
    
    # Best command
    best_parser = subparsers.add_parser("best", help="Find best run by metric")
    best_parser.add_argument("--metric", default="final_win_rate", 
                           help="Metric to optimize")
    best_parser.add_argument("--experiment", help="Experiment name")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare runs")
    compare_parser.add_argument("run_ids", nargs="+", help="Run IDs to compare")
    
    # Artifacts command
    artifacts_parser = subparsers.add_parser("artifacts", help="Manage artifacts")
    artifacts_parser.add_argument("run_id", help="Run ID")
    artifacts_parser.add_argument("--download", help="Download artifacts to path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ExperimentManager(args.tracking_uri)
    
    if args.command == "list":
        if args.type == "experiments":
            df = manager.list_experiments()
            print("Experiments:")
            print(df.to_string(index=False))
        else:
            df = manager.list_runs(args.experiment, args.max_results)
            print("Runs:")
            print(df.to_string(index=False))
    
    elif args.command == "find":
        df = manager.find_runs(args.filter, args.experiment)
        print("Found runs:")
        print(df.to_string(index=False))
    
    elif args.command == "best":
        best_run = manager.get_best_run(args.metric, args.experiment)
        if best_run:
            print(f"Best run by {args.metric}:")
            for key, value in best_run.items():
                print(f"  {key}: {value}")
        else:
            print("No runs found")
    
    elif args.command == "compare":
        df = manager.compare_runs(args.run_ids)
        print("Run comparison:")
        print(df.to_string(index=False))
    
    elif args.command == "artifacts":
        artifacts = manager.get_run_artifacts(args.run_id)
        print(f"Artifacts for run {args.run_id}:")
        for artifact in artifacts:
            print(f"  {artifact}")
        
        if args.download:
            manager.download_artifacts(args.run_id, dst_path=args.download)


if __name__ == "__main__":
    main()
