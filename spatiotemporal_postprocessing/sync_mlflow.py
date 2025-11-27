#!/usr/bin/env python3
"""
Sync local MLflow runs to remote tracking server.
Usage: python sync_mlflow.py [local_mlruns_path] [remote_uri]
"""
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import time

def sync_runs(local_path="./mlruns_local", remote_uri=None, skip_artifacts=False):
    """Sync all runs from local MLflow directory to remote server"""
    
    if remote_uri is None:
        remote_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not remote_uri or remote_uri == "mlruns":
            print("Error: Please provide remote URI or set MLFLOW_TRACKING_URI")
            return
    
    print(f"Syncing from: {local_path}")
    print(f"Syncing to: {remote_uri}")
    
    # Get local client
    local_client = MlflowClient(tracking_uri=local_path)
    
    # Get remote client
    remote_client = MlflowClient(tracking_uri=remote_uri)
    
    # Get all experiments from local
    local_experiments = local_client.search_experiments()
    
    for exp in local_experiments:
        if exp.name == "Default":
            continue
            
        print(f"\nProcessing experiment: {exp.name}")
        
        # Get or create experiment on remote
        try:
            remote_exp = remote_client.get_experiment_by_name(exp.name)
            if remote_exp is None:
                remote_exp_id = remote_client.create_experiment(exp.name)
                print(f"  Created remote experiment: {exp.name}")
            else:
                remote_exp_id = remote_exp.experiment_id
                print(f"  Found existing remote experiment: {exp.name}")
        except Exception as e:
            print(f"  Error with experiment: {e}")
            continue
        
        # Get all runs for this experiment
        runs = local_client.search_runs(experiment_ids=[exp.experiment_id])
        
        # Get existing remote runs to check duplicates (faster than individual get_run calls)
        print(f"  Fetching existing remote runs...")
        try:
            remote_runs = remote_client.search_runs(experiment_ids=[remote_exp_id])
            remote_run_ids = {r.info.run_id for r in remote_runs}
            print(f"  Found {len(remote_run_ids)} existing remote runs")
        except Exception as e:
            print(f"  Warning: Could not fetch remote runs, will try to sync all: {e}")
            remote_run_ids = set()
        
        for run in runs:
            run_id = run.info.run_id
            print(f"  Syncing run: {run_id} ({run.data.tags.get('mlflow.runName', 'unnamed')})")
            
            try:
                # Check if run already exists on remote (using pre-fetched list)
                if run_id in remote_run_ids:
                    print(f"    Run already exists on remote, skipping")
                    continue
                
                print(f"    Run does not exist on remote, proceeding to sync...")
                
                # Add small delay to avoid rate limiting
                time.sleep(1)
                
                # Create new run on remote
                mlflow.set_tracking_uri(remote_uri)
                with mlflow.start_run(
                    experiment_id=remote_exp_id,
                    run_name=run.data.tags.get('mlflow.runName'),
                    nested=False
                ) as remote_run:
                    # Log params in batch to reduce API calls
                    if run.data.params:
                        try:
                            mlflow.log_params(run.data.params)
                        except:
                            # Fallback to individual logging if batch fails
                            for key, value in run.data.params.items():
                                try:
                                    mlflow.log_param(key, value)
                                except Exception as e:
                                    print(f"\n      Warning: Could not log param {key}: {e}")
                    
                    # Log metrics with batching
                    for key in run.data.metrics.keys():
                        try:
                            # Get metric history
                            metric_history = local_client.get_metric_history(run_id, key)
                            for metric in metric_history:
                                mlflow.log_metric(key, metric.value, step=metric.step)
                        except Exception as e:
                            print(f"\n      Warning: Could not log metric {key}: {e}")
                    
                    # Log tags
                    for key, value in run.data.tags.items():
                        if not key.startswith('mlflow.'):
                            mlflow.set_tag(key, value)
                    
                    print(f"    ✓ Logged params, metrics, and tags")

                    # Copy artifacts (unless skipped)
                    if not skip_artifacts:
                        local_artifact_uri = run.info.artifact_uri
                        if local_artifact_uri:
                            # Get local artifact path
                            local_artifact_path = local_artifact_uri.replace('file://', '')
                            if os.path.exists(local_artifact_path):
                                # Count files first
                                all_files = []
                                for root, dirs, files in os.walk(local_artifact_path):
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        file_size = os.path.getsize(file_path)
                                        all_files.append((file_path, file, root, file_size))
                                
                                if all_files:
                                    total_size = sum(f[3] for f in all_files) / (1024*1024)  # MB
                                    print(f"    Uploading {len(all_files)} artifacts ({total_size:.1f} MB)")
                                    
                                    # Log all artifacts with progress
                                    for idx, (file_path, file, root, file_size) in enumerate(all_files, 1):
                                        # Get relative path for artifact directory structure
                                        rel_path = os.path.relpath(root, local_artifact_path)
                                        artifact_path = None if rel_path == '.' else rel_path
                                        try:
                                            print(f"      [{idx}/{len(all_files)}] {file} ({file_size/1024:.1f} KB)", end='\r')
                                            mlflow.log_artifact(file_path, artifact_path)
                                        except Exception as e:
                                            print(f"\n      Warning: Could not upload {file}: {e}")
                                    print()  # New line after progress
                    else:
                        print(f"    Skipping artifacts (--skip-artifacts flag set)")
                    
                print(f"    ✓ Synced successfully")
                
            except Exception as e:
                print(f"    Error syncing run: {e}")
    
    print("\n" + "="*50)
    print("Sync completed!")
    print("="*50)

if __name__ == "__main__":
    skip_artifacts = "--skip-artifacts" in sys.argv
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    
    local = args[0] if len(args) > 0 else "./mlruns"
    remote = args[1] if len(args) > 1 else None
    
    sync_runs(local, remote, skip_artifacts)
