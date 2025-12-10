"""
Simple hyperparameter tuning script using Optuna.

This script runs multiple training trials with different hyperparameters
and outputs a config file with the best parameters.

Usage:
    python tune_simple.py --n-trials 20 --base-config configs/default_training_conf.yaml
"""

import argparse
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from omegaconf import OmegaConf
from pathlib import Path

# Import the training function
from spatiotemporal_postprocessing.train import train_with_config, set_seed


def objective(trial, base_config_path, model_name=None):
    """Run one training trial with suggested hyperparameters."""
    
    # Load base config - handle Hydra composition if defaults are specified
    cfg = OmegaConf.load(base_config_path)
    
    # If config has defaults (Hydra composition), load and merge them
    if "defaults" in cfg:
        from pathlib import Path
        config_dir = Path(base_config_path).parent
        
        # Load default config first
        defaults = cfg.get("defaults", [])
        for default in defaults:
            if isinstance(default, str) and default.endswith(".yaml"):
                default_path = config_dir / default
            elif isinstance(default, str):
                default_path = config_dir / f"{default}.yaml"
            else:
                continue
                
            if default_path.exists():
                default_cfg = OmegaConf.load(default_path)
                # Merge: default first, then override with specific config
                cfg = OmegaConf.merge(default_cfg, cfg)
        
        # Remove defaults key from final config
        if "defaults" in cfg:
            del cfg["defaults"]
    
    # Suggest hyperparameters (modify these ranges as needed)
    cfg.training.optim.kwargs.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    cfg.model.kwargs.hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])  # Removed 256 to avoid OOM
    cfg.model.kwargs.num_layers = trial.suggest_categorical("num_layers", [1, 2, 3, 4])  # Removed 6 to avoid OOM
    cfg.model.kwargs.dropout_p = trial.suggest_float("dropout_p", 0.1, 0.5)
    
    # Try batch sizes from largest to smallest, only reducing on OOM
    batch_sizes = [32, 16]  # Largest to smallest
    
    # Reduce epochs for faster tuning
    cfg.training.epochs = 8  # Adjust this
    
    # Set run name if logging config exists
    run_name_base = f"{model_name}_tune_trial_{trial.number}" if model_name else f"tune_trial_{trial.number}"
    if OmegaConf.select(cfg, "logging.run_name") is not None:
        cfg.logging.run_name = run_name_base
    elif "logging" in cfg:
        cfg.logging.run_name = run_name_base
    else:
        # Create logging section if it doesn't exist
        cfg.logging = {"run_name": run_name_base, "mlflow_tracking_uri": "mlruns", "experiment_id": "tuning"}
    
    # Try each batch size until one works
    import torch
    for batch_size in batch_sizes:
        cfg.training.batch_size = batch_size
        
        try:
            val_loss = train_with_config(cfg, trial=trial)
            # If successful, log which batch size worked
            print(f"  → Trial {trial.number} succeeded with batch_size={batch_size}")
            trial.set_user_attr("batch_size", batch_size)
            return val_loss
        except optuna.TrialPruned:
            # Let Optuna handle pruned trials
            raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # If not the last batch size, try smaller
                if batch_size != batch_sizes[-1]:
                    print(f"  ⚠ Trial {trial.number} OOM with batch_size={batch_size}, trying smaller...")
                    continue
                else:
                    print(f"\n⚠ Trial {trial.number} failed with OOM even at smallest batch_size={batch_size}")
                    return float('inf')
            else:
                raise
    
    # Should not reach here
    return float('inf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15, help="Number of trials")
    parser.add_argument("--base-config", type=str, 
                       default="configs/default_training_conf.yaml")
    parser.add_argument("--output-name", type=str, default=None,
                       help="Name for output config file (default: base_config_name + '_best')")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Auto-generate output name if not provided
    if args.output_name is None:
        base_name = Path(args.base_config).stem  # Gets filename without extension
        args.output_name = f"{base_name}_best"
    
    # Extract model name for logging
    model_name = Path(args.base_config).stem
    
    set_seed(args.seed)
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        sampler=TPESampler(seed=args.seed)
    )
    
    print(f"Starting tuning with {args.n_trials} trials...")
    print(f"Base config: {args.base_config}\n")
    
    # Create tuning directory if it doesn't exist
    tuning_dir = Path("tuning")
    tuning_dir.mkdir(exist_ok=True)
    
    # Track all trials for report
    trial_report = []
    trial_report.append("="*80)
    trial_report.append(f"HYPERPARAMETER TUNING REPORT - {args.output_name}")
    trial_report.append("="*80)
    trial_report.append(f"Base config: {args.base_config}")
    trial_report.append(f"Target successful trials: {args.n_trials}")
    trial_report.append(f"Seed: {args.seed}")
    trial_report.append("="*80)
    trial_report.append("")
    
    # Run optimization - continue until we get n_trials valid runs (completed or pruned)
    # Only OOM failures (value=inf) don't count toward the target
    valid_trials = 0  # Successful completions + pruned trials
    max_attempts = args.n_trials * 5  # Allow more attempts for OOM cases
    
    for attempt in range(max_attempts):
        if valid_trials >= args.n_trials:
            break
        
        try:
            study.optimize(
                lambda trial: objective(trial, args.base_config, model_name=model_name),
                n_trials=1,
                show_progress_bar=False
            )
            # Check trial state
            last_trial = study.trials[-1]
            if last_trial.state == optuna.trial.TrialState.COMPLETE:
                if last_trial.value != float('inf'):
                    valid_trials += 1
                    msg = f"✓ Trial {valid_trials}/{args.n_trials} complete | val_loss: {last_trial.value:.6f}"
                    print(msg)
                    trial_report.append(f"Trial {last_trial.number}: SUCCESS")
                    trial_report.append(f"  Status: Complete")
                    trial_report.append(f"  Val Loss: {last_trial.value:.6f}")
                    trial_report.append(f"  Params: {last_trial.params}")
                    # Log actual batch size used (might be smaller due to OOM)
                    if "batch_size" in last_trial.user_attrs:
                        trial_report.append(f"  Actual Batch Size: {last_trial.user_attrs['batch_size']}")
                    trial_report.append("")
                else:
                    msg = f"⚠ Trial {last_trial.number} failed (OOM)"
                    print(msg)
                    trial_report.append(f"Trial {last_trial.number}: FAILED (OOM)")
                    trial_report.append(f"  Status: Complete but returned inf")
                    trial_report.append(f"  Params: {last_trial.params}")
                    trial_report.append("")
            elif last_trial.state == optuna.trial.TrialState.PRUNED:
                valid_trials += 1  # Pruned trials count as valid
                msg = f"✂ Trial {valid_trials}/{args.n_trials} pruned (underperforming)"
                print(msg)
                trial_report.append(f"Trial {last_trial.number}: PRUNED (counts as valid)")
                trial_report.append(f"  Status: Pruned by MedianPruner")
                trial_report.append(f"  Params: {last_trial.params}")
                trial_report.append("")
        except Exception as e:
            msg = f"⚠ Trial failed with exception: {e}"
            print(msg)
            trial_report.append(f"Trial {attempt}: EXCEPTION")
            trial_report.append(f"  Error: {str(e)}")
            trial_report.append("")
            continue
    
    # Check if we hit max attempts without reaching target
    if valid_trials < args.n_trials:
        print(f"\n⚠ WARNING: Only completed {valid_trials}/{args.n_trials} valid trials after {max_attempts} attempts")
        print(f"   (Too many OOM failures - consider reducing search space)")
    
    # Print results
    print("\n" + "="*60)
    print("TUNING COMPLETE!")
    print("="*60)
    
    # Check if we have any successful (non-inf) trials to use for best config
    successful_completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value != float('inf')]
    if len(successful_completed_trials) == 0:
        print("\n❌ ERROR: No successful trials completed!")
        print("All trials either failed with OOM or were pruned.")
        print("\nPlease check:")
        print("  - Reduce search space (smaller hidden_channels, fewer num_layers)")
        print("  - Use smaller batch sizes in tune_simple.py")
        print("  - Check if model architecture fits in memory")
        
        # Save error report
        trial_report.append("="*80)
        trial_report.append("TUNING FAILED")
        trial_report.append("="*80)
        trial_report.append(f"Total trials attempted: {len(study.trials)}")
        trial_report.append(f"Valid trials (completed or pruned): {valid_trials}")
        trial_report.append(f"All completed trials had OOM errors")
        trial_report.append("")
        
        report_path = tuning_dir / f"{args.output_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(trial_report))
        print(f"\n✓ Error report saved to: {report_path}")
        
        return
    
    print(f"\nBest validation loss: {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Add summary to report
    trial_report.append("="*80)
    trial_report.append("FINAL RESULTS")
    trial_report.append("="*80)
    trial_report.append(f"Total trials attempted: {len(study.trials)}")
    trial_report.append(f"Valid trials (completed or pruned): {valid_trials}")
    trial_report.append(f"Successful completions: {len(successful_completed_trials)}")
    trial_report.append(f"Pruned trials: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}")
    trial_report.append(f"OOM failures: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value == float('inf'))}")
    trial_report.append("")
    trial_report.append(f"Best validation loss: {study.best_value:.6f}")
    trial_report.append("Best hyperparameters:")
    for key, value in study.best_params.items():
        trial_report.append(f"  {key}: {value}")
    trial_report.append("")
    trial_report.append("="*80)
    
    # Save report to file
    report_path = tuning_dir / f"{args.output_name}_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(trial_report))
    
    print(f"\n✓ Tuning report saved to: {report_path}")
    
    # Create config with best parameters
    best_cfg = OmegaConf.load(args.base_config)
    
    # Handle Hydra defaults composition
    if "defaults" in best_cfg:
        # Add _self_ to defaults list for Hydra 1.1+
        defaults = best_cfg.get("defaults", [])
        if "_self_" not in defaults:
            defaults.append("_self_")
            best_cfg.defaults = defaults
    
    best_cfg.training.optim.kwargs.lr = study.best_params["lr"]
    best_cfg.model.kwargs.hidden_channels = study.best_params["hidden_channels"]
    best_cfg.model.kwargs.num_layers = study.best_params["num_layers"]
    best_cfg.model.kwargs.dropout_p = study.best_params["dropout_p"]
    
    # Use the actual batch size that worked (stored in user_attrs)
    best_trial = study.best_trial
    if "batch_size" in best_trial.user_attrs:
        best_cfg.training.batch_size = best_trial.user_attrs["batch_size"]
        print(f"  (using batch_size={best_trial.user_attrs['batch_size']} - largest that fit in memory)")
    else:
        best_cfg.training.batch_size = 32  # fallback
    
    # Ensure seed field exists (required for Hydra overrides)
    if "seed" not in best_cfg:
        best_cfg.seed = 42
    
    # Save to configs directory
    output_path = Path("configs") / f"{args.output_name}.yaml"
    OmegaConf.save(best_cfg, output_path)
    
    print(f"✓ Best config saved to: {output_path}")
    print(f"\nTo use it, run:")
    print(f"  python -m spatiotemporal_postprocessing.train --config-name {args.output_name}")


if __name__ == "__main__":
    main()
