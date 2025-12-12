#!/bin/bash

# Tune hyperparameters for each model, then run training with 5 different seeds
# Usage: bash tune_and_run.sh

set -e  # Exit on any error

echo "Starting tune-and-run experiments..."

# Array of config names to tune and run
configs=("model0" "bidirectional_rnn" "stgnn1" "stgnn2" ""default_training_config)

# Array of seeds for final training runs
seeds=(42 123 456 789 2024)

# Tuning parameters
n_trials=15  # Number of tuning trials per model

total_configs=${#configs[@]}
current_config=0

for config in "${configs[@]}"; do
    current_config=$((current_config + 1))
    
    echo ""
    echo "=========================================="
    echo "Model $current_config/$total_configs: $config"
    echo "=========================================="
    
    # Step 1: Tune hyperparameters (skip if config already exists)
    echo ""
    tuned_config="${config}_best"
    
    if [ -f "configs/${tuned_config}.yaml" ]; then
        echo "→ STEP 1: Skipping tuning for $config (configs/${tuned_config}.yaml already exists)"
    else
        echo "→ STEP 1: Tuning hyperparameters for $config..."
        echo "  (Running $n_trials trials)"
        
        python tune.py \
            --n-trials=$n_trials \
            --base-config="configs/${config}.yaml" \
            --output-name="$tuned_config" \
            --seed=42
        
        if [ $? -eq 0 ]; then
            echo "✓ Tuning completed for $config"
            echo "  → Best config saved as: configs/${tuned_config}.yaml"
        else
            echo "✗ Tuning failed for $config"
            exit 1
        fi
    fi
done

echo ""
echo "=========================================="
echo "tuning completed"
echo "=========================================="
echo ""
echo "Tuned configs saved in: configs/"
echo "Tuning reports saved in: tuning/"
echo "MLflow results in: mlruns/"
