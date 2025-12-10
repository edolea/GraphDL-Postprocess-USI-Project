#!/bin/bash

# Tune hyperparameters for each model, then run training with 5 different seeds
# Usage: bash tune_and_run.sh

set -e  # Exit on any error

echo "Starting tune-and-run experiments..."

# Array of config names to tune and run
configs=("model0" "bidirectional_rnn" "stgnn1" "stgnn2" "wavenet" "mlp")
configs=("bidirectional_rnn" "stgnn1" "stgnn2" "wavenet" "mlp")
configs=("stgnn1" "stgnn2" "wavenet" "mlp")

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
        
        python tune_simple.py \
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
    
    # Step 2: Run training with tuned config on multiple seeds
    echo ""
    echo "→ STEP 2: Running training with tuned config on ${#seeds[@]} seeds..."
    
    for seed in "${seeds[@]}"; do
        echo ""
        echo "  Training ${tuned_config} (seed=$seed)..."
        
        python train.py \
            --config-name="$tuned_config" \
            logging.run_name="${tuned_config}_seed${seed}" \
            seed="$seed"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Completed: ${tuned_config} (seed=$seed)"
        else
            echo "  ✗ Failed: ${tuned_config} (seed=$seed)"
            exit 1
        fi
    done
    
    echo ""
    echo "✓ All runs completed for $config"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "SUCCESS! All experiments completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Tuned $total_configs models"
echo "  - Ran ${#seeds[@]} training runs per model"
echo "  - Total training runs: $((total_configs * ${#seeds[@]}))"
echo ""
echo "Tuned configs saved in: configs/"
echo "Tuning reports saved in: tuning/"
echo "MLflow results in: mlruns/"
