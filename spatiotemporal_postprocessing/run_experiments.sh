#!/bin/bash

# Run multiple training configurations with different seeds
# Usage: bash run_experiments.sh

set -e  # Exit on any error

echo "Starting training experiments..."

# Array of config names
# configs=("model0" "bidirectional_rnn" "stgnn1")
configs=("model0")

# Array of seeds
seeds=(42 123 456 789 2024)

total_runs=$((${#configs[@]} * ${#seeds[@]}))
current_run=0

for config in "${configs[@]}"; do
    for seed in "${seeds[@]}"; do
        current_run=$((current_run + 1))
        echo "=========================================="
        echo "Run $current_run/$total_runs: $config (seed=$seed)"
        echo "=========================================="
        python train.py --config-name="$config" \
            logging.run_name="${config}_seed${seed}" \
            seed="$seed"
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed: $config (seed=$seed)"
        else
            echo "✗ Failed: $config (seed=$seed)"
            exit 1
        fi
        echo ""
    done
done

echo "=========================================="
echo "All $total_runs experiments completed successfully!"
echo "=========================================="
