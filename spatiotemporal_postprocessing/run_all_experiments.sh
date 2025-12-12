#!/bin/bash

# Run multiple training configurations with different seeds

set -e

echo "Starting training experiments..."

# Array of config names
configs=("model0_best" "bidirectional_rnn_best" "stgnn1_best" "default_best" "stgnn2_best")

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

    echo "*************************************************************"
    echo "Completed all runs for $config"
    echo "*************************************************************"
done

configs=("stgnn2_best" "model0_best")
seeds=(42 123 456 789 2024)

total_runs=$((${#configs[@]} * ${#seeds[@]}))
current_run=0

echo ""
echo "Starting graph augmentation experiments"
echo ""

## v1
for config in "${configs[@]}"; do
    for seed in "${seeds[@]}"; do
        current_run=$((current_run + 1))
        echo "=========================================="
        echo "Run $current_run/$total_runs: $config (seed=$seed) - Alpine"
        echo "=========================================="
        python train.py --config-name="$config" \
            ++graph_kwargs.theta_strategy=nn_median \
            ++graph_kwargs.theta_scale=0.7 \
            ++graph_kwargs.orography_var=terrain:elevation_50m \
            ++graph_kwargs.orography_scale=400 \
            ++graph_kwargs.orography_alpha=1.5 \
            logging.run_name="${config}_seed${seed}_alpine_v1" \
            seed="$seed"
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed: $config (seed=$seed) - Alpine"
        else
            echo "✗ Failed: $config (seed=$seed) - Alpine"
            exit 1
        fi
        echo ""
    done

    echo "*************************************************************"
    echo "Completed all runs for $config - Alpine"
    echo "*************************************************************"
done

## v2
for config in "${configs[@]}"; do
    for seed in "${seeds[@]}"; do
        current_run=$((current_run + 1))
        echo "=========================================="
        echo "Run $current_run/$total_runs: $config (seed=$seed) - Alpine"
        echo "=========================================="
        python train2.py --config-name="$config" \
            graph_kwargs.knn=8 \
            ++graph_kwargs.theta_strategy=nn_median \
            ++graph_kwargs.theta_scale=0.7 \
            ++graph_kwargs.max_distance_km=120 \
            ++graph_kwargs.mutual=true \
            ++graph_kwargs.threshold=0.35 \
            ++graph_kwargs.orography_var=terrain:distance_to_alpine_ridge \
            ++graph_kwargs.orography_scale=15000 \
            ++graph_kwargs.orography_alpha=1.0 \
            logging.run_name="${config}_seed${seed}_alpine_v2" \
            seed="$seed"


        
        if [ $? -eq 0 ]; then
            echo "✓ Completed: $config (seed=$seed) - Alpine"
        else
            echo "✗ Failed: $config (seed=$seed) - Alpine"
            exit 1
        fi
        echo ""
    done

    echo "*************************************************************"
    echo "Completed all runs for $config - Alpine"
    echo "*************************************************************"
done


## v3
for config in "${configs[@]}"; do
    for seed in "${seeds[@]}"; do
        current_run=$((current_run + 1))
        echo "=========================================="
        echo "Run $current_run/$total_runs: $config (seed=$seed) - Alpine"
        echo "=========================================="
        python train.py --config-name="$config" \
            graph_kwargs.knn=6 \
            ++graph_kwargs.theta_strategy=nn_median \
            ++graph_kwargs.theta_scale=0.6 \
            ++graph_kwargs.max_distance_km=100 \
            ++graph_kwargs.mutual=true \
            ++graph_kwargs.threshold=0.45 \
            ++graph_kwargs.orography_var=terrain:elevation_50m \
            ++graph_kwargs.orography_scale=300 \
            ++graph_kwargs.orography_alpha=2.0 \
            logging.run_name="${config}_seed${seed}_alpine_v3" \
            seed="$seed"

        
        if [ $? -eq 0 ]; then
            echo "✓ Completed: $config (seed=$seed) - Alpine"
        else
            echo "✗ Failed: $config (seed=$seed) - Alpine"
            exit 1
        fi
        echo ""
    done

    echo "*************************************************************"
    echo "Completed all runs for $config - Alpine"
    echo "*************************************************************"
done


# echo "=========================================="
# echo "All experiments completed successfully!"
# echo "=========================================="
