#!/usr/bin/env python3
"""
Script to calculate mean and standard deviation of CRPS and MAE metrics
across multiple test result files for each model.

This script processes all test_results*.txt files in subdirectories of the utils folder,
extracting CRPS and MAE values for Overall, t=1h, t=24h, t=48h, and t=96h lines.
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def parse_test_results_file(filepath: str) -> Dict[str, Tuple[float, float]]:
    """
    Parse a test results file and extract CRPS and MAE values.
    
    Args:
        filepath: Path to the test results file
        
    Returns:
        Dictionary mapping time keys (e.g., 'Overall', 't=1h') to (CRPS, MAE) tuples
    """
    results = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse line format: "Overall | CRPS: 0.6706 (NWP: 0.9808) | MAE: 0.9788 (NWP: 1.2060)"
            # or "t=   1h | CRPS: 0.6428 (NWP: 0.9804) | MAE: 0.9426 (NWP: 1.1543)"
            
            # Extract the time key (Overall, t=1h, t=24h, etc.)
            if line.startswith('Overall'):
                time_key = 'Overall'
            elif line.startswith('t='):
                # Extract t=1h, t=24h, etc.
                match = re.match(r't=\s*(\d+)h', line)
                if match:
                    time_key = f't={match.group(1)}h'
                else:
                    continue
            else:
                continue
            
            # Extract CRPS value (first occurrence, not the NWP one)
            crps_match = re.search(r'CRPS:\s+([\d.]+)\s+\(NWP:', line)
            mae_match = re.search(r'MAE:\s+([\d.]+)\s+\(NWP:', line)
            
            if crps_match and mae_match:
                crps = float(crps_match.group(1))
                mae = float(mae_match.group(1))
                results[time_key] = (crps, mae)
    
    return results


def calculate_statistics(model_dir: Path) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Calculate mean and std for CRPS and MAE across all test_results*.txt files in a model directory.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Dictionary mapping time keys to statistics:
        {
            'Overall': {'crps_mean': x, 'crps_std': y, 'mae_mean': z, 'mae_std': w},
            't=1h': {...},
            ...
        }
    """
    # Find all test_results*.txt files
    test_files = sorted(model_dir.glob('test_results*.txt'))
    
    if not test_files:
        return {}
    
    # Collect all results
    all_results = []
    for test_file in test_files:
        results = parse_test_results_file(test_file)
        all_results.append(results)
    
    # Organize by time key
    time_keys = ['Overall', 't=1h', 't=24h', 't=48h', 't=96h']
    statistics = {}
    
    for time_key in time_keys:
        crps_values = []
        mae_values = []
        
        for results in all_results:
            if time_key in results:
                crps, mae = results[time_key]
                crps_values.append(crps)
                mae_values.append(mae)
        
        if crps_values and mae_values:
            statistics[time_key] = {
                'crps_mean': np.mean(crps_values),
                'crps_std': np.std(crps_values, ddof=1) if len(crps_values) > 1 else 0.0,
                'mae_mean': np.mean(mae_values),
                'mae_std': np.std(mae_values, ddof=1) if len(mae_values) > 1 else 0.0,
                'n_samples': len(crps_values)
            }
    
    return statistics


def format_output(model_name: str, statistics: Dict) -> str:
    """
    Format statistics output for a model.
    
    Args:
        model_name: Name of the model
        statistics: Statistics dictionary from calculate_statistics
        
    Returns:
        Formatted string output
    """
    output = [f"\n{'='*80}"]
    output.append(f"Model: {model_name}")
    output.append(f"{'='*80}\n")
    
    if not statistics:
        output.append("No test results found.")
        return '\n'.join(output)
    
    time_keys = ['Overall', 't=1h', 't=24h', 't=48h', 't=96h']
    
    for time_key in time_keys:
        if time_key in statistics:
            stats = statistics[time_key]
            
            # Format the time label to match original format
            if time_key == 'Overall':
                time_label = 'Overall'
            else:
                # Extract hour value (e.g., '1' from 't=1h')
                hour = time_key.replace('t=', '').replace('h', '')
                time_label = f't={hour:>3s}h'
            
            # Format like: "Overall | CRPS: 0.6706 ± 0.0123 | MAE: 0.9788 ± 0.0045"
            output.append(
                f"{time_label} | "
                f"CRPS: {stats['crps_mean']:.4f} ± {stats['crps_std']:.4f} | "
                f"MAE: {stats['mae_mean']:.4f} ± {stats['mae_std']:.4f} "
                f"(n={stats['n_samples']})"
            )
    
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate mean and std of CRPS and MAE metrics for model test results'
    )
    parser.add_argument(
        '--utils-dir',
        type=str,
        default='.',
        help='Path to the utils directory (default: current directory)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)'
    )
    
    args = parser.parse_args()
    
    utils_dir = Path(args.utils_dir)
    
    if not utils_dir.exists():
        print(f"Error: Utils directory not found: {utils_dir}")
        return
    
    # Find all model directories (subdirectories containing test_results*.txt files)
    model_dirs = []
    for item in utils_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and not item.name.startswith('__'):
            # Check if directory contains test_results*.txt files
            if list(item.glob('test_results*.txt')):
                model_dirs.append(item)
    
    model_dirs = sorted(model_dirs, key=lambda x: x.name)
    
    if not model_dirs:
        print(f"No model directories with test results found in {utils_dir}")
        return
    
    # Process each model
    all_output = []
    all_output.append(f"CRPS and MAE Statistics Summary")
    all_output.append(f"Generated from: {utils_dir.absolute()}")
    all_output.append(f"Number of models: {len(model_dirs)}")
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        statistics = calculate_statistics(model_dir)
        output = format_output(model_name, statistics)
        all_output.append(output)
    
    final_output = '\n'.join(all_output)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            f.write(final_output)
        print(f"Results written to: {args.output}")
    else:
        print(final_output)


if __name__ == '__main__':
    main()
