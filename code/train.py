#!/usr/bin/env python3
"""
Main training script for GNN experiments on malicious account detection.
Supports both weighted and unweighted loss functions.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    set_seed, load_mat_to_pyg_data, get_dataset_info, MODEL_DICT, DATASETS,
    split_indices, train_and_evaluate, save_results, create_results_table,
    plot_training_curves
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train GNN models for malicious account detection')
    parser.add_argument('--weighted', action='store_true', 
                       help='Use weighted loss function')
    parser.add_argument('--unweighted', action='store_true',
                       help='Use unweighted loss function')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension size')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of runs for each experiment')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing dataset files')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--figures-dir', type=str, default='figures',
                       help='Directory to save figures')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save-training-curves', action='store_true',
                       help='Save training curves for each experiment')
    parser.add_argument('--early-stopping-patience', type=int, default=20,
                       help='Early stopping patience')
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup device for training"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def create_directories(results_dir, figures_dir):
    """Create necessary directories"""
    Path(results_dir).mkdir(exist_ok=True)
    Path(figures_dir).mkdir(exist_ok=True)
    Path(figures_dir + '/training_curves').mkdir(exist_ok=True)
    Path(figures_dir + '/confusion_matrices').mkdir(exist_ok=True)

def run_experiment(dataset_name, mat_file, model_name, ModelClass, device, args, data_dir):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Loss: {'Weighted' if args.weighted else 'Unweighted'}")
    print(f"{'='*60}")
    
    # Load dataset
    mat_path = os.path.join(data_dir, mat_file)
    if not os.path.exists(mat_path):
        print(f"Dataset file not found: {mat_path}")
        return None
    
    try:
        data = load_mat_to_pyg_data(mat_path)
        dataset_info = get_dataset_info(data)
        print(f"Dataset info: {dataset_info}")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None
    
    # Run multiple experiments
    all_metrics = []
    all_training_curves = []
    
    for run in range(args.runs):
        print(f"\nRun {run + 1}/{args.runs}")
        
        # Set seed for reproducibility
        set_seed(args.seed + run)
        
        # Split data
        train_idx, val_idx, test_idx = split_indices(
            data.x.shape[0], data.y.cpu().numpy(), 
            val_ratio=0.2, test_ratio=0.2, random_state=args.seed + run
        )
        
        # Initialize model
        model = ModelClass(
            in_channels=data.x.shape[1],
            hidden_channels=args.hidden_dim,
            out_channels=int(data.y.max().item()) + 1
        )
        
        # Train and evaluate
        start_time = time.time()
        metrics = train_and_evaluate(
            model=model,
            data=data,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            weighted=args.weighted,
            early_stopping_patience=args.early_stopping_patience
        )
        end_time = time.time()
        
        # Store results
        all_metrics.append(metrics)
        all_training_curves.append({
            'train_losses': metrics['train_losses'],
            'val_losses': metrics['val_losses'],
            'val_f1s': metrics['val_f1s']
        })
        
        # Print results
        print(f"Run {run + 1} Results:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and metric not in ['train_losses', 'val_losses', 'val_f1s', 'best_epoch']:
                print(f"  {metric}: {value:.2f}")
        print(f"  Training time: {end_time - start_time:.2f}s")
    
    # Aggregate results
    if not all_metrics:
        return None
    
    # Calculate means and standard deviations
    metric_names = [k for k in all_metrics[0].keys() 
                   if isinstance(all_metrics[0][k], (int, float)) and 
                   k not in ['train_losses', 'val_losses', 'val_f1s', 'best_epoch']]
    
    means = {}
    stds = {}
    for metric in metric_names:
        values = [m[metric] for m in all_metrics]
        means[metric] = np.mean(values)
        stds[metric] = np.std(values)
    
    # Print aggregated results
    print(f"\nAggregated Results ({args.runs} runs):")
    for metric in metric_names:
        print(f"  {metric}: {means[metric]:.2f} ± {stds[metric]:.2f}")
    
    # Save training curves if requested
    if args.save_training_curves:
        # Use the best run for visualization
        best_run = np.argmax([m['F1-Score'] for m in all_metrics])
        curves = all_training_curves[best_run]
        
        save_path = os.path.join(
            args.figures_dir, 'training_curves',
            f'{dataset_name}_{model_name}_{"weighted" if args.weighted else "unweighted"}_curves.png'
        )
        plot_training_curves(
            curves['train_losses'], curves['val_losses'], curves['val_f1s'],
            model_name, dataset_name, save_path
        )
        print(f"Training curves saved to: {save_path}")
    
    return (means, stds)

def main():
    """Main function"""
    args = parse_args()
    
    # Validate arguments
    if not args.weighted and not args.unweighted:
        print("Error: Must specify either --weighted or --unweighted")
        sys.exit(1)
    
    # Setup
    set_seed(args.seed)
    device = setup_device(args.device)
    create_directories(args.results_dir, args.figures_dir)
    
    # Determine loss type
    loss_type = "weighted" if args.weighted else "unweighted"
    
    print(f"Starting GNN experiments with {loss_type} loss")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Models: {len(MODEL_DICT)}")
    print(f"Runs per experiment: {args.runs}")
    print(f"Epochs: {args.epochs}")
    
    # Run experiments
    results = {}
    total_experiments = len(DATASETS) * len(MODEL_DICT)
    completed_experiments = 0
    
    for dataset_name, mat_file in DATASETS:
        for model_name, ModelClass in MODEL_DICT.items():
            completed_experiments += 1
            print(f"\nProgress: {completed_experiments}/{total_experiments}")
            
            result = run_experiment(
                dataset_name, mat_file, model_name, ModelClass, 
                device, args, args.data_dir
            )
            
            if result is not None:
                results[(dataset_name, model_name)] = result
    
    # Save results
    if results:
        # Save detailed results
        results_file = os.path.join(args.results_dir, f'gnn_results_{loss_type}.json')
        save_results(results, results_file)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Create and save summary table
        table_file = os.path.join(args.results_dir, f'gnn_results_{loss_type}.csv')
        df = create_results_table(results, table_file)
        print(f"Summary table saved to: {table_file}")
        
        # Print LaTeX table
        print("\n" + "="*80)
        print("LATEX TABLE")
        print("="*80)
        print_latex_table(results)
        
        print(f"\nExperiments completed successfully!")
        print(f"Total experiments: {len(results)}")
    else:
        print("No experiments completed successfully.")

def print_latex_table(results):
    """Print results in LaTeX table format"""
    print("\\begin{table*}[htbp]")
    print("\\centering")
    print("\\caption{Performance Metrics (\\%) for Malicious Account Detection across All Datasets}")
    print("\\label{tab:gnn_all_datasets}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{|l|l|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{AUC-ROC} & \\textbf{Avg-Prec} \\\\")
    print("\\hline")
    
    # Group by dataset
    datasets = sorted(list(set([key[0] for key in results.keys()])))
    models = sorted(list(set([key[1] for key in results.keys()])))
    
    for dataset in datasets:
        first = True
        for model in models:
            if (dataset, model) in results:
                means, stds = results[(dataset, model)]
                row = (
                    f"{dataset if first else ''} & {model} "
                    f"& {means['Accuracy']:.2f} $\\pm$ {stds['Accuracy']:.2f} "
                    f"& {means['Precision']:.2f} $\\pm$ {stds['Precision']:.2f} "
                    f"& {means['Recall']:.2f} $\\pm$ {stds['Recall']:.2f} "
                    f"& {means['F1-Score']:.2f} $\\pm$ {stds['F1-Score']:.2f} "
                    f"& {means['AUC-ROC']:.2f} $\\pm$ {stds['AUC-ROC']:.2f} "
                    f"& {means['Avg-Prec']:.2f} $\\pm$ {stds['Avg-Prec']:.2f} \\\\"
                )
                print(row)
                first = False
        print("\\hline")
    
    print("\\end{tabular}")
    print("}")
    print("\\end{table*}")

if __name__ == '__main__':
    main() 