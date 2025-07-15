#!/usr/bin/env python3
"""
Baseline machine learning experiments for malicious account detection.
Compares traditional ML models (KNN, MLP, RF, SVC) with GNN performance.
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, DATASETS, save_results, create_results_table

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run baseline ML experiments')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing dataset files')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--figures-dir', type=str, default='figures',
                       help='Directory to save figures')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of runs for each experiment')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save individual plots for each experiment')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of jobs for parallel processing')
    
    return parser.parse_args()

def load_dataset_features(mat_path):
    """Load features and labels from .mat file"""
    mat = sio.loadmat(mat_path)
    features = mat['Attributes']
    labels = mat['Label']
    
    # Convert to numpy arrays
    if hasattr(features, 'toarray'):
        features = features.toarray()
    
    # Handle label dimensions
    if labels.shape[0] == 1:
        labels = labels.flatten()
    elif labels.shape[1] == 1:
        labels = labels.flatten()
    
    return features, labels

def get_baseline_models():
    """Get baseline machine learning models"""
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        'SVC': SVC(probability=True, random_state=42)
    }
    return models

def evaluate_model(model, X_train, X_test, y_train, y_test, scaler=None):
    """Evaluate a single model"""
    # Scale features if scaler is provided
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Compute metrics
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_test, y_pred) * 100
    metrics['Precision'] = precision_score(y_test, y_pred, average='binary', zero_division=0) * 100
    metrics['Recall'] = recall_score(y_test, y_pred, average='binary', zero_division=0) * 100
    metrics['F1-Score'] = f1_score(y_test, y_pred, average='binary', zero_division=0) * 100
    metrics['AUC-ROC'] = roc_auc_score(y_test, y_prob) * 100
    metrics['Avg-Prec'] = average_precision_score(y_test, y_prob) * 100
    
    # Additional metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['Specificity'] = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    metrics['Sensitivity'] = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    
    return metrics, y_pred, y_prob

def run_baseline_experiment(dataset_name, mat_file, model_name, model, args, data_dir):
    """Run a single baseline experiment"""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    # Load dataset
    mat_path = os.path.join(data_dir, mat_file)
    if not os.path.exists(mat_path):
        print(f"Dataset file not found: {mat_path}")
        return None
    
    try:
        features, labels = load_dataset_features(mat_path)
        print(f"Dataset shape: {features.shape}")
        print(f"Class distribution: {np.bincount(labels)}")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Run multiple experiments
    all_metrics = []
    all_predictions = []
    
    for run in range(args.runs):
        print(f"\nRun {run + 1}/{args.runs}")
        
        # Set seed for reproducibility
        set_seed(args.seed + run)
        
        # Split data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=args.seed + run)
        for train_idx, test_idx in sss.split(features, labels):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Further split training data into train/validation
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed + run)
            for train_sub_idx, val_idx in sss_val.split(X_train, y_train):
                X_train_sub = X_train[train_sub_idx]
                X_val = X_train[val_idx]
                y_train_sub = y_train[train_sub_idx]
                y_val = y_train[val_idx]
                break
            break
        
        # Train and evaluate
        start_time = time.time()
        metrics, y_pred, y_prob = evaluate_model(
            model, X_train_sub, X_test, y_train_sub, y_test, scaler
        )
        end_time = time.time()
        
        # Store results
        all_metrics.append(metrics)
        all_predictions.append({
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        })
        
        # Print results
        print(f"Run {run + 1} Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
        print(f"  Training time: {end_time - start_time:.2f}s")
    
    # Aggregate results
    if not all_metrics:
        return None
    
    # Calculate means and standard deviations
    metric_names = list(all_metrics[0].keys())
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
    
    # Save individual plots if requested
    if args.save_plots:
        save_individual_plots(
            dataset_name, model_name, all_predictions, args.figures_dir
        )
    
    return (means, stds)

def save_individual_plots(dataset_name, model_name, all_predictions, figures_dir):
    """Save individual plots for each experiment"""
    # Use the best run for visualization (highest F1 score)
    best_run = 0  # You could implement logic to find the best run
    
    pred_data = all_predictions[best_run]
    y_true = pred_data['y_true']
    y_pred = pred_data['y_pred']
    y_prob = pred_data['y_prob']
    
    # Create subdirectories
    roc_dir = os.path.join(figures_dir, 'roc_curves')
    cm_dir = os.path.join(figures_dir, 'confusion_matrices')
    featimp_dir = os.path.join(figures_dir, 'feature_importance')
    loss_dir = os.path.join(figures_dir, 'loss_curves')
    
    for dir_path in [roc_dir, cm_dir, featimp_dir, loss_dir]:
        Path(dir_path).mkdir(exist_ok=True)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve\n{model_name} on {dataset_name}', fontsize=16, pad=15)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(roc_dir, f'roc_{dataset_name}_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'], 
                yticklabels=['Benign', 'Malicious'])
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Confusion Matrix\n{model_name} on {dataset_name}', fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(cm_dir, f'cm_{dataset_name}_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature Importance (for RF only)
    if model_name == 'RF':
        # This would require access to the trained model
        # For now, we'll create a placeholder
        plt.figure(figsize=(8, 5))
        plt.title(f'Feature Importances\n{model_name} on {dataset_name}', fontsize=16, pad=15)
        plt.ylabel('Importance', fontsize=14)
        plt.xlabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(featimp_dir, f'featimp_{dataset_name}_{model_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Loss Curve (placeholder for ML models)
    plt.figure(figsize=(7, 5))
    plt.title(f'Training Progress\n{model_name} on {dataset_name}', fontsize=16, pad=15)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f'loss_{dataset_name}_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plots(results, args):
    """Create comparison plots across all models and datasets"""
    # Extract data for plotting
    datasets = sorted(list(set([key[0] for key in results.keys()])))
    models = sorted(list(set([key[1] for key in results.keys()])))
    
    # Create comparison plots
    metrics_to_plot = ['Accuracy', 'F1-Score', 'AUC-ROC']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        data_for_plot = []
        for dataset in datasets:
            for model in models:
                if (dataset, model) in results:
                    means, stds = results[(dataset, model)]
                    data_for_plot.append({
                        'Dataset': dataset,
                        'Model': model,
                        'Mean': means[metric],
                        'Std': stds[metric]
                    })
        
        df_plot = pd.DataFrame(data_for_plot)
        
        # Create grouped bar plot
        plt.figure(figsize=(15, 8))
        x = np.arange(len(datasets))
        width = 0.2
        
        for i, model in enumerate(models):
            model_data = df_plot[df_plot['Model'] == model]
            means = [model_data[model_data['Dataset'] == d]['Mean'].iloc[0] if len(model_data[model_data['Dataset'] == d]) > 0 else 0 for d in datasets]
            stds = [model_data[model_data['Dataset'] == d]['Std'].iloc[0] if len(model_data[model_data['Dataset'] == d]) > 0 else 0 for d in datasets]
            
            plt.bar(x + i * width, means, width, label=model, alpha=0.8, yerr=stds, capsize=5)
        
        plt.xlabel('Dataset', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.title(f'{metric} Comparison Across Datasets and Models', fontsize=16, pad=15)
        plt.xticks(x + width * 2, datasets, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(args.figures_dir, f'comparison_{metric.lower()}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved: {save_path}")

def main():
    """Main function"""
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    Path(args.results_dir).mkdir(exist_ok=True)
    Path(args.figures_dir).mkdir(exist_ok=True)
    
    print("Starting baseline ML experiments")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Models: {len(get_baseline_models())}")
    print(f"Runs per experiment: {args.runs}")
    
    # Get baseline models
    baseline_models = get_baseline_models()
    
    # Run experiments
    results = {}
    total_experiments = len(DATASETS) * len(baseline_models)
    completed_experiments = 0
    
    for dataset_name, mat_file in DATASETS:
        for model_name, model in baseline_models.items():
            completed_experiments += 1
            print(f"\nProgress: {completed_experiments}/{total_experiments}")
            
            result = run_baseline_experiment(
                dataset_name, mat_file, model_name, model, args, args.data_dir
            )
            
            if result is not None:
                results[(dataset_name, model_name)] = result
    
    # Save results
    if results:
        # Save detailed results
        results_file = os.path.join(args.results_dir, 'baseline_results.json')
        save_results(results, results_file)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Create and save summary table
        table_file = os.path.join(args.results_dir, 'baseline_results.csv')
        df = create_results_table(results, table_file)
        print(f"Summary table saved to: {table_file}")
        
        # Create comparison plots
        create_comparison_plots(results, args)
        
        # Print LaTeX table
        print("\n" + "="*80)
        print("LATEX TABLE")
        print("="*80)
        print_latex_table(results)
        
        print(f"\nBaseline experiments completed successfully!")
        print(f"Total experiments: {len(results)}")
    else:
        print("No experiments completed successfully.")

def print_latex_table(results):
    """Print results in LaTeX table format"""
    print("\\begin{table*}[htbp]")
    print("\\centering")
    print("\\caption{Baseline ML Performance Metrics (\\%) for Malicious Account Detection}")
    print("\\label{tab:baseline_all_datasets}")
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