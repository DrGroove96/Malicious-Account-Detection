#!/usr/bin/env python3
"""
Example usage script for GNN Malicious Account Detection experiments.
This script demonstrates how to use the main components of the project.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    set_seed, load_mat_to_pyg_data, get_dataset_info, MODEL_DICT, DATASETS,
    split_indices, train_and_evaluate, save_results, create_results_table
)

def example_single_experiment():
    """Example of running a single experiment"""
    print("=" * 60)
    print("Example: Single Experiment")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Check if we have a dataset file
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Creating example with synthetic data.")
        return example_synthetic_data()
    
    # Try to load a dataset
    dataset_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    if not dataset_files:
        print("No .mat files found in data directory. Creating example with synthetic data.")
        return example_synthetic_data()
    
    # Use the first available dataset
    dataset_file = dataset_files[0]
    dataset_name = Path(dataset_file).stem
    print(f"Using dataset: {dataset_name}")
    
    try:
        # Load dataset
        data = load_mat_to_pyg_data(os.path.join(data_dir, dataset_file))
        dataset_info = get_dataset_info(data)
        print(f"Dataset info: {dataset_info}")
        
        # Split data
        train_idx, val_idx, test_idx = split_indices(
            data.x.shape[0], data.y.cpu().numpy(), 
            val_ratio=0.2, test_ratio=0.2, random_state=42
        )
        
        # Initialize model (use GraphSAGE as example)
        model = MODEL_DICT['GraphSAGE'](
            in_channels=data.x.shape[1],
            hidden_channels=64,
            out_channels=int(data.y.max().item()) + 1
        )
        
        # Train and evaluate
        print(f"\nTraining GraphSAGE on {dataset_name}...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        metrics = train_and_evaluate(
            model=model,
            data=data,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            device=device,
            epochs=50,  # Reduced for example
            weighted=True
        )
        
        # Print results
        print(f"\nResults for GraphSAGE on {dataset_name}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and metric not in ['train_losses', 'val_losses', 'val_f1s', 'best_epoch']:
                print(f"  {metric}: {value:.2f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating example with synthetic data.")
        return example_synthetic_data()

def example_synthetic_data():
    """Example using synthetic data when real datasets are not available"""
    print("\n" + "=" * 60)
    print("Example: Synthetic Data Experiment")
    print("=" * 60)
    
    # Create synthetic graph data
    n_nodes = 1000
    n_features = 10
    n_classes = 2
    
    # Create random features
    features = torch.randn(n_nodes, n_features)
    
    # Create random adjacency matrix (sparse)
    edge_prob = 0.01
    edges = torch.randint(0, n_nodes, (int(n_nodes * n_nodes * edge_prob), 2))
    edge_index = edges.t()
    
    # Create labels (imbalanced)
    labels = torch.zeros(n_nodes, dtype=torch.long)
    labels[:int(n_nodes * 0.1)] = 1  # 10% malicious
    
    # Create PyTorch Geometric Data object
    from torch_geometric.data import Data
    data = Data(x=features, edge_index=edge_index, y=labels)
    
    print(f"Synthetic dataset created:")
    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  Features: {data.x.shape[1]}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Classes: {int(data.y.max().item()) + 1}")
    print(f"  Class distribution: {torch.bincount(data.y).tolist()}")
    
    # Split data
    train_idx, val_idx, test_idx = split_indices(
        data.x.shape[0], data.y.cpu().numpy(), 
        val_ratio=0.2, test_ratio=0.2, random_state=42
    )
    
    # Test different models
    models_to_test = ['GCN', 'GraphSAGE', 'HybridGNN']
    results = {}
    
    for model_name in models_to_test:
        print(f"\nTraining {model_name}...")
        
        # Initialize model
        model = MODEL_DICT[model_name](
            in_channels=data.x.shape[1],
            hidden_channels=32,  # Smaller for synthetic data
            out_channels=int(data.y.max().item()) + 1
        )
        
        # Train and evaluate
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        metrics = train_and_evaluate(
            model=model,
            data=data,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            device=device,
            epochs=30,  # Reduced for example
            weighted=True
        )
        
        # Store results
        results[model_name] = metrics
        
        # Print results
        print(f"Results for {model_name}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and metric not in ['train_losses', 'val_losses', 'val_f1s', 'best_epoch']:
                print(f"  {metric}: {value:.2f}")
    
    return results

def example_batch_experiments():
    """Example of running batch experiments"""
    print("\n" + "=" * 60)
    print("Example: Batch Experiments")
    print("=" * 60)
    
    # Set random seed
    set_seed(42)
    
    # Create synthetic data for demonstration
    n_nodes = 500
    n_features = 8
    n_classes = 2
    
    # Create random features
    features = torch.randn(n_nodes, n_features)
    
    # Create random adjacency matrix
    edge_prob = 0.02
    edges = torch.randint(0, n_nodes, (int(n_nodes * n_nodes * edge_prob), 2))
    edge_index = edges.t()
    
    # Create labels
    labels = torch.zeros(n_nodes, dtype=torch.long)
    labels[:int(n_nodes * 0.15)] = 1  # 15% malicious
    
    # Create PyTorch Geometric Data object
    from torch_geometric.data import Data
    data = Data(x=features, edge_index=edge_index, y=labels)
    
    # Split data
    train_idx, val_idx, test_idx = split_indices(
        data.x.shape[0], data.y.cpu().numpy(), 
        val_ratio=0.2, test_ratio=0.2, random_state=42
    )
    
    # Test all models
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_name, ModelClass in MODEL_DICT.items():
        print(f"\nTraining {model_name}...")
        
        # Initialize model
        model = ModelClass(
            in_channels=data.x.shape[1],
            hidden_channels=32,
            out_channels=int(data.y.max().item()) + 1
        )
        
        # Train and evaluate
        metrics = train_and_evaluate(
            model=model,
            data=data,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            device=device,
            epochs=20,  # Reduced for example
            weighted=True
        )
        
        # Store results
        results[model_name] = metrics
    
    # Create comparison table
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    
    comparison_data = []
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Accuracy': f"{metrics['Accuracy']:.2f}",
            'Precision': f"{metrics['Precision']:.2f}",
            'Recall': f"{metrics['Recall']:.2f}",
            'F1-Score': f"{metrics['F1-Score']:.2f}",
            'AUC-ROC': f"{metrics['AUC-ROC']:.2f}"
        }
        comparison_data.append(row)
    
    # Print comparison table
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    return results

def example_visualization():
    """Example of creating visualizations"""
    print("\n" + "=" * 60)
    print("Example: Creating Visualizations")
    print("=" * 60)
    
    # Create synthetic data for visualization
    np.random.seed(42)
    
    # Generate synthetic results
    models = ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'HybridGNN']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    # Create synthetic performance data
    results = {}
    for model in models:
        # Simulate realistic performance
        base_performance = {
            'GCN': 0.75,
            'GAT': 0.78,
            'GraphSAGE': 0.82,
            'GIN': 0.76,
            'HybridGNN': 0.85
        }
        
        base = base_performance[model]
        results[model] = {}
        
        for metric in metrics:
            # Add some variation
            value = base + np.random.normal(0, 0.05)
            if 'Precision' in metric:
                value = base + 0.02 + np.random.normal(0, 0.03)
            elif 'Recall' in metric:
                value = base - 0.03 + np.random.normal(0, 0.04)
            elif 'F1' in metric:
                value = base + 0.01 + np.random.normal(0, 0.02)
            elif 'AUC' in metric:
                value = base + 0.05 + np.random.normal(0, 0.03)
            
            results[model][metric] = max(0, min(100, value * 100))
    
    # Create visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    plot_data = []
    for model in models:
        for metric in metrics:
            plot_data.append({
                'Model': model,
                'Metric': metric,
                'Value': results[model][metric]
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create grouped bar plot
    sns.barplot(data=df, x='Metric', y='Value', hue='Model', palette='viridis')
    plt.title('Model Performance Comparison (Synthetic Data)', fontsize=16, pad=15)
    plt.ylabel('Performance (%)', fontsize=14)
    plt.xlabel('Metric', fontsize=14)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    save_path = "example_visualization.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")
    
    return results

def main():
    """Main function to run all examples"""
    print("GNN Malicious Account Detection - Example Usage")
    print("=" * 60)
    
    # Example 1: Single experiment
    try:
        example_single_experiment()
    except Exception as e:
        print(f"Error in single experiment: {e}")
    
    # Example 2: Batch experiments
    try:
        example_batch_experiments()
    except Exception as e:
        print(f"Error in batch experiments: {e}")
    
    # Example 3: Visualization
    try:
        example_visualization()
    except Exception as e:
        print(f"Error in visualization: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nTo run full experiments:")
    print("1. Place your .mat dataset files in the data/ directory")
    print("2. Run: python train.py --weighted")
    print("3. Run: python train.py --unweighted")
    print("4. Run: python baseline_experiments.py")
    print("5. Run: python visualizations.py")
    print("\nOr use the automated script:")
    print("  ./run_experiments.sh")

if __name__ == '__main__':
    main() 