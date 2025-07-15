import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
import random
import scipy.io as sio
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import pandas as pd
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ========== DATA LOADING ==========
def load_mat_to_pyg_data(mat_path):
    """Load .mat file and convert to PyTorch Geometric Data object"""
    mat = sio.loadmat(mat_path)
    if 'Network' not in mat or 'Attributes' not in mat or 'Label' not in mat:
        raise ValueError(f"File {mat_path} missing required keys.")
    
    adj = mat['Network']
    features = mat['Attributes']
    labels = mat['Label']
    
    # Ensure features is a 2D array
    if not hasattr(features, 'shape') or len(features.shape) != 2:
        raise ValueError(f"Features in {mat_path} is not a 2D array.")
    
    # Handle label dimensions
    if labels.shape[0] == 1:
        labels = labels.flatten()
    elif labels.shape[1] == 1:
        labels = labels.flatten()
    
    # Convert to PyTorch Geometric format
    edge_index, _ = from_scipy_sparse_matrix(adj)
    if hasattr(features, 'toarray'):
        features = features.toarray()
    
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data

def get_dataset_info(data):
    """Get basic information about a dataset"""
    info = {
        'num_nodes': data.x.shape[0],
        'num_features': data.x.shape[1],
        'num_edges': data.edge_index.shape[1],
        'num_classes': int(data.y.max().item()) + 1,
        'class_distribution': torch.bincount(data.y).tolist()
    }
    return info

# ========== MODEL DEFINITIONS ==========
class GCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    """GraphSAGE Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

class GIN(torch.nn.Module):
    """Graph Isomorphism Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels), 
            torch.nn.ReLU(), 
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels), 
            torch.nn.ReLU(), 
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        nn3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels), 
            torch.nn.ReLU(), 
            torch.nn.Linear(hidden_channels, out_channels)
        )
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.conv3 = GINConv(nn3)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

class HybridGNN(torch.nn.Module):
    """Hybrid GNN combining GCN and GAT"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=True, dropout=dropout)
        self.conv3 = GCNConv(hidden_channels * 4, out_channels)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

# ========== TRAINING AND EVALUATION ==========
def compute_class_weights(y):
    """Compute class weights for imbalanced datasets"""
    classes, counts = np.unique(y, return_counts=True)
    weights = np.sum(counts) / (counts * len(classes))
    return torch.tensor(weights, dtype=torch.float32)

def evaluate(y_true, y_pred, y_prob):
    """Compute comprehensive evaluation metrics"""
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred) * 100
    metrics['Precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0) * 100
    metrics['Recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0) * 100
    metrics['F1-Score'] = f1_score(y_true, y_pred, average='binary', zero_division=0) * 100
    metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob) * 100
    metrics['Avg-Prec'] = average_precision_score(y_true, y_prob) * 100
    
    # Additional metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['Specificity'] = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    metrics['Sensitivity'] = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    
    return metrics

def split_indices(n_nodes, y, val_ratio=0.2, test_ratio=0.2, random_state=42):
    """Split indices into train/validation/test sets"""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio+test_ratio, random_state=random_state)
    idx = np.arange(n_nodes)
    
    for train_idx, temp_idx in sss.split(idx, y):
        # Further split temp_idx into validation and test
        sss_temp = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio/(val_ratio+test_ratio), random_state=random_state)
        for val_idx, test_idx in sss_temp.split(temp_idx, y[temp_idx]):
            val_idx = temp_idx[val_idx]
            test_idx = temp_idx[test_idx]
            break
        break
    
    return (torch.tensor(train_idx, dtype=torch.long), 
            torch.tensor(val_idx, dtype=torch.long), 
            torch.tensor(test_idx, dtype=torch.long))

def train_and_evaluate(model, data, train_idx, val_idx, test_idx, device, 
                      epochs=200, lr=0.01, weight_decay=5e-4, weighted=True, 
                      early_stopping_patience=20):
    """Train and evaluate a GNN model"""
    model = model.to(device)
    data = data.to(device)
    
    # Setup loss function
    if weighted:
        y_train = data.y[train_idx].cpu().numpy()
        class_weights = compute_class_weights(y_train).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    # Training history
    train_losses = []
    val_losses = []
    val_f1s = []
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = criterion(val_out[val_idx], data.y[val_idx])
            val_pred = val_out[val_idx].argmax(dim=1)
            val_f1 = f1_score(data.y[val_idx].cpu(), val_pred.cpu(), average='binary', zero_division=0)
        
        # Record history
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        val_f1s.append(val_f1)
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)
        y_prob = torch.softmax(test_out[test_idx], dim=1)[:, 1].cpu().numpy()
        y_pred = test_out[test_idx].argmax(dim=1).cpu().numpy()
        y_true = data.y[test_idx].cpu().numpy()
    
    # Compute metrics
    metrics = evaluate(y_true, y_pred, y_prob)
    
    # Add training history
    metrics['train_losses'] = train_losses
    metrics['val_losses'] = val_losses
    metrics['val_f1s'] = val_f1s
    metrics['best_epoch'] = np.argmax(val_f1s)
    
    return metrics

# ========== EXPERIMENT MANAGEMENT ==========
def save_results(results, filename):
    """Save experimental results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    serializable_results[key][k] = v.tolist()
                else:
                    serializable_results[key][k] = v
        else:
            serializable_results[key] = value
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def load_results(filename):
    """Load experimental results from JSON file"""
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

def create_results_table(results, save_path=None):
    """Create a formatted results table"""
    # Extract dataset and model names
    datasets = set()
    models = set()
    for key in results.keys():
        if isinstance(key, tuple) and len(key) == 2:
            dataset, model = key
            datasets.add(dataset)
            models.add(model)
    
    datasets = sorted(list(datasets))
    models = sorted(list(models))
    
    # Create DataFrame
    data = []
    for dataset in datasets:
        for model in models:
            if (dataset, model) in results:
                means, stds = results[(dataset, model)]
                row = {
                    'Dataset': dataset,
                    'Model': model,
                    'Accuracy': f"{means['Accuracy']:.2f} ± {stds['Accuracy']:.2f}",
                    'Precision': f"{means['Precision']:.2f} ± {stds['Precision']:.2f}",
                    'Recall': f"{means['Recall']:.2f} ± {stds['Recall']:.2f}",
                    'F1-Score': f"{means['F1-Score']:.2f} ± {stds['F1-Score']:.2f}",
                    'AUC-ROC': f"{means['AUC-ROC']:.2f} ± {stds['AUC-ROC']:.2f}",
                    'Avg-Prec': f"{means['Avg-Prec']:.2f} ± {stds['Avg-Prec']:.2f}"
                }
                data.append(row)
    
    df = pd.DataFrame(data)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df

# ========== VISUALIZATION HELPERS ==========
def plot_training_curves(train_losses, val_losses, val_f1s, model_name, dataset_name, save_path=None):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Curves: {model_name} on {dataset_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 score curve
    ax2.plot(val_f1s, label='Validation F1', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title(f'F1 Score: {model_name} on {dataset_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ========== DATASET AND MODEL CONFIGURATIONS ==========
DATASETS = [
    ("Amazon", "Amazon.mat"),
    ("BlogCatalog", "BlogCatalog.mat"),
    ("Flickr", "Flickr.mat"),
    ("Reddit", "Reddit.mat"),
    ("Reddit PT", "Reddit.mat"),
    ("Small Amazon", "small_amazon.mat"),
    ("Twitter", "twitter.mat"),
    ("YelpChi-All", "YelpChi-all.mat"),
    ("YelpChi", "YelpChi.mat"),
]

MODEL_DICT = {
    'GCN': GCN,
    'GAT': GAT,
    'GraphSAGE': GraphSAGE,
    'GIN': GIN,
    'HybridGNN': HybridGNN
}

# Import PyTorch Geometric modules
try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
except ImportError:
    print("Warning: torch_geometric not installed. GNN models will not be available.")
    GCNConv = GATConv = SAGEConv = GINConv = None 