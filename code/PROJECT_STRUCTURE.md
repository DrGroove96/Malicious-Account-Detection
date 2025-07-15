# Project Structure

This document describes the complete structure of the GNN Malicious Account Detection project.

## Directory Structure

```
gnn-malicious-detection/
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
├── run_experiments.sh           # Automated experiment runner
├── example_usage.py             # Example usage demonstrations
├── PROJECT_STRUCTURE.md         # This file
│
├── Core Scripts
├── train.py                     # Main GNN training script
├── baseline_experiments.py      # Baseline ML experiments
├── visualizations.py            # Visualization generation
├── curve.py                     # Convergence analysis
├── utils.py                     # Utility functions
│
├── Data Directory (to be created)
├── data/                        # Dataset files
│   ├── Amazon.mat              # Amazon dataset
│   ├── BlogCatalog.mat         # BlogCatalog dataset
│   ├── Flickr.mat              # Flickr dataset
│   ├── Reddit.mat              # Reddit dataset
│   ├── small_amazon.mat        # Small Amazon dataset
│   ├── twitter.mat             # Twitter dataset
│   ├── YelpChi.mat             # YelpChi dataset
│   └── YelpChi-all.mat         # YelpChi-all dataset
│
├── Results Directory (auto-generated)
├── results/                     # Experimental results
│   ├── gnn_results_weighted.json    # GNN results with weighted loss
│   ├── gnn_results_unweighted.json  # GNN results with unweighted loss
│   ├── baseline_results.json        # Baseline ML results
│   ├── gnn_results_weighted.csv     # CSV format results
│   ├── gnn_results_unweighted.csv   # CSV format results
│   └── baseline_results.csv         # CSV format results
│
└── Figures Directory (auto-generated)
    └── figures/                 # Generated visualizations
        ├── roc_curves/          # ROC curve plots
        ├── confusion_matrices/  # Confusion matrix plots
        ├── loss_curves/         # Training loss curves
        ├── convergence_plots/   # Convergence analysis
        ├── comparison_plots/    # Model comparison plots
        └── feature_importance/  # Feature importance plots
```

## File Descriptions

### Core Scripts

#### `train.py`
- **Purpose**: Main training script for GNN experiments
- **Features**:
  - Supports both weighted and unweighted loss functions
  - Implements 5 GNN architectures (GCN, GAT, GraphSAGE, GIN, HybridGNN)
  - Runs multiple experiments with different random seeds
  - Saves training curves and results
  - Generates LaTeX tables for publication
- **Usage**: `python train.py --weighted` or `python train.py --unweighted`

#### `baseline_experiments.py`
- **Purpose**: Baseline machine learning experiments
- **Features**:
  - Implements 4 baseline models (KNN, MLP, RF, SVC)
  - Compares with GNN performance
  - Generates individual and comparison plots
  - Saves results in JSON and CSV formats
- **Usage**: `python baseline_experiments.py`

#### `visualizations.py`
- **Purpose**: Comprehensive visualization generation
- **Features**:
  - ROC curves for all models and datasets
  - Confusion matrices
  - Training loss curves
  - Feature importance plots
  - Performance comparison plots
  - Multiple output formats (PNG, PDF, SVG)
- **Usage**: `python visualizations.py`

#### `curve.py`
- **Purpose**: Convergence analysis and training curves
- **Features**:
  - Analyzes training convergence patterns
  - Creates smoothed convergence curves
  - Generates performance heatmaps
  - Supports real and simulated data
- **Usage**: `python curve.py`

#### `utils.py`
- **Purpose**: Utility functions and shared components
- **Features**:
  - Data loading and preprocessing
  - Model definitions (GCN, GAT, GraphSAGE, GIN, HybridGNN)
  - Training and evaluation functions
  - Results saving and loading
  - Visualization helpers
  - Experiment management utilities

### Configuration Files

#### `requirements.txt`
- Lists all Python dependencies with version constraints
- Includes PyTorch, PyTorch Geometric, scikit-learn, matplotlib, etc.

#### `setup.py`
- Package installation script
- Defines project metadata and dependencies
- Creates command-line entry points

#### `.gitignore`
- Excludes data files, results, figures, and temporary files
- Prevents large files from being committed to Git

### Automation Scripts

#### `run_experiments.sh`
- **Purpose**: Automated experiment runner
- **Features**:
  - Runs all experiments in sequence
  - Installs dependencies
  - Creates necessary directories
  - Provides progress feedback
- **Usage**: `./run_experiments.sh`

#### `example_usage.py`
- **Purpose**: Demonstrates how to use the code
- **Features**:
  - Single experiment example
  - Batch experiments example
  - Visualization example
  - Works with synthetic data if real data unavailable
- **Usage**: `python example_usage.py`

## Data Format

### Input Data (.mat files)
Each dataset file should contain:
- `Network`: Adjacency matrix (sparse)
- `Attributes`: Node features (dense matrix)
- `Label`: Node labels (binary: 0=benign, 1=malicious)

### Output Results
- **JSON format**: Detailed results with training history
- **CSV format**: Summary tables for easy analysis
- **LaTeX format**: Publication-ready tables

## Model Architectures

### GNN Models
1. **GCN (Graph Convolutional Network)**: Standard graph convolution
2. **GAT (Graph Attention Network)**: Attention-based convolution
3. **GraphSAGE**: Inductive graph neural network
4. **GIN (Graph Isomorphism Network)**: Graph isomorphism network
5. **HybridGNN**: Combination of GCN and GAT layers

### Baseline Models
1. **KNN**: k-Nearest Neighbors
2. **MLP**: Multi-Layer Perceptron
3. **RF**: Random Forest
4. **SVC**: Support Vector Classifier

## Evaluation Metrics

The experiments evaluate models using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for malicious class
- **Recall**: Recall for malicious class
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **Average Precision**: Area under the precision-recall curve
- **Specificity**: True negative rate
- **Sensitivity**: True positive rate

## Workflow

1. **Setup**: Install dependencies and prepare data
2. **GNN Experiments**: Run weighted and unweighted loss experiments
3. **Baseline Experiments**: Run traditional ML models
4. **Analysis**: Generate convergence curves and performance summaries
5. **Visualization**: Create all plots and figures
6. **Results**: Review CSV tables and generated figures

## Customization

### Adding New Models
1. Define the model class in `utils.py`
2. Add to `MODEL_DICT` in `utils.py`
3. Update visualization scripts if needed

### Adding New Datasets
1. Place .mat file in `data/` directory
2. Add to `DATASETS` list in `utils.py`
3. Ensure proper data format (Network, Attributes, Label)

### Modifying Experiments
1. Edit hyperparameters in training scripts
2. Modify evaluation metrics in `utils.py`
3. Update visualization functions as needed

## Dependencies

### Core Dependencies
- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric 2.0+
- scikit-learn 1.0+
- matplotlib 3.5+
- seaborn 0.11+
- numpy 1.21+
- pandas 1.3+

### Optional Dependencies
- CUDA (for GPU acceleration)
- Jupyter (for interactive analysis)
- pytest (for testing)

## Performance Considerations

- **GPU Usage**: Automatically detects and uses CUDA if available
- **Memory Management**: Handles large graphs efficiently
- **Parallel Processing**: Supports multi-GPU training
- **Early Stopping**: Prevents overfitting and reduces training time
- **Checkpointing**: Saves best models during training

## Troubleshooting

### Common Issues
1. **Missing datasets**: Use `example_usage.py` for synthetic data
2. **Memory errors**: Reduce batch size or use smaller models
3. **CUDA errors**: Set `--device cpu` to use CPU only
4. **Import errors**: Install dependencies with `pip install -r requirements.txt`

### Getting Help
1. Check the example usage script
2. Review error messages in console output
3. Verify data format and file paths
4. Ensure all dependencies are installed correctly 