# Graph Neural Networks for Malicious Account Detection

This repository contains the implementation and experimental results for detecting malicious accounts using Graph Neural Networks (GNNs) across multiple datasets.

## Overview

This project investigates the effectiveness of various GNN architectures (GCN, GAT, GraphSAGE, GIN, and HybridGNN) for malicious account detection on nine different social network datasets. The experiments compare both weighted and unweighted loss functions and include baseline machine learning models for comparison.

## Datasets

The experiments use the following nine datasets:
- **Amazon**: E-commerce review network
- **BlogCatalog**: Blog social network
- **Flickr**: Photo-sharing social network
- **Reddit**: Online discussion forum network
- **Reddit PT**: Reddit network with different preprocessing
- **Small Amazon**: Subset of Amazon dataset
- **Twitter**: Social media network
- **YelpChi**: Yelp review network
- **YelpChi-All**: Complete Yelp review network

## Models

### Graph Neural Networks
- **GCN (Graph Convolutional Network)**: Standard graph convolution layers
- **GAT (Graph Attention Network)**: Attention-based graph convolution
- **GraphSAGE**: Inductive graph neural network
- **GIN (Graph Isomorphism Network)**: Graph isomorphism network
- **HybridGNN**: Combination of GCN and GAT layers

### Baseline Models
- **KNN**: k-Nearest Neighbors
- **MLP**: Multi-Layer Perceptron
- **RF**: Random Forest
- **SVC**: Support Vector Classifier

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd gnn-malicious-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running GNN Experiments

```bash
# Run GNN experiments with weighted loss
python train.py --weighted

# Run GNN experiments with unweighted loss
python train.py --unweighted

# Run baseline ML experiments
python baseline_experiments.py

# Generate visualizations
python visualizations.py
```

### Key Scripts

- `train.py`: Main training script for GNN models
- `baseline_experiments.py`: Baseline machine learning experiments
- `visualizations.py`: Generate all plots and visualizations
- `curve.py`: Convergence analysis and training curves
- `utils.py`: Utility functions for data loading and evaluation

## Results

The experiments evaluate models using the following metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for malicious class detection
- **Recall**: Recall for malicious class detection
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **Average Precision**: Area under the precision-recall curve

### Key Findings

1. **Weighted vs Unweighted Loss**: Weighted loss functions generally improve performance on imbalanced datasets
2. **GNN Performance**: GraphSAGE and HybridGNN show superior performance across most datasets
3. **Dataset Characteristics**: Performance varies significantly based on dataset size and class imbalance
4. **Convergence**: Different GNN architectures show varying convergence patterns

## File Structure

```
├── README.md
├── requirements.txt
├── train.py                 # Main GNN training script
├── baseline_experiments.py  # Baseline ML experiments
├── visualizations.py        # Visualization generation
├── curve.py                # Convergence analysis
├── utils.py                # Utility functions
├── data/                   # Dataset files
│   ├── Amazon.mat
│   ├── BlogCatalog.mat
│   ├── Flickr.mat
│   ├── Reddit.mat
│   ├── small_amazon.mat
│   ├── twitter.mat
│   ├── YelpChi.mat
│   └── YelpChi-all.mat
├── results/                # Experimental results
│   ├── gnn_results.csv
│   ├── baseline_results.csv
│   └── convergence_data.csv
└── figures/               # Generated visualizations
    ├── roc_curves/
    ├── loss_curves/
    ├── confusion_matrices/
    └── convergence_plots/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@thesis{chiu2024gnn,
  title={Graph Neural Networks for Malicious Account Detection},
  author={Shen-Han Chiu},
  year={2024},
  school={Penn State University}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact [your-email@example.com] 