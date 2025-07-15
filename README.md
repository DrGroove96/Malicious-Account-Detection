Of course. Here is a complete, well-formatted `README.md` file ready for you to copy and paste directly into GitHub.

***

```markdown
# Malicious Account Detection in Social Networks via Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.5-orange.svg)](https://pyg.org/)

This repository contains the official code and experiments for the thesis **"Malicious Accounts Detection in Social Networks via Graph Neural Networks"** by Shen-Han Chiu, submitted in partial fulfillment of the requirements for a baccalaureate degree with honors at The Pennsylvania State University.

## 📖 Introduction

With the widespread usage of social media, malicious accounts and social scam bots have become a major concern. This project addresses the critical challenge of detecting these accounts by leveraging the power of **Graph Neural Networks (GNNs)**. Unlike traditional methods that often analyze users in isolation, this work treats social networks as interconnected graphs, allowing models to learn from user features, their connections, and the overall network structure.

This research provides a comprehensive comparison between various GNN architectures and traditional machine learning baselines, demonstrating the superior performance of graph-based methods in identifying malicious actors in complex social systems.

## ✨ Key Features

*   **State-of-the-Art GNNs:** Implementation and evaluation of five GNN architectures:
    *   Graph Convolutional Network (GCN)
    *   Graph Attention Network (GAT)
    *   **GraphSAGE** (shown to be the top performer)
    *   Graph Isomorphism Network (GIN)
    *   HybridGNN
*   **Baseline Machine Learning Models:** Four traditional models for robust performance comparison:
    *   K-Nearest Neighbors (KNN)
    *   Multilayer Perceptron (MLP)
    *   Random Forest (RF)
    *   Support Vector Classifier (SVC)
*   **Real-World Datasets:** Experiments conducted on nine diverse social network graph datasets.
*   **Handling Class Imbalance:** Employs a weighted cross-entropy loss function to effectively train models on imbalanced datasets, where malicious accounts are typically the minority.
*   **In-Depth Analysis:** Includes extensive performance evaluation using metrics like Accuracy, Precision, Recall, F1-Score, and AUC-ROC, supported by data visualizations.

## 🔬 Methodology

The core methodology compares two paradigms for malicious account detection:

1.  **Graph Neural Networks (GNNs):** These models learn node representations by aggregating information from their neighbors, effectively capturing both user attributes and relational patterns.
2.  **Traditional Machine Learning:** These baseline models classify users based on their features alone, without considering the rich structural information of the social graph.

By performing a large-scale analysis across multiple datasets, this work systematically evaluates which models and techniques are most successful for this task.

## 📊 Datasets

The experiments are performed on nine benchmark datasets, originating from the paper "A comprehensive survey on graph anomaly detection with deep learning" by Ma et al. (2021).

*   Amazon
*   BlogCatalog
*   Flickr
*   Reddit
*   Reddit_pt
*   Small_Amazon
*   Twitter
*   YelpChi
*   YelpChi-All

## ⚙️ Installation

To set up the environment and run the experiments, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/DrGroove96/Malicious-Account-Detection.git
cd Malicious-Account-Detection

# Install dependencies using pip
pip install torch torch-geometric scikit-learn pandas matplotlib
```

## 🚀 Usage

The scripts are designed to be run from the command line, allowing you to easily specify the model and dataset for each experiment.

1.  Ensure the required datasets are placed in the appropriate directory within the project.
2.  Run the experiment scripts.

**Example: Training a GNN Model**
To train and evaluate the **GraphSAGE** model on the **Amazon** dataset:
```bash
python train_gnn.py --model GraphSAGE --dataset Amazon
```

**Example: Training a Baseline Model**
To train and evaluate the **KNN** baseline model on the **Amazon** dataset:
```bash
python train_baseline.py --model KNN --dataset Amazon
```

## 📈 Results

The experimental results consistently demonstrate that **GraphSAGE significantly outperforms** both traditional machine learning models and other GNN architectures across the majority of datasets.

**Key Findings:**
*   GNNs that leverage both node features and graph structure are superior for malicious account detection.
*   **GraphSAGE** achieves the highest performance in terms of accuracy, F1-score, and other key metrics.
*   The use of a weighted loss function is crucial for achieving robust performance on imbalanced real-world data.

For detailed performance tables, loss curves, ROC curves, and other visualizations, please refer to the full thesis document.

## ✍️ Citation

If you use the code or findings from this project in your research, please cite the following thesis:

```bibtex
@phdthesis{chiu2025malicious,
  author  = {Chiu, Shen-Han},
  title   = {Malicious Accounts Detection in Social Networks via Graph Neural Networks},
  school  = {The Pennsylvania State University},
  year    = {2025},
  note    = {Schreyer Honors College Thesis}
}
```

## 👨‍💻 Author

*   **Shen-Han Chiu** - [LinkedIn Profile](https://www.linkedin.com/in/chiushenhan/)

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```
