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

## 📊 Dataset Statistics

The experiments in this research were conducted on nine real-world social network and graph-based datasets. The statistics for each are summarized below. These datasets originate from the paper "A comprehensive survey on graph anomaly detection with deep learning" by Ma et al. (2021).

| Dataset        |         Nodes |           Edges |        Features | Classes | Avg. Degree |   Density | Components | Clustering Coeff. |
| :------------- | --------------:| ----------------:| ----------------:| --------:| ------------:| ----------:| -----------:| ------------------:|
| **Amazon**     |         10,224 |         175,608 |              25 |        2 |       34.35 | 0.003360   |         331 |              0.691 |
| **BlogCatalog**|          5,196 |         172,783 |           8,189 |        2 |       66.51 | 0.012802   |           1 |              0.123 |
| **Flickr**     |          7,575 |         241,304 |          12,047 |        9 |       63.71 | 0.008412   |           1 |              0.330 |
| **Reddit**     |         10,984 |          89,500 |              64 |        2 |       16.30 | 0.001484   |           3 |              0.000 |
| **Reddit PT**  |         10,984 |          89,500 |              64 |        2 |       16.30 | 0.001484   |           3 |              0.000 |
| **Small Amazon**|         1,549 |          18,983 |             661 |        2 |       24.51 | 0.015833   |           1 |              0.640 |
| **Twitter**    |          4,865 |         139,305 |             820 |        2 |       57.27 | 0.011774   |           9 |              0.267 |
| **YelpChi-all**|         45,941 |       3,846,979 |              32 |        2 |      167.47 | 0.003646   |          13 |              0.774 |
| **YelpChi**    |         23,831 |          49,315 |              32 |        2 |        4.14 | 0.000174   |       7,308 |              0.658 |

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

## 📈 Full Experimental Results

This section provides the complete, unabridged performance metrics from the thesis. The tables are placed within collapsible sections to maintain readability. Click on any table title to view the full data.

---

<details>
<summary><b>Table 4.2: Full GNN Performance Metrics (Unweighted Loss)</b></summary>

| Dataset | Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC-ROC (%) | Avg-Prec (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **amazon** | GCN | 73.69 ± 14.73 | 19.86 ± 6.61 | 72.58 ± 13.87 | 29.88 ± 7.28 | 80.86 ± 3.57 | 25.01 ± 3.39 |
| | GAT | 73.20 ± 21.55 | 19.68 ± 8.37 | 59.89 ± 28.46 | 25.25 ± 9.31 | 71.37 ± 12.04 | 20.28 ± 8.76 |
| | GraphSAGE | 96.53 ± 0.76 | 73.17 ± 8.27 | 78.76 ± 3.44 | 75.54 ± 4.53 | 93.88 ± 1.17 | 81.04 ± 3.09 |
| | GIN | 73.22 ± 30.30 | 11.10 ± 6.76 | 28.83 ± 32.82 | 11.57 ± 5.41 | 52.94 ± 2.75 | 8.07 ± 1.03 |
| | HybridGNN | 72.64 ± 15.77 | 18.14 ± 6.11 | 66.02 ± 20.14 | 26.48 ± 7.14 | 77.73 ± 6.38 | 24.09 ± 4.79 |
| **blogcatalog** | GCN | 66.54 ± 22.02 | 14.93 ± 11.03 | 51.99 ± 23.80 | 17.68 ± 6.57 | 65.48 ± 5.24 | 16.78 ± 5.74 |
| | GAT | 86.59 ± 6.49 | 25.29 ± 9.74 | 48.23 ± 11.19 | 31.37 ± 8.19 | 71.84 ± 4.68 | 34.51 ± 10.61 |
| | GraphSAGE | 94.44 ± 0.76 | 52.76 ± 7.15 | 37.23 ± 9.31 | 43.04 ± 7.32 | 77.48 ± 6.24 | 36.77 ± 7.06 |
| | GIN | 41.33 ± 35.49 | 5.46 ± 4.42 | 61.28 ± 39.49 | 9.08 ± 5.07 | 52.20 ± 5.41 | 6.70 ± 2.35 |
| | HybridGNN | 88.36 ± 9.76 | 35.11 ± 13.71 | 58.87 ± 12.09 | 41.56 ± 11.53 | 80.94 ± 5.74 | 48.51 ± 8.47 |
| **flickr** | GCN | 88.13 ± 11.12 | 37.08 ± 14.45 | 57.82 ± 8.79 | 42.44 ± 10.92 | 78.47 ± 3.26 | 55.85 ± 4.48 |
| | GAT | 89.63 ± 3.05 | 32.50 ± 10.39 | 54.37 ± 6.08 | 39.48 ± 6.81 | 76.16 ± 3.50 | 44.16 ± 7.25 |
| | GraphSAGE | 92.85 ± 1.12 | 44.12 ± 6.23 | 60.15 ± 5.10 | 50.85 ± 4.02 | 81.23 ± 2.91 | 54.12 ± 4.01 |
| | GIN | 85.21 ± 7.34 | 28.45 ± 8.12 | 55.23 ± 9.45 | 37.45 ± 6.23 | 75.12 ± 3.87 | 41.23 ± 5.12 |
| | HybridGNN | 93.37 ± 1.03 | 46.33 ± 5.90 | 59.72 ± 5.22 | 51.80 ± 3.91 | 82.17 ± 2.85 | 55.79 ± 4.38 |
| **reddit** | GCN | 49.06 ± 8.20 | 4.32 ± 0.53 | 67.41 ± 10.27 | 8.10 ± 0.92 | 63.37 ± 1.89 | 6.14 ± 1.06 |
| | GAT | 90.93 ± 17.20 | 2.24 ± 4.10 | 9.15 ± 25.74 | 1.90 ± 3.47 | 65.81 ± 2.33 | 5.54 ± 1.06 |
| | GraphSAGE | 41.93 ± 41.10 | 3.31 ± 2.52 | 64.94 ± 42.16 | 6.07 ± 4.29 | 64.98 ± 6.81 | 5.72 ± 1.41 |
| | GIN | 35.11 ± 18.88 | 4.20 ± 0.70 | 81.99 ± 20.35 | 7.91 ± 1.11 | 62.28 ± 3.46 | 5.12 ± 1.16 |
| | HybridGNN | 40.33 ± 5.81 | 4.33 ± 0.55 | 80.14 ± 8.05 | 8.20 ± 0.98 | 62.25 ± 1.96 | 4.84 ± 0.92 |
| **reddit\_pt** | GCN | 49.59 ± 7.23 | 4.33 ± 0.55 | 66.79 ± 9.06 | 8.12 ± 0.96 | 63.42 ± 1.85 | 6.15 ± 1.06 |
| | GAT | 85.82 ± 19.99 | 2.81 ± 2.95 | 16.85 ± 29.10 | 3.61 ± 3.86 | 65.78 ± 2.54 | 5.56 ± 1.12 |
| | GraphSAGE | 32.29 ± 38.45 | 3.00 ± 1.93 | 73.44 ± 41.03 | 5.65 ± 3.42 | 64.25 ± 6.98 | 5.62 ± 1.40 |
| | GIN | 40.22 ± 23.12 | 4.20 ± 0.85 | 74.55 ± 26.58 | 7.81 ± 1.38 | 61.38 ± 4.13 | 5.00 ± 1.03 |
| | HybridGNN | 41.02 ± 5.43 | 4.35 ± 0.58 | 79.43 ± 7.59 | 8.23 ± 1.04 | 62.40 ± 1.98 | 4.88 ± 0.90 |
| **small\_amazon**| GCN | 71.92 ± 6.60 | 19.29 ± 4.83 | 56.01 ± 11.23 | 28.06 ± 5.40 | 70.61 ± 5.07 | 29.26 ± 6.51 |
| | GAT | 78.08 ± 5.96 | 24.92 ± 9.18 | 53.71 ± 7.49 | 33.06 ± 8.20 | 72.88 ± 5.16 | 33.84 ± 9.10 |
| | GraphSAGE | 94.55 ± 1.60 | 73.77 ± 8.86 | 70.02 ± 12.75 | 71.04 ± 8.61 | 90.50 ± 3.98 | 72.68 ± 7.38 |
| | GIN | 61.14 ± 9.19 | 15.32 ± 3.62 | 63.64 ± 11.42 | 24.33 ± 4.65 | 68.57 ± 5.20 | 28.17 ± 11.26 |
| | HybridGNN | 81.07 ± 4.75 | 28.18 ± 7.37 | 55.70 ± 10.81 | 36.51 ± 6.41 | 75.36 ± 5.26 | 37.23 ± 10.15 |
| **twitter** | GCN | 54.69 ± 7.31 | 10.71 ± 1.25 | 48.66 ± 9.82 | 17.42 ± 1.83 | 52.82 ± 2.27 | 11.16 ± 1.09 |
| | GAT | 55.12 ± 10.84 | 11.13 ± 1.92 | 50.52 ± 15.38 | 17.91 ± 3.17 | 54.16 ± 4.17 | 11.38 ± 1.64 |
| | GraphSAGE | 81.94 ± 1.66 | 9.91 ± 2.88 | 10.22 ± 3.41 | 9.93 ± 2.83 | 51.62 ± 2.49 | 10.66 ± 1.27 |
| | GIN | 45.76 ± 18.42 | 10.23 ± 1.31 | 56.76 ± 22.52 | 16.82 ± 2.09 | 50.92 ± 2.75 | 10.86 ± 1.44 |
| | HybridGNN | 60.21 ± 12.22 | 11.19 ± 1.77 | 42.59 ± 15.11 | 17.27 ± 2.46 | 54.84 ± 3.21 | 11.75 ± 2.44 |
| **yelpchi** | GCN | 86.77 ± 1.45 | 24.64 ± 2.30 | 76.16 ± 3.62 | 37.14 ± 2.59 | 88.70 ± 1.66 | 45.04 ± 4.02 |
| | GAT | 88.78 ± 1.56 | 29.10 ± 2.87 | 81.28 ± 3.10 | 42.73 ± 3.06 | 91.99 ± 1.12 | 54.38 ± 3.59 |
| | GraphSAGE | 89.32 ± 2.43 | 29.53 ± 4.51 | 72.93 ± 5.97 | 41.61 ± 3.89 | 89.08 ± 1.61 | 47.63 ± 3.52 |
| | GIN | 64.02 ± 3.16 | 11.62 ± 0.90 | 90.92 ± 2.48 | 20.59 ± 1.37 | 87.21 ± 1.13 | 30.13 ± 2.92 |
| | HybridGNN | 87.00 ± 1.77 | 25.25 ± 2.88 | 76.96 ± 2.69 | 37.90 ± 3.12 | 89.09 ± 1.20 | 45.85 ± 2.87 |
| **yelpchi-all** | GCN | 58.33 ± 9.85 | 17.92 ± 2.14 | 49.53 ± 12.21 | 25.60 ± 1.10 | 57.32 ± 1.29 | 21.32 ± 1.49 |
| | GraphSAGE | 70.67 ± 4.68 | 29.82 ± 3.07 | 72.03 ± 6.74 | 41.86 ± 2.04 | 79.02 ± 1.22 | 43.50 ± 2.35 |
| | GIN | 52.88 ± 27.31 | 16.00 ± 3.41 | 47.08 ± 37.36 | 18.62 ± 7.25 | 50.73 ± 1.62 | 14.77 ± 0.70 |
| | HybridGNN | 43.20 ± 24.94 | 17.16 ± 3.86 | 64.68 ± 32.50 | 23.16 ± 5.96 | 55.39 ± 1.96 | 20.00 ± 1.46 |

</details>

<details>
<summary><b>Table 4.3: Full GNN Performance Metrics (Weighted Loss)</b></summary>

| Dataset | Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC-ROC (%) | Avg-Prec (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **amazon** | GCN | 85.69 ± 2.73 | 82.86 ± 3.61 | 84.58 ± 2.87 | 83.88 ± 3.28 | 86.86 ± 2.57 | 85.01 ± 2.39 |
| | GAT | 87.20 ± 2.55 | 83.68 ± 3.37 | 85.89 ± 2.46 | 84.25 ± 3.31 | 87.37 ± 2.04 | 86.28 ± 2.76 |
| | GraphSAGE | 96.53 ± 0.76 | 93.17 ± 1.27 | 94.76 ± 1.44 | 94.54 ± 1.53 | 95.88 ± 1.17 | 95.04 ± 1.09 |
| | GIN | 83.22 ± 3.30 | 81.10 ± 3.76 | 82.83 ± 3.82 | 81.57 ± 3.41 | 84.94 ± 2.75 | 83.07 ± 2.03 |
| | HybridGNN | 95.64 ± 1.77 | 92.14 ± 2.11 | 93.02 ± 2.14 | 92.48 ± 2.14 | 94.73 ± 2.38 | 93.09 ± 2.79 |
| **blogcatalog** | GCN | 84.54 ± 2.02 | 81.93 ± 2.03 | 83.99 ± 2.80 | 82.68 ± 2.57 | 85.48 ± 2.24 | 83.78 ± 2.74 |
| | GAT | 86.59 ± 1.49 | 82.29 ± 2.74 | 84.23 ± 2.19 | 83.37 ± 2.19 | 86.84 ± 2.68 | 84.51 ± 2.61 |
| | GraphSAGE | 94.44 ± 0.76 | 92.76 ± 1.15 | 93.23 ± 1.31 | 93.04 ± 1.32 | 94.48 ± 1.24 | 93.77 ± 1.06 |
| | GIN | 81.33 ± 3.49 | 79.46 ± 3.42 | 81.28 ± 3.49 | 80.08 ± 3.07 | 82.20 ± 2.41 | 80.70 ± 2.35 |
| | HybridGNN | 93.36 ± 1.76 | 91.11 ± 2.71 | 92.87 ± 2.09 | 91.56 ± 2.53 | 93.94 ± 2.74 | 92.51 ± 2.47 |
| **flickr** | GCN | 88.13 ± 2.12 | 85.08 ± 2.45 | 86.82 ± 2.79 | 85.44 ± 2.92 | 88.47 ± 2.26 | 86.85 ± 2.48 |
| | GAT | 89.63 ± 2.05 | 86.50 ± 2.39 | 87.37 ± 2.08 | 86.48 ± 2.81 | 89.16 ± 2.50 | 87.16 ± 2.25 |
| | GraphSAGE | 94.85 ± 1.12 | 92.12 ± 1.23 | 93.15 ± 1.10 | 92.85 ± 1.02 | 94.23 ± 1.91 | 93.12 ± 1.01 |
| | GIN | 85.21 ± 2.34 | 82.45 ± 2.12 | 84.23 ± 2.45 | 83.45 ± 2.23 | 85.12 ± 2.87 | 84.23 ± 2.12 |
| | HybridGNN | 93.37 ± 1.03 | 91.33 ± 1.90 | 92.72 ± 1.22 | 91.80 ± 1.91 | 93.17 ± 2.85 | 92.79 ± 1.38 |
| **reddit** | GCN | 84.06 ± 2.20 | 81.32 ± 2.53 | 82.41 ± 2.27 | 81.10 ± 2.92 | 83.37 ± 2.89 | 82.14 ± 2.06 |
| | GAT | 85.93 ± 2.20 | 82.24 ± 2.10 | 84.15 ± 2.74 | 82.90 ± 2.47 | 85.81 ± 2.33 | 83.54 ± 2.06 |
| | GraphSAGE | 94.93 ± 1.10 | 92.31 ± 1.52 | 93.94 ± 1.16 | 93.07 ± 1.29 | 94.98 ± 1.81 | 93.72 ± 1.41 |
| | GIN | 82.11 ± 2.88 | 80.20 ± 2.70 | 81.99 ± 2.35 | 80.91 ± 2.11 | 82.28 ± 2.46 | 81.12 ± 2.16 |
| | HybridGNN | 93.33 ± 1.81 | 90.33 ± 2.55 | 91.14 ± 2.05 | 90.20 ± 2.98 | 92.25 ± 2.96 | 91.84 ± 2.92 |
| **reddit\_pt** | GCN | 84.59 ± 2.23 | 81.33 ± 2.55 | 82.79 ± 2.06 | 81.12 ± 2.96 | 83.42 ± 2.85 | 82.15 ± 2.06 |
| | GAT | 85.82 ± 2.99 | 82.81 ± 2.95 | 84.85 ± 2.10 | 83.61 ± 2.86 | 85.78 ± 2.54 | 83.56 ± 2.12 |
| | GraphSAGE | 94.29 ± 1.45 | 92.00 ± 1.93 | 93.44 ± 2.03 | 92.65 ± 1.42 | 94.25 ± 1.98 | 93.62 ± 1.40 |
| | GIN | 82.22 ± 2.12 | 80.20 ± 2.85 | 81.55 ± 2.58 | 80.81 ± 2.38 | 82.38 ± 2.13 | 81.00 ± 2.03 |
| | HybridGNN | 93.02 ± 2.43 | 90.35 ± 2.58 | 91.43 ± 2.59 | 90.23 ± 2.04 | 92.40 ± 2.98 | 91.88 ± 1.90 |
| **small\_amazon**| GCN | 85.92 ± 2.60 | 82.29 ± 2.83 | 84.01 ± 2.23 | 83.06 ± 2.40 | 85.61 ± 2.07 | 84.26 ± 2.51 |
| | GAT | 87.08 ± 2.96 | 83.92 ± 2.18 | 85.71 ± 2.49 | 84.56 ± 2.20 | 86.88 ± 2.16 | 85.84 ± 2.10 |
| | GraphSAGE | 94.55 ± 1.60 | 92.77 ± 1.86 | 93.02 ± 1.75 | 92.04 ± 1.61 | 94.50 ± 1.98 | 93.68 ± 1.38 |
| | GIN | 83.14 ± 2.19 | 80.32 ± 2.62 | 82.64 ± 2.42 | 81.33 ± 2.65 | 83.57 ± 2.20 | 82.17 ± 2.26 |
| | HybridGNN | 93.07 ± 2.75 | 90.18 ± 2.37 | 91.70 ± 2.81 | 90.51 ± 2.41 | 92.36 ± 2.26 | 91.23 ± 2.15 |
| **twitter** | GCN | 84.69 ± 2.31 | 81.71 ± 2.25 | 83.66 ± 2.82 | 82.42 ± 2.83 | 84.82 ± 2.27 | 83.16 ± 2.09 |
| | GAT | 85.12 ± 2.84 | 82.13 ± 2.92 | 84.52 ± 2.38 | 83.91 ± 2.17 | 85.16 ± 2.17 | 84.38 ± 2.64 |
| | GraphSAGE | 94.94 ± 1.66 | 92.91 ± 2.88 | 93.22 ± 2.41 | 93.93 ± 2.83 | 94.62 ± 2.49 | 93.66 ± 2.27 |
| | GIN | 82.76 ± 2.42 | 80.23 ± 2.31 | 81.76 ± 2.52 | 80.82 ± 2.09 | 82.92 ± 2.75 | 81.86 ± 2.44 |
| | HybridGNN | 93.21 ± 2.22 | 91.19 ± 2.77 | 92.59 ± 2.11 | 91.27 ± 2.46 | 93.84 ± 2.21 | 92.75 ± 2.44 |
| **yelpchi** | GCN | 86.77 ± 2.45 | 84.64 ± 2.30 | 85.16 ± 2.62 | 84.14 ± 2.59 | 86.70 ± 2.66 | 85.04 ± 2.02 |
| | GAT | 88.78 ± 2.56 | 86.10 ± 2.87 | 87.28 ± 2.10 | 86.73 ± 2.06 | 88.99 ± 2.12 | 87.38 ± 2.59 |
| | GraphSAGE | 94.32 ± 2.43 | 92.53 ± 2.51 | 93.93 ± 2.97 | 93.61 ± 2.89 | 94.08 ± 2.61 | 93.63 ± 2.52 |
| | GIN | 84.02 ± 2.16 | 81.62 ± 2.90 | 82.92 ± 2.48 | 82.59 ± 2.37 | 84.21 ± 2.13 | 83.13 ± 2.92 |
| | HybridGNN | 93.00 ± 2.77 | 90.25 ± 2.88 | 91.96 ± 2.69 | 91.90 ± 2.12 | 93.09 ± 2.20 | 92.85 ± 2.87 |
| **yelpchi-all** | GCN | 84.33 ± 2.85 | 81.92 ± 2.14 | 83.53 ± 2.21 | 82.60 ± 2.10 | 84.32 ± 2.29 | 83.32 ± 2.49 |
| | GraphSAGE | 94.67 ± 1.68 | 92.82 ± 2.07 | 94.03 ± 2.74 | 93.86 ± 2.04 | 94.02 ± 2.22 | 93.50 ± 2.35 |
| | GIN | 82.88 ± 2.31 | 80.00 ± 2.41 | 81.08 ± 2.36 | 80.62 ± 2.25 | 82.73 ± 2.62 | 81.77 ± 2.70 |
| | HybridGNN | 93.20 ± 2.94 | 90.16 ± 2.86 | 91.68 ± 2.50 | 90.16 ± 2.96 | 92.39 ± 2.96 | 91.00 ± 2.46 |

</details>

<details>
<summary><b>Table 5.3: Full Baseline Model Performance Metrics</b></summary>

| Dataset | Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC-ROC (%) | Avg-Prec (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **twitter** | KNN | 88.67 | 8.11 | 1.04 | 1.80 | 48.94 | 9.97 |
| | MLP | 85.78 | 10.94 | 6.28 | 7.96 | 49.81 | 10.32 |
| | RF | 65.45 | 10.26 | 31.97 | 15.49 | 52.11 | 10.80 |
| | SVC | 84.95 | 8.64 | 5.74 | 6.86 | 47.70 | 9.63 |
| **yelpchi** | KNN | 94.78 | 38.05 | 3.85 | 6.99 | 65.05 | 13.02 |
| | MLP | 94.60 | 41.46 | 13.00 | 19.63 | 80.38 | 25.27 |
| | RF | 92.33 | 32.65 | 47.12 | 38.55 | 84.94 | 33.43 |
| | SVC | 79.38 | 14.50 | 62.03 | 23.51 | 78.95 | 21.06 |
| **yelpchi-all** | KNN | 85.57 | 50.77 | 23.28 | 31.91 | 74.31 | 37.18 |
| | MLP | 88.93 | 67.54 | 46.27 | 54.78 | 87.51 | 62.25 |
| | RF | 87.19 | 54.45 | 72.41 | 62.15 | 90.05 | 66.74 |
| | SVC | 76.24 | 35.33 | 76.51 | 48.34 | 84.54 | 52.38 |
| **amazon** | KNN | 98.04 | 93.29 | 76.60 | 84.10 | 89.94 | 78.54 |
| | MLP | 97.74 | 86.64 | 79.49 | 82.77 | 97.45 | 86.45 |
| | RF | 97.29 | 78.66 | 82.00 | 80.28 | 97.76 | 88.88 |
| | SVC | 97.95 | 90.17 | 78.04 | 83.65 | 92.14 | 79.00 |
| **small\_amazon**| KNN | 91.15 | 57.40 | 33.43 | 41.82 | 70.19 | 31.74 |
| | MLP | 90.77 | 53.51 | 38.38 | 44.36 | 66.20 | 32.21 |
| | RF | 90.77 | 54.68 | 41.76 | 46.96 | 72.12 | 42.70 |
| | SVC | 84.90 | 32.54 | 47.77 | 38.05 | 70.77 | 34.72 |
| **flickr** | KNN | 94.59 | 58.39 | 29.43 | 39.07 | 65.78 | 22.00 |
| | MLP | 93.53 | 44.39 | 32.60 | 37.49 | 59.92 | 19.80 |
| | RF | 94.96 | 65.37 | 30.76 | 41.79 | 66.38 | 33.57 |
| | SVC | 93.47 | 43.58 | 34.79 | 38.67 | 66.34 | 28.67 |
| **reddit** | KNN | 96.56 | 39.99 | 3.33 | 6.07 | 56.66 | 7.23 |
| | MLP | 96.67 | 0.00 | 0.00 | 0.00 | 58.49 | 4.63 |
| | RF | 92.02 | 8.64 | 14.44 | 10.62 | 68.42 | 6.98 |
| | SVC | 79.52 | 6.97 | 41.83 | 11.92 | 66.47 | 6.87 |
| **blogcatalog** | KNN | 94.25 | 50.39 | 25.76 | 33.78 | 64.10 | 19.14 |
| | MLP | 93.65 | 42.82 | 29.59 | 34.71 | 59.84 | 18.38 |
| | RF | 94.59 | 55.29 | 33.24 | 41.28 | 64.30 | 30.35 |
| | SVC | 93.69 | 44.10 | 34.67 | 38.49 | 63.91 | 27.35 |
| **reddit\_pt** | KNN | 95.12 | 41.23 | 3.50 | 6.58 | 57.32 | 8.12 |
| | MLP | 96.24 | 0.00 | 0.00 | 0.00 | 59.48 | 5.23 |
| | RF | 93.48 | 9.42 | 15.02 | 11.48 | 69.01 | 7.25 |
| | SVC | 81.64 | 7.23 | 42.67 | 12.18 | 67.53 | 7.13 |

</details>

<details>
<summary><b>Table 4.4: KNN Distance Metric Comparison</b></summary>

| Dataset | Distance Metric | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC-ROC (%) | Avg-Prec (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **twitter** | Euclidean | 89.09 | 11.58 | 1.04 | 1.86 | 49.39 | 9.88 |
| | Manhattan | 89.09 | 11.58 | 1.04 | 1.86 | 49.39 | 9.88 |
| **yelpchi** | Euclidean | 94.79 | 38.49 | 3.78 | 6.87 | 64.75 | 10.65 |
| | Manhattan | 94.98 | 56.52 | 6.43 | 11.52 | 69.07 | 16.00 |
| **yelpchi-all**| Euclidean | 85.57 | 50.71 | 23.10 | 31.73 | 73.88 | 33.13 |
| | Manhattan | 87.38 | 64.09 | 29.82 | 40.69 | 78.48 | 42.49 |
| **amazon** | Euclidean | 98.02 | 93.27 | 76.32 | 83.93 | 89.91 | 77.64 |
| | Manhattan | 97.96 | 93.59 | 74.75 | 83.10 | 90.72 | 78.60 |
| **small\_amazon**| Euclidean | 91.80 | 66.96 | 29.93 | 40.99 | 71.46 | 36.58 |
| | Manhattan | 91.80 | 66.96 | 29.93 | 40.99 | 71.46 | 36.58 |
| **flickr** | Euclidean | 95.02 | 78.95 | 21.24 | 33.44 | 66.37 | 28.21 |
| | Manhattan | 95.02 | 78.95 | 21.24 | 33.44 | 66.29 | 27.67 |
| **reddit** | Euclidean | 96.67 | 52.67 | 2.74 | 5.19 | 56.47 | 6.13 |
| | Manhattan | 96.64 | 43.33 | 2.37 | 4.48 | 58.44 | 5.89 |
| **blogcatalog**| Euclidean | 94.67 | 68.61 | 14.95 | 24.17 | 66.70 | 25.18 |
| | Manhattan | 94.69 | 73.14 | 13.83 | 22.91 | 66.58 | 24.70 |

</details>

<details>
<summary><b>Tables 4.1 & 5.4: Computational Complexity Analysis</b></summary>

**GNN Models**
| Model | Training Time | Inference Time | Training Space | Inference Space |
| :--- | :--- | :--- | :--- | :--- |
| **GCN** | `O(K|E|d + Knd²)` | `O(K|E|d + Knd²)` | `O(nd + |E|)` | `O(nd + |E|)` |
| **GIN** | `O(K|E|d + Knd²)` | `O(K|E|d + Knd²)` | `O(nd + |E|)` | `O(nd + |E|)` |
| **GAT** | `O(KH|E|d + Knd²)` | `O(KH|E|d + Knd²)` | `O(H|E| + nd)` | `O(H|E| + nd)` |
| **GraphSAGE** | `O(Knd²)` | `O(Knd²)` | `O(nd)` | `O(nd)` |
| **HybridGNN** | `O(K(|E|+H|E|)d + Knd²)`| `O(K(|E|+H|E|)d + Knd²)`| `O(H|E| + nd)`| `O(H|E| + nd)`|

**Baseline Models**
| Model | Training Time | Prediction Time | Auxiliary Space |
| :--- | :--- | :--- | :--- |
| **KNN** | `O(1)` | `O(nd)` | `O(n)` |
| **MLP** | `O(ndhe)` | `O(dh)` | `O(dh)` |
| **Random Forest**| `O(tn log n)` | `O(t log n)` | `O(tn)` |
| **SVC** | `O(n²d)` to `O(n³)` | `O(sd)` | `O(n)` |

</details>

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

👨‍💻 Author

**Shen-Han Chiu** - [LinkedIn Profile](https://www.linkedin.com/in/chiushenhan/)

📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
