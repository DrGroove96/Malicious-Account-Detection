# Malicious Account Detection in Social Networks via Graph Neural Networks

This repository contains the experimental code for the thesis "Malicious Accounts Detection in Social Networks via Graph Neural Networks" by Shen-Han Chiu, submitted to the Schreyer Honors College at The Pennsylvania State University.

## Introduction

This project investigates the effectiveness of Graph Neural Networks (GNNs) for detecting malicious accounts on social media platforms. The proliferation of social scam bots and other malicious activities poses a significant threat to online communities, making robust detection methods crucial. This work compares the performance of various GNN architectures against traditional machine learning models on this task.

## Features

*   **GNN Models:** Implementation and evaluation of five different GNN architectures:
    *   Graph Convolutional Network (GCN)
    *   Graph Attention Network (GAT)
    *   GraphSAGE
    *   Graph Isomorphism Network (GIN)
    *   HybridGNN
*   **Traditional Machine Learning Models:** For baseline comparison, the following four models are implemented:
    *   K-Nearest Neighbors (KNN)
    *   Multilayer Perceptron (MLP)
    *   Random Forest (RF)
    *   Support Vector Classifier (SVC)
*   **Datasets:** The experiments are conducted on nine real-world social network datasets, providing a comprehensive evaluation.
*   **Performance Analysis:** The models are evaluated using a range of metrics, including accuracy, precision, recall, F1-score, and AUC-ROC.
*   **Data Visualization:** The repository includes code to generate visualizations such as ROC curves, loss plots, and confusion matrices to interpret the model performance.

## Methodology

The core of this research revolves around a comparative analysis of two primary paradigms for malicious account detection:

1.  **Graph Neural Networks (GNNs):** These models leverage the inherent graph structure of social networks, where users are nodes and connections are edges. GNNs can capture complex relational patterns that are often missed by traditional methods.
2.  **Traditional Machine Learning:** These models typically rely on feature-based analysis of user profiles and activities, without explicitly considering the network structure.

A key aspect of the methodology is the use of a weighted loss function to address the class imbalance problem, which is common in malicious account detection tasks where malicious accounts are a minority.

## Datasets

The following nine publicly available datasets were used for the experiments:

*   Amazon
*   BlogCatalog
*   Flickr
*   Reddit
*   Reddit\_pt
*   Small\_Amazon
*   Twitter
*   YelpChi
*   YelpChi-All

These datasets originate from the paper "A comprehensive survey on graph anomaly detection with deep learning" by Ma et al. (2021).

## Installation

To run the experiments, you need to have Python and the following libraries installed. You can install them using pip:

```bash
pip install torch torch-geometric scikit-learn pandas matplotlib

## Usage

1.  Clone the repository:
    ```bash
    git clone https://github.com/DrGroove96/Malicious-Account-Detection.git
    cd Malicious-Account-Detection
    ```
2.  Place the datasets in the appropriate directory.
3.  Run the experiment scripts. For example, to train and evaluate the GraphSAGE model:
    ```bash
    python train_gnn.py --model=GraphSAGE --dataset=Amazon
    ```
    To train and evaluate a baseline model like KNN:
    ```bash
    python train_baseline.py --model=KNN --dataset=Amazon
    ```

## Results

The experiments demonstrate that GNNs, particularly **GraphSAGE**, consistently outperform traditional machine learning models in detecting malicious accounts across various datasets. The ability of GNNs to leverage both node features and the underlying graph structure proves to be a significant advantage.

Key findings include:

*   **GraphSAGE** achieves state-of-the-art performance on most of the benchmark datasets.
*   The use of a weighted loss function significantly improves the performance of the models in handling imbalanced data.
*   GNN models generally show higher accuracy, precision, and F1-scores compared to baseline models.

For detailed results, including performance tables and visualizations, please refer to the thesis document.

## Citation

If you use the code or findings from this project in your research, please cite the following thesis:
Chiu, S.-H. (2025). Malicious Accounts Detection in Social Networks via Graph Neural Networks (Unpublished honors thesis). The Pennsylvania State University, University Park, PA.


## Author

*   **SHEN-HAN CHIU** - [LinkedIn](https://www.linkedin.com/in/chiushenhan/)

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
