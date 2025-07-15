#!/bin/bash

# Script to run all GNN experiments for malicious account detection
# This script will run both weighted and unweighted loss experiments,
# baseline ML experiments, and generate all visualizations

set -e  # Exit on any error

echo "=========================================="
echo "GNN Malicious Account Detection Experiments"
echo "=========================================="

# Create necessary directories
mkdir -p data results figures

# Check if data files exist
echo "Checking for dataset files..."
if [ ! -d "data" ] || [ -z "$(ls -A data 2>/dev/null)" ]; then
    echo "Warning: No dataset files found in data/ directory."
    echo "Please place your .mat dataset files in the data/ directory."
    echo "Required files: Amazon.mat, BlogCatalog.mat, Flickr.mat, Reddit.mat, small_amazon.mat, twitter.mat, YelpChi.mat, YelpChi-all.mat"
    echo ""
fi

# Install dependencies if needed
echo "Installing dependencies..."
pip install -r requirements.txt

# Run GNN experiments with weighted loss
echo ""
echo "=========================================="
echo "Running GNN experiments with weighted loss"
echo "=========================================="
python train.py --weighted --runs 5 --epochs 200 --save-training-curves

# Run GNN experiments with unweighted loss
echo ""
echo "=========================================="
echo "Running GNN experiments with unweighted loss"
echo "=========================================="
python train.py --unweighted --runs 5 --epochs 200 --save-training-curves

# Run baseline ML experiments
echo ""
echo "=========================================="
echo "Running baseline ML experiments"
echo "=========================================="
python baseline_experiments.py --runs 5 --save-plots

# Generate convergence analysis
echo ""
echo "=========================================="
echo "Generating convergence analysis"
echo "=========================================="
python curve.py

# Generate all visualizations
echo ""
echo "=========================================="
echo "Generating all visualizations"
echo "=========================================="
python visualizations.py --dpi 300 --format png

echo ""
echo "=========================================="
echo "Experiments completed successfully!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "- results/gnn_results_weighted.json"
echo "- results/gnn_results_unweighted.json"
echo "- results/baseline_results.json"
echo "- results/gnn_results_weighted.csv"
echo "- results/gnn_results_unweighted.csv"
echo "- results/baseline_results.csv"
echo ""
echo "Figures saved in:"
echo "- figures/roc_curves/"
echo "- figures/confusion_matrices/"
echo "- figures/loss_curves/"
echo "- figures/convergence_plots/"
echo "- figures/comparison_plots/"
echo "- figures/feature_importance/"
echo ""
echo "To view results, check the CSV files in the results/ directory."
echo "To view figures, check the PNG files in the figures/ directory." 