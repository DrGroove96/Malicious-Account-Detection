#!/usr/bin/env python3
"""
Setup script for GNN Malicious Account Detection package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gnn-malicious-detection",
    version="1.0.0",
    author="Shen-Han Chiu",
    author_email="your-email@example.com",
    description="Graph Neural Networks for Malicious Account Detection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gnn-malicious-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "gnn-train=train:main",
            "gnn-baseline=baseline_experiments:main",
            "gnn-viz=visualizations:main",
            "gnn-curve=curve:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    keywords="graph neural networks, malicious detection, social networks, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/gnn-malicious-detection/issues",
        "Source": "https://github.com/yourusername/gnn-malicious-detection",
        "Documentation": "https://github.com/yourusername/gnn-malicious-detection#readme",
    },
) 