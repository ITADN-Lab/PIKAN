# PIKAN
PIKAN
Power Flow Neural Network
A PIKAN for power flow analysis using deep learning with physical constraints.

# Overview
This project implements a neural network model that combines traditional power flow analysis with deep learning. The model predicts voltage magnitudes and angles while enforcing physical constraints through a custom loss function that incorporates power balance equations.

# Requirements
Python Packages

Python 3.8+

PyTorch 1.9.0+

NumPy 1.20.0+

Matplotlib 3.3.0+

Pandapower 2.7.0+

SQLite3 (included in Python standard library)

Optional Dependencies

KAN library (required if using KAN model architecture)


# Project Structure
project/
├── main.py              # Main entry point

├── config.py            # Configuration parameters

├── data_loader.py       # Data loading and preprocessing

├── models.py            # Neural network model definitions

├── physics.py           # Physical constraint calculations

├── train.py             # Training and evaluation logic

└── utils.py             # Utility functions and visualization


# Usage
Running the Model

Execute the main script to start training:

python main.py

# Model selection
network = 'kan'  # 'kan' or 'mlp'

optimizer_type = 'adam'  # 'sgd', 'adam', 'adamW'


# Loss weights
alpha1, alpha2, alpha3 = 10, 8, 6

Data Preparation

The model expects a SQLite database file at ./case30_samples_PQ_wsl.dbwith the following structure:

Table: samples

Columns: labeland feature columns (50 input features + 60 voltage features)

Output

The training process generates:

Model Checkpoints: Saved PyTorch models

Training Curves: Loss and metric plots

Text Files: Detailed metrics and error logs

Normalization Parameters: Pickle file for data normalization

Key Files

main.py: Orchestrates the entire training pipeline

config.py: Centralized configuration management

data_loader.py: Handles data loading and preprocessing

models.py: Defines neural network architectures

physics.py: Implements physical constraint calculations

train.py: Contains training and evaluation logic

utils.py: Provides utility functions and visualization

Model Architectures

MLP Model

Multi-layer perceptron with separate branches for voltage magnitude and angle prediction

Dropout layers for regularization

ReLU activation functions

KAN Model

Kolmogorov-Arnold Network implementation

Adaptive grid size and spline order

Potentially better performance for complex physical systems

Citation

If you use this code in your research, please cite the relevant papers for the KAN architecture and physics-informed neural networks.
License

This project is for research purposes. Please ensure you have the proper licenses for any commercial use.

