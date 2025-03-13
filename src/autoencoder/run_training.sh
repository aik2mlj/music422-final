#!/bin/bash

# This script runs the autoencoder training with specified parameters

# Activate your virtual environment if needed
# source venv/bin/activate

# Run the training script with parameters
python train.py \
    --batch_size 64 \
    --epochs 100 \
    --latent_dim 256 \
    --learning_rate 0.001

echo "Training complete!"

# Note: You can adjust these parameters as needed
# --data_path: Path to your data file (.txt or .npy)
# --batch_size: Number of samples per batch
# --epochs: Number of training epochs
# --latent_dim: Dimension of the latent space (compression level)
# --learning_rate: Learning rate for the optimizer 