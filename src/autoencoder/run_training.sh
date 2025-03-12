#!/bin/bash

# This script runs the autoencoder training with specified parameters

# Activate your virtual environment if needed
# source venv/bin/activate

# Set data path - CHANGE THIS TO YOUR ACTUAL DATA PATH
DATA_PATH="path/to/your/data.txt"

# Run the training script with parameters
python train.py \
    --data_path $DATA_PATH \
    --batch_size 32 \
    --epochs 200 \
    --latent_dim 32 \
    --learning_rate 0.001 \
    --test_split 0.1

echo "Training complete!"

# Note: You can adjust these parameters as needed
# --data_path: Path to your data file (.txt or .npy)
# --batch_size: Number of samples per batch
# --epochs: Number of training epochs
# --latent_dim: Dimension of the latent space (compression level)
# --learning_rate: Learning rate for the optimizer 