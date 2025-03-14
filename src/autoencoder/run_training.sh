#!/bin/bash

python train.py \
    --batch_size 64 \
    --epochs 200 \
    --latent_dim 64 \
    --learning_rate 0.0005 \
    --model_type conv \
    # --beta 0.25  # Weight for the VQ loss component

echo "Training complete!"

# --data_path: Path to data file (.txt or .npy)
# --batch_size: # samples per batch
# --epochs: # epochs
# --latent_dim: Dimension of the latent space (compression level) (could derermine based on bit-allocation)
# --learning_rate: Learning rate for optimizer
# --model_type: Type of autoencoder model ("linear", "conv", or "vq") 
# --beta: Weight for the VQ losses (mse + commitment loss) (only used if model_type is 'vq') 