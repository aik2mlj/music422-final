import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

# import model
from model import AE, ConvAE, ConvAE_VQ

# import data
from dataloader import get_data_loaders


def train(
    model,
    train_loader,
    val_loader,
    criterion=None,
    optimizer=None,
    num_epochs=100,
    device=None,
    use_vq=False,
    beta=0.25,
):
    if device is None:
        # Check for MPS (Mac GPU) first, then CUDA, then fall back to CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU) for training")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA (NVIDIA GPU) for training")
        else:
            device = torch.device("cpu")
            print("Using CPU for training (this will be slower)")

    if criterion is None:
        criterion = nn.MSELoss()  # Default reconstruction loss

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir="runs/autoencoder")
    print("TensorBoard logs will be saved to 'runs/autoencoder'")
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    epoch_bar = trange(num_epochs, desc="Epochs", position=0)
    for epoch in epoch_bar:
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        batch_bar = tqdm(
            train_loader,
            desc=f"Training (Epoch {epoch + 1}/{num_epochs})",
            leave=False,
            position=1,
            total=num_batches,
        )

        for batch_idx, (data, _) in enumerate(batch_bar):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass:
            if use_vq:
                # For VQ models, the forward pass returns (recon, vq_loss, perplexity)
                recon, vq_loss, perplexity = model(data)
                recon_loss = criterion(recon, data)

                # Apply weight to VQ loss on top of reconstruction loss
                total_loss = recon_loss + beta * vq_loss

                # print loss for first batch each epoch
                if batch_idx == 0:
                    # Calculate diversity metrics for logging
                    target_perplexity = model.vq_layer.num_embeddings * 0.1
                    perplexity_ratio = min(10.0, target_perplexity / (perplexity.item() + 1e-10))
                    diversity_loss = perplexity_ratio * 0.1

                    print(
                        f"Epoch {epoch + 1}, Batch {batch_idx}: "
                        f"Recon Loss: {recon_loss.item():.6f}, "
                        f"VQ Loss: {vq_loss.item():.6f}, "
                        f"Perplexity: {perplexity.item():.2f}, "
                        f"Target: {target_perplexity:.1f}, "
                        f"Diversity Loss: {diversity_loss:.6f}"
                    )

                # Add individual losses to tensorboard
                writer.add_scalar(
                    "Loss/recon_train", recon_loss.item(), epoch * len(train_loader) + batch_idx
                )
                writer.add_scalar(
                    "Loss/vq_train", vq_loss.item(), epoch * len(train_loader) + batch_idx
                )
                writer.add_scalar(
                    "Metrics/perplexity_train",
                    perplexity.item(),
                    epoch * len(train_loader) + batch_idx,
                )

                # Calculate and log diversity metrics
                target_perplexity = model.vq_layer.num_embeddings * 0.1
                perplexity_ratio = min(10.0, target_perplexity / (perplexity.item() + 1e-10))
                diversity_loss = perplexity_ratio * 0.1
                writer.add_scalar(
                    "Loss/diversity_train", diversity_loss, epoch * len(train_loader) + batch_idx
                )
                writer.add_scalar(
                    "Metrics/perplexity_ratio",
                    perplexity_ratio,
                    epoch * len(train_loader) + batch_idx,
                )
            else:
                # For standard AE/ConvAE return only reconstruction
                recon = model(data)
                total_loss = criterion(recon, data)

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            batch_bar.set_postfix(loss=f"{total_loss.item():.4f}")

        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation loop:
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(
            val_loader,
            desc=f"Validation (Epoch {epoch + 1}/{num_epochs})",
            leave=False,
            position=1,
            total=len(val_loader),
        )
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_bar):
                data = data.to(device)
                if use_vq:
                    recon, vq_loss, perplexity = model(data)
                    recon_loss = criterion(recon, data)
                    loss_val = recon_loss + beta * vq_loss

                    # Log validation metrics for the first batch
                    if batch_idx == 0:
                        # Calculate diversity metrics
                        target_perplexity = model.vq_layer.num_embeddings * 0.1
                        perplexity_ratio = min(
                            10.0, target_perplexity / (perplexity.item() + 1e-10)
                        )
                        diversity_loss = perplexity_ratio * 0.1

                        writer.add_scalar("Loss/recon_val", recon_loss.item(), epoch)
                        writer.add_scalar("Loss/vq_val", vq_loss.item(), epoch)
                        writer.add_scalar("Metrics/perplexity_val", perplexity.item(), epoch)
                        writer.add_scalar("Loss/diversity_val", diversity_loss, epoch)
                        writer.add_scalar("Metrics/perplexity_ratio_val", perplexity_ratio, epoch)
                else:
                    recon = model(data)
                    loss_val = criterion(recon, data)
                val_loss += loss_val.item()
                val_bar.set_postfix(loss=f"{loss_val.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(f"Parameters/{name}", param, epoch)

        epoch_end_time = time.time() - epoch_start_time
        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}",
            val_loss=f"{avg_val_loss:.4f}",
            time=f"{epoch_end_time:.2f}s",
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                "best_model.pth",
            )
            epoch_bar.set_postfix(
                train_loss=f"{avg_train_loss:.4f}",
                val_loss=f"{avg_val_loss:.4f} (best)",
                time=f"{epoch_end_time:.2f}s",
            )

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                f"checkpoints/model_epoch_{epoch + 1}.pth",
            )

    writer.close()
    np.save("train_losses.npy", train_losses)
    np.save("val_losses.npy", val_losses)
    print("Training complete. Use 'tensorboard --logdir=runs' to visualize training curves.")
    return train_losses, val_losses


def evaluate(model, test_loader, criterion=None, device=None, use_vq=False, beta=0.25):
    if device is None:
        # Check for MPS (Mac GPU) first, then CUDA, then fall back to CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if criterion is None:
        criterion = nn.MSELoss()

    model.to(device)
    model.eval()
    test_loss = 0.0
    recon_loss_total = 0.0
    vq_loss_total = 0.0
    test_bar = tqdm(test_loader, desc="Testing", total=len(test_loader))
    with torch.no_grad():
        for data, _ in test_bar:
            data = data.to(device)
            if use_vq:
                recon, vq_loss, perplexity = model(data)
                recon_loss = criterion(recon, data)
                loss = recon_loss + beta * vq_loss
                recon_loss_total += recon_loss.item()
                vq_loss_total += vq_loss.item()
            else:
                recon = model(data)
                loss = criterion(recon, data)
            test_loss += loss.item()
            test_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    if use_vq:
        avg_recon_loss = recon_loss_total / len(test_loader)
        avg_vq_loss = vq_loss_total / len(test_loader)
        print(f"Reconstruction Loss: {avg_recon_loss:.4f}, VQ Loss: {avg_vq_loss:.4f}")

    return avg_test_loss


def main():
    parser = argparse.ArgumentParser(description="Train an autoencoder")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/dump.npy",
        help="Path to data file (.txt or .npy)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=32, help="Dimension of latent space")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test set split ratio")
    parser.add_argument(
        "--model_type",
        type=str,
        default="linear",
        choices=["linear", "conv", "vq"],
        help="Model type: 'linear' for standard AE, 'conv' for ConvAE, 'vq' for ConvAE_VQ",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.25,
        help="Weight for the VQ loss component (only used if model_type is 'vq')",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    train_loader, val_loader = get_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        val_split=0.2,
        num_workers=0,
        normalize=True,
    )
    print("Data loaded successfully! Ready for training.")

    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]
    print(f"Input dimension detected: {input_dim}")

    # For ConvAE, check if input_dim is divisible by 8
    if args.model_type in ["conv", "vq"] and input_dim % 8 != 0:
        print(
            f"WARNING: Input dimension {input_dim} is not divisible by 8, which may cause issues with convolutional models."
        )

    # Instantiate the model based on selected model type
    if args.model_type == "conv":
        model = ConvAE(input_dim=input_dim, latent_dim=args.latent_dim)
        use_vq = False
        print(f"Created convolutional autoencoder with latent dimension: {args.latent_dim}")
    elif args.model_type == "vq":
        model = ConvAE_VQ(input_dim=input_dim, latent_dim=args.latent_dim, num_embeddings=1024)
        use_vq = True
        print(f"Created VQ convolutional autoencoder with latent dimension: {args.latent_dim}")
    else:
        model = AE(input_dim=input_dim, latent_dim=args.latent_dim)
        use_vq = False
        print(f"Created linear autoencoder with latent dimension: {args.latent_dim}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Choose device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "mps (Apple Silicon GPU)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "cuda (NVIDIA GPU)"
    else:
        device = torch.device("cpu")
        device_name = "cpu"

    print("Starting training with:")
    print(f"  - Model type: {args.model_type}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Device: {device_name}")

    train_losses, val_losses = train(
        model,
        train_loader,
        val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        use_vq=use_vq,
        beta=args.beta,
        device=device,
    )

    # Load best model for evaluation
    if args.model_type == "conv":
        best_model = ConvAE(input_dim=input_dim, latent_dim=args.latent_dim)
    elif args.model_type == "vq":
        best_model = ConvAE_VQ(input_dim=input_dim, latent_dim=args.latent_dim, num_embeddings=1024)
    else:
        best_model = AE(input_dim=input_dim, latent_dim=args.latent_dim)

    best_model.load_state_dict(torch.load("best_model.pth")["model"])
    print("Evaluating best model on validation set...")
    evaluate(
        best_model, val_loader, criterion=criterion, use_vq=use_vq, beta=args.beta, device=device
    )

    print("Training complete!")
    print("Best model saved to best_model.pth")
    print("To view training curves, run: tensorboard --logdir=runs")


if __name__ == "__main__":
    main()
