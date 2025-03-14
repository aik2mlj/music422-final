import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

# import the model
# from model import AE
from vqvae import VQVAE

# import the data
from dataloader import get_data_loaders

# train the model


def train(model, train_loader, val_loader, criterion, optimizer=None, num_epochs=100, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    os.makedirs("checkpoints", exist_ok=True)

    # Create tensorboard writer
    writer = SummaryWriter(log_dir="runs/autoencoder")
    print(f"TensorBoard logs will be saved to 'runs/autoencoder'")

    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")  # Initialize with infinity

    # Create epoch progress bar
    epoch_bar = trange(num_epochs, desc="Epochs", position=0)

    for epoch in epoch_bar:
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)  # check for number of batches

        # Create batch progress bar
        batch_bar = tqdm(
            train_loader,
            desc=f"Training (Epoch {epoch + 1}/{num_epochs})",
            leave=False,
            position=1,
            total=num_batches,
        )

        for batch_idx, (data, _) in enumerate(batch_bar):
            data = data.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            output = model(data)
            loss, loss_dict = criterion(output, data)

            # backpropagate
            loss.backward()
            optimizer.step()

            # update the train loss
            train_loss += loss.item()
            for key, value in loss_dict.items():
                writer.add_scalar(
                    f"Loss/train/{key}", value.item(), epoch * len(train_loader) + batch_idx
                )

            # Update batch progress bar
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        # calculate the average train loss
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)

        # validate the model
        model.eval()
        val_loss = 0.0

        # Create validation progress bar
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
                output = model(data)
                loss, loss_dict = criterion(output, data)
                # Alternative loss calculation with L1Loss (commented out):
                # loss = nn.L1Loss()(output, data)
                val_loss += loss.item()
                for key, value in loss_dict.items():
                    writer.add_scalar(
                        f"Loss/validation/{key}",
                        value.item(),
                        epoch * len(train_loader) + batch_idx,
                    )

                # Update validation progress bar
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        # calculate the average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train/loss", avg_train_loss, epoch)
        writer.add_scalar("Loss/validation/loss", avg_val_loss, epoch)

        # Log model parameters histograms (optional)
        for name, param in model.named_parameters():
            writer.add_histogram(f"Parameters/{name}", param, epoch)

        # print the progress
        epoch_end_time = time.time() - epoch_start_time

        # Update epoch progress bar
        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}",
            val_loss=f"{avg_val_loss:.4f}",
            time=f"{epoch_end_time:.2f}s",
        )

        # save the model if it has the lowest validation loss
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

        # save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                f"checkpoints/model_epoch_{epoch + 1}.pth",
            )

    # Close the TensorBoard writer
    writer.close()

    # save the training and validation losses
    np.save("train_losses.npy", train_losses)
    np.save("val_losses.npy", val_losses)

    print("Training complete. Use 'tensorboard --logdir=runs' to visualize training curves.")

    return train_losses, val_losses


def evaluate(model, test_loader, criterion, device=None):
    """
    Evaluate the model on a test dataset
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    test_loss = 0.0

    # Create test progress bar
    test_bar = tqdm(test_loader, desc="Testing", total=len(test_loader))

    with torch.no_grad():
        for data, _ in test_bar:
            data = data.to(device)
            output = model(data)
            loss, loss_dict = criterion(output, data)
            # Alternative loss calculation:
            # loss = nn.L1Loss()(output, data)
            test_loss += loss.item()

            # Update test progress bar
            test_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Visualize a few reconstructions
    # data, _ = next(iter(test_loader))
    # data = data[:5].to(device)  # Get first 5 samples

    # model.eval()
    # with torch.no_grad():
    #     output = model(data)
    #
    # # Convert back to CPU for plotting
    # data = data.cpu().numpy()
    # output = output.cpu().numpy()

    return avg_test_loss


def main():
    """
    Main function to parse arguments and run the training
    """
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
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test set split ratio")
    # Commented out loss function choice
    # parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1"],
    #                     help="Loss function: 'mse' for Mean Squared Error, 'l1' for Mean Absolute Error")
    args = parser.parse_args()

    # Get train and validation data loaders
    print(f"Loading data from {args.data_path}...")
    train_loader, val_loader = get_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        val_split=0.2,  # Use 20% of data for validation
        num_workers=0,
        normalize=True,
    )
    print(f"Data loaded successfully! Ready for training.")

    # Get input dimension from a sample batch
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]  # Should be 512 for your vectors
    print(f"Input dimension detected: {input_dim}")

    # Create the model
    model = VQVAE()
    # print(f"Created autoencoder with latent dimension: {args.latent_dim}")

    # Set up criterion and optimizer
    def criterion(output, target):
        recon, vq_loss, perplexity = output
        recon_loss = nn.MSELoss()(recon, target)
        loss = recon_loss + vq_loss
        return loss, {
            "vq_loss": vq_loss,
            "recon_loss": recon_loss,
        }

    # Alternative loss function based on argument (commented out)
    # if args.loss == "l1":
    #     criterion = nn.L1Loss()
    #     print("Using L1Loss (Mean Absolute Error)")
    # else:
    #     criterion = nn.MSELoss()
    #     print("Using MSELoss (Mean Squared Error)")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Print training configuration
    print(f"Starting training with:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Train the model
    train_losses, val_losses = train(
        model,
        train_loader,
        val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
    )

    # Load the best model for evaluation
    best_model = VQVAE()
    best_model.load_state_dict(torch.load("best_model.pth")["model"])

    # Evaluate on validation set
    print("Evaluating best model on validation set...")
    evaluate(best_model, val_loader, criterion)

    print("Training complete!")
    print(f"Best model saved to best_model.pth")
    print(f"To view training curves, run: tensorboard --logdir=runs")


if __name__ == "__main__":
    main()
