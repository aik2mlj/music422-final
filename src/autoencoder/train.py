import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# import the model
from model import AE

# import the data
from dataloader import get_data_loaders

# train the model


def train(
    model, train_loader, val_loader, criterion=None, optimizer=None, num_epochs=100, device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion is None:
        criterion = nn.MSELoss()  # Default: Mean Squared Error loss
        # Alternative loss functions:
        # criterion = nn.L1Loss()  # Mean Absolute Error - can produce sharper reconstructions
        # criterion = nn.SmoothL1Loss()  # Smooth L1 Loss - less sensitive to outliers than MSE
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    os.makedirs("checkpoints", exist_ok=True)

    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")  # Initialize with infinity

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)  # check for number of batches

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            output = model(data)
            loss = criterion(output, data)
            # Alternative loss calculation with L1Loss (commented out):
            # loss = nn.L1Loss()(output, data)  # Direct instantiation in-place

            # backpropagate
            loss.backward()
            optimizer.step()

            # update the train loss
            train_loss += loss.item()

            # print the progress
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx + 1}/{num_batches}] Loss: {loss.item():.4f}"
                )

        # calculate the average train loss
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)

        # validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_loader):
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data)
                # Alternative loss calculation with L1Loss (commented out):
                # loss = nn.L1Loss()(output, data)
                val_loss += loss.item()

        # calculate the average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # print the progress
        epoch_end_time = time.time() - epoch_start_time
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} Time: {epoch_end_time:.2f}s"
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

    # plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig("losses.png")

    # save the training and validation losses
    np.save("train_losses.npy", train_losses)
    np.save("val_losses.npy", val_losses)

    return train_losses, val_losses


def evaluate(model, test_loader, criterion=None, device=None):
    """
    Evaluate the model on a test dataset
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion is None:
        criterion = nn.MSELoss()
        # Alternative: Use L1Loss for evaluation
        # criterion = nn.L1Loss()

    model.to(device)
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data)
            # Alternative loss calculation:
            # loss = nn.L1Loss()(output, data)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Visualize a few reconstructions
    data, _ = next(iter(test_loader))
    data = data[:5].to(device)  # Get first 5 samples

    model.eval()
    with torch.no_grad():
        output = model(data)

    # Convert back to CPU for plotting
    data = data.cpu().numpy()
    output = output.cpu().numpy()

    # Plot original vs reconstructed
    plt.figure(figsize=(12, 6))
    for i in range(5):
        # Original - Plot first 20 dimensions for visibility
        plt.subplot(2, 5, i + 1)
        plt.plot(data[i, :20])
        plt.title(f"Original {i + 1}")

        # Reconstruction - Plot first 20 dimensions for visibility
        plt.subplot(2, 5, i + 6)
        plt.plot(output[i, :20])
        plt.title(f"Reconstructed {i + 1}")

    plt.tight_layout()
    plt.savefig("test_reconstructions.png")

    return avg_test_loss


def main():
    """
    Main function to parse arguments and run the training
    """
    parser = argparse.ArgumentParser(description="Train an autoencoder")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to data file (.txt or .npy)"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=32, help="Dimension of latent space")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test set split ratio")
    # Commented out loss function choice
    # parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1"],
    #                     help="Loss function: 'mse' for Mean Squared Error, 'l1' for Mean Absolute Error")
    args = parser.parse_args()

    # Get train and validation data loaders
    train_loader, val_loader = get_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        val_split=0.2,  # Use 20% of data for validation
        num_workers=4,
        normalize=True,
    )

    # Get input dimension from a sample batch
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]  # Should be 512 for your vectors
    print(f"Input dimension detected: {input_dim}")

    # Create the model
    model = AE(input_dim=input_dim, latent_dim=args.latent_dim)
    print(f"Created autoencoder with latent dimension: {args.latent_dim}")

    # Set up criterion and optimizer
    criterion = nn.MSELoss()
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
    best_model = AE(input_dim=input_dim, latent_dim=args.latent_dim)
    best_model.load_state_dict(torch.load("best_model.pth")["model"])

    # Evaluate on validation set
    print("Evaluating best model on validation set...")
    evaluate(best_model, val_loader, criterion)

    print("Training complete!")
    print(f"Best model saved to best_model.pth")
    print(f"Loss plot saved to losses.png")


if __name__ == "__main__":
    main()
