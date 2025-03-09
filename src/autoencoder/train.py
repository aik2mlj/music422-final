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
# TODO: import the data

# train the model


def train(
    model, train_loader, val_loader, criterion=None, optimizer=None, num_epochs=100, device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion is None:
        criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    os.makedirs("checkpoints", exist_ok=True)

    model.to(device)

    train_losses = []
    val_losses = []

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
            {"model": model.state_dict(), "train_loss": avg_train_loss, "val_loss": avg_val_loss},
            "best_model.pth",
        )

    # save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(
            {"model": model.state_dict(), "train_loss": avg_train_loss, "val_loss": avg_val_loss},
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
