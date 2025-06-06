import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class AutoencoderDataset(Dataset):
    """
    Custom dataset for autoencoder training
    """

    def __init__(self, data_path, transform=None, normalize=True):
        """
        Args:
            data_path (str): Path to the data file
            transform (callable, optional): Optional transform to apply to data
            normalize (bool): Whether to normalize the data
        """
        # Load data from a text file with 512-dimensional vectors
        self.data = []

        if data_path.endswith(".txt"):
            with open(data_path, "r") as f:
                for i, line in enumerate(f):
                    try:
                        values = line.strip().split()

                        vector = [float(val) for val in values]

                        if len(vector) != 512:
                            print(
                                f"Warning: Line {i + 1} has vector with length {len(vector)}, expected 512"
                            )
                            continue
                        self.data.append(vector)
                    except ValueError as e:
                        print(f"Error parsing line {i + 1}: {e}")
                        continue

            # Convert to numpy array
            self.data = np.array(self.data, dtype=np.float32)
        elif data_path.endswith(".npy"):
            self.data = np.load(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}. Expected .txt or .npy")

        # Normalize the data
        if normalize:
            self.mean = np.mean(self.data, axis=0)
            self.std = np.std(self.data, axis=0)
            # Add small epsilon to avoid division by zero
            self.data = (self.data - self.mean) / (self.std + 1e-8)

        self.transform = transform

        print(f"Loaded {len(self.data)} vectors with {self.data.shape[1]} dimensions")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)
        else:
            sample = torch.FloatTensor(sample)

        # Return sample twice (for autoencoder, input = target)
        return sample, sample


def get_data_loaders(
    data_path="../../data/dump.npy", batch_size=32, val_split=0.2, num_workers=4, normalize=True
):
    """
    Create train and validation dataloaders

    Args:
        data_path (str): Path to data file, defaults to "../../data/dump.npy"
        batch_size (int): Batch size
        val_split (float): Validation split ratio (0-1)
        num_workers (int): Number of worker processes
        normalize (bool): Whether to normalize the data

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create dataset
    dataset = AutoencoderDataset(data_path, normalize=normalize)

    # train/val split
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
