import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        """
        Args:
            num_embeddings: Number of discrete embeddings (codebook size)
            embedding_dim: Dimensionality of each embedding vector
            commitment_cost: Weight for the commitment loss term
        """
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Create an embedding table of shape (num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs, return_indices=False):
        """
        Args:
            inputs: Tensor of shape (B, embedding_dim)
            return_indices: If True, returns the discrete indices instead of the quantized tensor.
        Returns:
            Either:
              - quantized: Tensor of shape (B, embedding_dim) with gradients preserved
              - indices: LongTensor of shape (B,) containing discrete indices
            along with the vq_loss and perplexity.
        """
        # inputs is of shape (B, embedding_dim)
        flat_input = inputs  # already (B, embedding_dim)

        # Compute distances between each input vector and embedding vectors.
        # distances: (B, num_embeddings)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # Get the encoding indices for the closest embedding vector for each input.
        encoding_indices = torch.argmin(distances, dim=1)  # shape: (B,)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)

        # Quantize: multiply one-hot encodings with embedding weight to get quantized vectors.
        quantized = torch.matmul(encodings, self.embedding.weight)  # shape: (B, embedding_dim)

        # Compute commitment and codebook loss
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        q_latent_loss = F.mse_loss(quantized, flat_input.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Compute perplexity (for monitoring)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if return_indices:
            # Return the indices (shape: (B,)) along with the loss and perplexity.
            return encoding_indices, loss, perplexity
        else:
            # Use straight-through estimator: propagate gradients from inputs.
            quantized = flat_input + (quantized - flat_input).detach()
            return quantized, loss, perplexity


class VQVAE(nn.Module):
    def __init__(self, input_dim=512, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        """
        Args:
            input_dim: Dimensionality of the input vector (default 512)
            embedding_dim: Dimensionality of the latent representation (default 64)
            num_embeddings: Number of embeddings in the codebook
            commitment_cost: Weight for the commitment loss term
        """
        super(VQVAE, self).__init__()
        # Encoder: compresses the input vector into a latent representation.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, embedding_dim)
        )

        # Vector quantizer: discretizes the latent representation.
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # Decoder: first looks up the embedding vectors based on integer indices and then reconstructs the input.
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Linear(256, input_dim)
        )

    def decode(self, indices):
        """
        Given integer indices of shape (B,), look up the corresponding embeddings and decode.
        """
        # Look up embedding vectors from the codebook.
        quantized = self.vq_layer.embedding(indices)  # shape: (B, embedding_dim)
        # Decode to reconstruct the original input vector.
        recon = self.decoder(quantized)
        return recon

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, input_dim)
        Returns:
            recon: Reconstructed input (B, input_dim)
            vq_loss: The vector quantization loss
            perplexity: A measure of the codebook utilization
        """
        # Encode input to latent space.
        z = self.encoder(x)  # shape: (B, embedding_dim)

        # Quantize the latent representation
        quantized, vq_loss, perplexity = self.vq_layer(z, return_indices=False)

        # Decode.
        recon = self.decoder(quantized)
        return recon, vq_loss, perplexity


# Example usage:
if __name__ == "__main__":
    # Create a sample batch of vectors with shape (BATCH_SIZE, 512)
    batch_size = 4
    input_dim = 512
    x = torch.randn(batch_size, input_dim)

    model = VQVAE(input_dim=input_dim, embedding_dim=64, num_embeddings=512, commitment_cost=0.25)
    recon, vq_loss, perplexity = model(x)

    print("Input shape:", x.shape)
    print("Reconstructed shape:", recon.shape)
    print("VQ Loss:", vq_loss.item())
    print("Perplexity:", perplexity.item())
