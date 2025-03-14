import torch
import torch.nn as nn
import torch.nn.functional as F


# Vanilla Autoencoder with linear layers
class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        h = self.encoder(x)

        return self.decoder(h)


class ConvAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ConvAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: 1D conv layers
        self.encoder_conv = nn.Sequential(
            # Input shape: (batch, 1, input_dim)
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (batch, 16, input_dim/2)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (batch, 32, input_dim/4)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (batch, 64, input_dim/8)
            nn.ReLU(),
        )

        conv_out_dim = (input_dim // 8) * 64

        # Fully connected layers
        self.fc_enc = nn.Sequential(
            nn.Linear(conv_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

        # Decoder: Fully connected layers to expand the latent vector back,
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, conv_out_dim),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            # Reshape to (batch, 64, input_dim/8) and then apply transposed conv layers.
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        # x shape: (batch, input_dim)
        # Reshape to (batch, 1, input_dim) for convolution
        x = x.unsqueeze(1)
        x = self.encoder_conv(x)  # Shape: (batch, 64, input_dim//8)
        x = x.view(x.size(0), -1)  # Flatten to (batch, conv_out_dim)
        latent = self.fc_enc(x)

        x = self.fc_dec(latent)
        # Reshape back into convolutional feature map: (batch, 64, input_dim//8)
        x = x.view(x.size(0), 64, self.input_dim // 8)
        recon = self.decoder_conv(x)
        # Squeeze channel dimension to output shape (batch, input_dim)
        recon = recon.squeeze(1)
        return recon


# Vector Quantization layer (basically copying https://huggingface.co/blog/ariG23498/understand-vq)
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        """
        Args:
            num_embeddings (int): Number of latent embeddings (codebook size)
            embedding_dim (int): Dimensionality of the latent vectors
            commitment_cost (float): Weight for the commitment loss
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Create codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        # Uniformly initialize codebook weights
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        """
        z: (batch, latent_dim) continuous latent vectors from encoder
        Returns:
            z_q: quantized latent vectors (batch, latent_dim)
            vq_loss: vector quantization loss (codebook loss + commitment loss)
            perplexity: scalar, for monitoring the codebook usage
        """
        # Reshape z to [batch_size, embedding_dim]
        batch_size = z.size(0)
        z_flat = z.reshape(batch_size, self.embedding_dim)

        # z: [batch_size, embedding_dim] -> [batch_size, 1, embedding_dim]
        # embeddings: [num_embeddings, embedding_dim] -> [1, num_embeddings, embedding_dim]
        z_reshaped = z_flat.unsqueeze(1)
        embeddings = self.embeddings.weight.unsqueeze(0)

        distances = torch.sum((z_reshaped - embeddings) ** 2, dim=2)

        # Find nearest embedding for each z vector
        encoding_indices = torch.argmin(distances, dim=1)  # Shape: [batch_size]

        # Convert to one-hot encodings: [batch_size, num_embeddings]
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Quantize latent vectors: [batch_size, embedding_dim]
        z_q = torch.matmul(encodings, self.embeddings.weight)

        # Computation of losses:
        # Codebook loss
        loss_codebook = F.mse_loss(z_q.detach(), z_flat)
        # Commitment loss
        loss_commit = F.mse_loss(z_q, z_flat.detach())

        # Compute average encoding probabilities across batch
        avg_probs = torch.mean(encodings, dim=0)

        # Compute perplexity for monitoring codebook usage
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Tried adding diversity loss to force higher perplexity
        target_perplexity = self.num_embeddings * 0.1  # Target at least 10% utilization
        perplexity_ratio = torch.clamp(target_perplexity / (perplexity + 1e-10), min=0.0, max=10.0)
        diversity_loss = perplexity_ratio * 0.1  # Scale to not dominate other losses

        # Add up all losses
        vq_loss = loss_codebook + self.commitment_cost * loss_commit + diversity_loss

        # Use a straight-through estimator: in the forward pass, replace z with z_q,
        # but during backprop, the gradient flows as if z was used.
        z_q = z_flat + (z_q - z_flat).detach()
        # shape check
        z_q = z_q.reshape(z.shape)

        return z_q, vq_loss, perplexity


# Modified ConvAE with Vector Quantization layer
class ConvAE_VQ(nn.Module):
    def __init__(self, input_dim, latent_dim, num_embeddings=512):
        super(ConvAE_VQ, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: 1D conv layers
        self.encoder_conv = nn.Sequential(
            # Input shape: (batch, 1, input_dim)
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (batch, 16, input_dim/2)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (batch, 32, input_dim/4)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (batch, 64, input_dim/8)
            nn.ReLU(),
        )
        self.conv_out_dim = (input_dim // 8) * 64

        # Fully connected layers
        self.fc_enc = nn.Sequential(
            nn.Linear(self.conv_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

        # Vector Quantization layer (discretizes latent space)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost=0.25)

        # Decoder: Fully connected layers, then 1D transposed conv layers to reconstruct signal
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.conv_out_dim),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            # Reshape to (batch, 64, input_dim/8) then apply transposed conv layers
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # x shape: (batch, input_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        x = self.encoder_conv(x)  # -> (batch, 64, input_dim/8)

        x_flat = x.view(batch_size, -1)  # Flatten to (batch, conv_out_dim)
        latent = self.fc_enc(x_flat)  # (batch, latent_dim)

        # Pass through VQ layer to get quantized latent vector
        z_q, vq_loss, perplexity = self.vq_layer(latent)

        # Decode the quantized latent vector
        x = self.fc_dec(z_q)

        # Calculate expected dimensions
        expected_dim = self.input_dim // 8
        x = x.view(batch_size, 64, expected_dim)

        recon = self.decoder_conv(x)
        recon = recon.squeeze(1)  # (batch, input_dim)

        return recon, vq_loss, perplexity
