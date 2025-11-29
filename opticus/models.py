"""
Neural Network Models for Flow Matching

Phase 1 uses a simple CNN to predict velocity fields.
Later phases will upgrade to Diffusion Transformers (DiT).

The model takes:
- x_t: Noised image at timestep t, shape (B, C, H, W)
- t: Timestep value in [0, 1], shape (B,)

And outputs:
- v: Predicted velocity field, shape (B, C, H, W)
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal embeddings for timestep conditioning.

    Maps scalar timestep t to a high-dimensional vector using sin/cos functions
    at different frequencies. This gives the model a rich representation of time.

    Same technique used in Transformer positional encodings and diffusion models.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Output embedding dimension (should be even).
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        Embed timesteps into sinusoidal features.

        Args:
            t: Timestep values, shape (B,).

        Returns:
            Embeddings of shape (B, dim).
        """
        device = t.device
        half_dim = self.dim // 2

        # Compute frequencies: exp(-log(10000) * i / half_dim)
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Outer product: (B, 1) * (half_dim,) -> (B, half_dim)
        emb = t[:, None] * emb[None, :]

        # Concatenate sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb


class ResidualBlock(nn.Module):
    """
    Residual convolutional block with timestep conditioning.

    Architecture:
        x -> Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> + -> SiLU -> out
        |                                                      |
        +-------------------(shortcut)-------------------------+

    Timestep embedding is added after the first normalization via a linear projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = 8
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        # Project time embedding to channel dimension
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # Shortcut connection if channels change
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.SiLU()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Forward pass with timestep conditioning.

        Args:
            x: Input features, shape (B, C_in, H, W).
            t_emb: Time embedding, shape (B, time_emb_dim).

        Returns:
            Output features, shape (B, C_out, H, W).
        """
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)

        # Add time embedding (broadcast over spatial dimensions)
        t_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_proj

        h = self.activation(h)

        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)

        # Residual connection
        return self.activation(h + self.shortcut(x))


class SimpleUNet(nn.Module):
    """
    Simple U-Net style CNN for velocity prediction.

    This is a lightweight architecture suitable for MNIST (28x28).
    Uses encoder-decoder structure with skip connections.

    Architecture:
        Input (1, 28, 28)
          |
        Encoder: downsample with strided convs
          |
        Middle: residual blocks at lowest resolution
          |
        Decoder: upsample with transposed convs + skip connections
          |
        Output (1, 28, 28)
    """

    def __init__(
        self,
        in_channels: int = 1,
        model_channels: int = 64,
        time_emb_dim: int = 128,
    ):
        """
        Args:
            in_channels: Number of input image channels (1 for MNIST).
            model_channels: Base channel count (doubled at each level).
            time_emb_dim: Dimension of timestep embeddings.
        """
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial projection
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # Encoder (28 -> 14 -> 7)
        self.down1 = ResidualBlock(model_channels, model_channels, time_emb_dim)
        self.down_conv1 = nn.Conv2d(model_channels, model_channels, kernel_size=3, stride=2, padding=1)

        self.down2 = ResidualBlock(model_channels, model_channels * 2, time_emb_dim)
        self.down_conv2 = nn.Conv2d(model_channels * 2, model_channels * 2, kernel_size=3, stride=2, padding=1)

        # Middle (7x7)
        self.mid1 = ResidualBlock(model_channels * 2, model_channels * 2, time_emb_dim)
        self.mid2 = ResidualBlock(model_channels * 2, model_channels * 2, time_emb_dim)

        # Decoder (7 -> 14 -> 28)
        self.up_conv2 = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = ResidualBlock(model_channels * 4, model_channels, time_emb_dim)  # *4 due to skip

        self.up_conv1 = nn.ConvTranspose2d(model_channels, model_channels, kernel_size=4, stride=2, padding=1)
        self.up1 = ResidualBlock(model_channels * 2, model_channels, time_emb_dim)  # *2 due to skip

        # Final projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Predict velocity field given noised image and timestep.

        Args:
            x: Noised images x_t, shape (B, C, H, W).
            t: Timesteps in [0, 1], shape (B,).

        Returns:
            Predicted velocity, shape (B, C, H, W).
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Initial conv
        h = self.conv_in(x)

        # Encoder
        h1 = self.down1(h, t_emb)
        h = self.down_conv1(h1)

        h2 = self.down2(h, t_emb)
        h = self.down_conv2(h2)

        # Middle
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # Decoder with skip connections
        h = self.up_conv2(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up2(h, t_emb)

        h = self.up_conv1(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up1(h, t_emb)

        # Output
        return self.conv_out(h)
