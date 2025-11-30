"""
Diffusion Transformer (DiT) for Flow Matching

Phase 2 replaces the simple CNN with a transformer architecture that:
1. Patchifies images into sequences of patches
2. Uses positional embeddings for 2D spatial information
3. Applies adaptive layer norm (adaLN) for timestep conditioning
4. Predicts the velocity field via transformer blocks

Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2022)
https://arxiv.org/abs/2212.09748
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class PatchEmbed(nn.Module):
    """
    Convert image into a sequence of patch embeddings.

    For a 28x28 image with patch_size=4:
    - 28/4 = 7 patches per dimension
    - 7x7 = 49 patches total
    - Each patch is 4x4x1 = 16 pixels (for grayscale)
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        # Linear projection of flattened patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Images of shape (B, C, H, W)

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        # (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        # (B, embed_dim, grid, grid) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class SinusoidalPosEmb2D(nn.Module):
    """
    2D sinusoidal positional embeddings for image patches.

    Creates separate sin/cos embeddings for row and column positions,
    then concatenates them. This gives the model spatial awareness.
    """

    def __init__(self, embed_dim: int, grid_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        # Create fixed positional embeddings
        pos_embed = self._make_pos_embed()
        self.register_buffer("pos_embed", pos_embed)

    def _make_pos_embed(self) -> Tensor:
        """Generate 2D sinusoidal positional embeddings."""
        half_dim = self.embed_dim // 4  # Quarter for each of sin_x, cos_x, sin_y, cos_y

        # Frequencies for positional encoding
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)

        # Create position indices
        pos = torch.arange(self.grid_size)

        # Compute embeddings for each position
        pos_emb = pos[:, None] * emb[None, :]  # (grid_size, half_dim)

        # Sin and cos embeddings
        sin_emb = torch.sin(pos_emb)  # (grid_size, half_dim)
        cos_emb = torch.cos(pos_emb)  # (grid_size, half_dim)

        # Combine into full embedding: (grid_size, embed_dim//2)
        pos_emb_1d = torch.cat([sin_emb, cos_emb], dim=-1)

        # Create 2D grid of embeddings
        # For position (i, j), concatenate row_emb[i] and col_emb[j]
        row_emb = pos_emb_1d.unsqueeze(1).expand(-1, self.grid_size, -1)
        col_emb = pos_emb_1d.unsqueeze(0).expand(self.grid_size, -1, -1)

        # (grid_size, grid_size, embed_dim)
        pos_2d = torch.cat([row_emb, col_emb], dim=-1)

        # Flatten to (num_patches, embed_dim)
        pos_2d = pos_2d.view(-1, self.embed_dim)

        return pos_2d

    def forward(self, x: Tensor) -> Tensor:
        """Add positional embeddings to patch embeddings."""
        return x + self.pos_embed


class TimestepEmbedding(nn.Module):
    """
    Embed scalar timestep into a vector representation.

    Uses sinusoidal encoding followed by MLP, same as the CNN model.
    Output is used for adaptive layer norm modulation.
    """

    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Timesteps of shape (B,) in [0, 1]

        Returns:
            Time embeddings of shape (B, hidden_dim)
        """
        # Sinusoidal encoding
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # MLP
        return self.mlp(emb)


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (adaLN).

    Instead of learned scale/shift parameters, we predict them from
    the timestep embedding. This conditions each layer on the timestep.

    output = scale * LayerNorm(x) + shift
    where scale, shift = f(timestep_embedding)
    """

    def __init__(self, embed_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, embed_dim * 2)  # scale and shift

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, embed_dim)
            cond: Conditioning tensor of shape (B, cond_dim)

        Returns:
            Normalized and modulated tensor of shape (B, N, embed_dim)
        """
        # Get scale and shift from conditioning
        params = self.proj(cond)  # (B, embed_dim * 2)
        scale, shift = params.chunk(2, dim=-1)  # Each (B, embed_dim)

        # Apply layer norm
        x = self.norm(x)

        # Modulate: expand scale/shift to (B, 1, embed_dim) for broadcasting
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return x


class DiTBlock(nn.Module):
    """
    Diffusion Transformer block with adaLN conditioning.

    Architecture:
        x -> adaLN -> Self-Attention -> + -> adaLN -> MLP -> + -> out
        |                              |                    |
        +-----------(residual)---------+------(residual)----+

    The timestep conditions both the attention and MLP paths via adaLN.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Attention with adaLN
        self.norm1 = AdaLN(embed_dim, cond_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout, batch_first=True
        )

        # MLP with adaLN
        self.norm2 = AdaLN(embed_dim, cond_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: Patch embeddings of shape (B, N, embed_dim)
            cond: Timestep embedding of shape (B, cond_dim)

        Returns:
            Transformed embeddings of shape (B, N, embed_dim)
        """
        # Attention block with residual
        x_norm = self.norm1(x, cond)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP block with residual
        x_norm = self.norm2(x, cond)
        x = x + self.mlp(x_norm)

        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for velocity field prediction.

    Architecture:
        1. Patchify: Image -> patch embeddings
        2. Add positional embeddings
        3. Process through DiT blocks (with timestep conditioning)
        4. Final layer norm and linear projection
        5. Unpatchify: patch predictions -> image

    For MNIST (28x28, 1 channel):
        - patch_size=4 gives 7x7=49 patches
        - Each patch covers 4x4=16 pixels
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            img_size: Input image size (assumes square images).
            patch_size: Size of each patch (assumes square patches).
            in_channels: Number of input channels.
            embed_dim: Transformer embedding dimension.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dim = embed_dim * mlp_ratio.
            dropout: Dropout rate.
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size

        # Positional embedding
        self.pos_embed = SinusoidalPosEmb2D(embed_dim, grid_size)

        # Timestep embedding
        cond_dim = embed_dim * 4
        self.time_embed = TimestepEmbedding(embed_dim, cond_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, cond_dim, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final layers
        self.final_norm = nn.LayerNorm(embed_dim)

        # Project back to patch pixels
        patch_dim = patch_size * patch_size * in_channels
        self.final_proj = nn.Linear(embed_dim, patch_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following DiT paper recommendations."""
        # Initialize all linear layers and embeddings
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Zero-init the final projection for stable training
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def unpatchify(self, x: Tensor) -> Tensor:
        """
        Convert patch predictions back to image.

        Args:
            x: Patch predictions of shape (B, num_patches, patch_dim)

        Returns:
            Image of shape (B, C, H, W)
        """
        B = x.shape[0]
        p = self.patch_size
        h = w = self.img_size // p
        c = self.in_channels

        # (B, num_patches, patch_dim) -> (B, h, w, p, p, c)
        x = x.view(B, h, w, p, p, c)

        # (B, h, w, p, p, c) -> (B, c, h, p, w, p)
        x = x.permute(0, 5, 1, 3, 2, 4)

        # (B, c, h, p, w, p) -> (B, c, h*p, w*p)
        x = x.reshape(B, c, h * p, w * p)

        return x

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Predict velocity field given noised image and timestep.

        Args:
            x: Noised images of shape (B, C, H, W).
            t: Timesteps in [0, 1] of shape (B,).

        Returns:
            Predicted velocity of shape (B, C, H, W).
        """
        # Patchify and embed
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional embeddings
        x = self.pos_embed(x)

        # Get timestep conditioning
        cond = self.time_embed(t)  # (B, cond_dim)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, cond)

        # Final projection
        x = self.final_norm(x)
        x = self.final_proj(x)  # (B, num_patches, patch_dim)

        # Unpatchify to image
        x = self.unpatchify(x)  # (B, C, H, W)

        return x
