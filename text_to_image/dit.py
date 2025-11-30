"""
Diffusion Transformer (DiT) for Flow Matching

Phase 2 replaces the simple CNN with a transformer architecture that:
1. Patchifies images into sequences of patches
2. Uses positional embeddings for 2D spatial information
3. Applies adaptive layer norm (adaLN) for timestep conditioning
4. Predicts the velocity field via transformer blocks

Phase 3 adds class-conditional generation:
1. Adds class embedding (0-9 for MNIST digits)
2. Combines class embedding with timestep embedding
3. Supports classifier-free guidance (CFG) via label dropout

Phase 4 adds text-conditional generation:
1. Uses CLIP text encoder (frozen) to embed text prompts
2. Adds cross-attention layers to attend to text tokens
3. Enables natural language control: "a photo of a cat" → cat image

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


# =============================================================================
# Phase 3: Class-Conditional DiT
# =============================================================================


class ClassEmbedding(nn.Module):
    """
    Embed discrete class labels into continuous vectors.

    For MNIST, we have 10 classes (digits 0-9). Each class gets mapped to
    a learnable embedding vector that the model uses to understand what
    digit to generate.

    Similar to word embeddings in NLP: just as "cat" → vector, here "7" → vector.
    """

    def __init__(self, num_classes: int, embed_dim: int):
        """
        Args:
            num_classes: Number of classes (10 for MNIST).
            embed_dim: Dimension of embedding vectors (matches timestep embed).
        """
        super().__init__()
        # Standard learnable embedding table
        # +1 for the "null" class used during CFG (unconditional generation)
        self.embed = nn.Embedding(num_classes + 1, embed_dim)
        self.num_classes = num_classes
        self.null_class = num_classes  # Index for unconditional

    def forward(self, labels: Tensor) -> Tensor:
        """
        Args:
            labels: Class indices of shape (B,). Values in [0, num_classes-1],
                   or num_classes for unconditional (null class).

        Returns:
            Class embeddings of shape (B, embed_dim).
        """
        return self.embed(labels)


class ConditionalDiT(nn.Module):
    """
    Diffusion Transformer with class conditioning for controlled generation.

    Architecture extends DiT by:
    1. Adding a class embedding that's summed with timestep embedding
    2. The combined embedding conditions all transformer blocks via adaLN
    3. During training, randomly drop class labels for classifier-free guidance

    Key insight: By training with occasional "null" class (no conditioning),
    the model learns both conditional and unconditional generation. At inference,
    we can blend both predictions to achieve stronger conditioning.

    The conditioning formula is simple:
        combined_embedding = timestep_embedding + class_embedding

    This combined vector then modulates every layer through adaLN, telling
    the model both "how noisy is this?" (timestep) and "what should I make?" (class).
    """

    def __init__(
        self,
        num_classes: int = 10,
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
            num_classes: Number of classes for conditioning (10 for MNIST).
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

        self.num_classes = num_classes
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

        # Conditioning embeddings
        cond_dim = embed_dim * 4
        self.time_embed = TimestepEmbedding(embed_dim, cond_dim)
        self.class_embed = ClassEmbedding(num_classes, cond_dim)

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
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Zero-init the final projection for stable training
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

        # Initialize class embeddings with small values
        nn.init.normal_(self.class_embed.embed.weight, std=0.02)

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

        x = x.view(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, c, h * p, w * p)

        return x

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor | None = None,
    ) -> Tensor:
        """
        Predict velocity field given noised image, timestep, and optional class.

        Args:
            x: Noised images of shape (B, C, H, W).
            t: Timesteps in [0, 1] of shape (B,).
            y: Class labels of shape (B,). If None, uses null class (unconditional).
               Values should be in [0, num_classes-1].

        Returns:
            Predicted velocity of shape (B, C, H, W).
        """
        batch_size = x.shape[0]

        # Patchify and embed
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional embeddings
        x = self.pos_embed(x)

        # Get timestep conditioning
        time_cond = self.time_embed(t)  # (B, cond_dim)

        # Get class conditioning
        if y is None:
            # Use null class for unconditional generation
            y = torch.full(
                (batch_size,), self.class_embed.null_class,
                dtype=torch.long, device=x.device
            )
        class_cond = self.class_embed(y)  # (B, cond_dim)

        # Combine conditioning: simple addition works well
        # Both embeddings live in the same space and jointly modulate the network
        cond = time_cond + class_cond  # (B, cond_dim)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, cond)

        # Final projection
        x = self.final_norm(x)
        x = self.final_proj(x)  # (B, num_patches, patch_dim)

        # Unpatchify to image
        x = self.unpatchify(x)  # (B, C, H, W)

        return x


# =============================================================================
# Phase 4: Text-Conditional DiT with Cross-Attention
# =============================================================================


class CrossAttention(nn.Module):
    """
    Cross-attention layer for attending to text embeddings.

    Mathematical Framework
    ----------------------
    Cross-attention allows image patches to attend to text tokens:

        CrossAttn(X, Z) = softmax(Q K^T / sqrt(d)) V

    where:
        Q = X @ W_Q in R^{N x d}     (queries from image patches)
        K = Z @ W_K in R^{M x d}     (keys from text tokens)
        V = Z @ W_V in R^{M x d}     (values from text tokens)

    Here:
        X in R^{N x d} = image patch embeddings (N patches)
        Z in R^{M x d} = text token embeddings (M tokens)
        d = attention dimension

    The attention matrix A = softmax(QK^T / sqrt(d)) in R^{N x M} tells us
    how much each image patch should attend to each text token.

    Intuition
    ---------
    For the prompt "a RED dog RUNNING":
    - When generating color regions, patches attend strongly to "RED"
    - When generating motion/pose, patches attend to "RUNNING"
    - All patches attend somewhat to "dog" for overall structure

    This is how text guides the generation at a fine-grained level.

    Comparison to Self-Attention
    ----------------------------
    | Aspect       | Self-Attention        | Cross-Attention          |
    |--------------|----------------------|--------------------------|
    | Q, K, V from | Same sequence        | Q from X, K/V from Z     |
    | Purpose      | Inter-patch relations | Text-to-image transfer   |
    | Attention    | N x N                 | N x M                    |

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(
        self,
        embed_dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        """
        Args:
            embed_dim: Dimension of image patch embeddings (query dimension).
            context_dim: Dimension of text embeddings (key/value dimension).
            num_heads: Number of attention heads.
            dropout: Dropout rate on attention weights.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Query projection (from image patches)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        # Key and Value projections (from text tokens)
        self.k_proj = nn.Linear(context_dim, embed_dim)
        self.v_proj = nn.Linear(context_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Compute cross-attention from image patches to text tokens.

        Args:
            x: Image patch embeddings of shape (B, N, embed_dim)
               where N = number of patches.
            context: Text token embeddings of shape (B, M, context_dim)
               where M = number of text tokens.
            context_mask: Optional attention mask of shape (B, M)
               True for real tokens, False for padding.

        Returns:
            Attended features of shape (B, N, embed_dim).

        Mathematical Details
        --------------------
        1. Project queries, keys, values:
           Q = x @ W_Q,  K = context @ W_K,  V = context @ W_V

        2. Reshape for multi-head attention:
           Q: (B, N, H, d) -> (B, H, N, d)
           K, V: (B, M, H, d) -> (B, H, M, d)

        3. Compute attention scores:
           A = softmax(Q @ K^T / sqrt(d))  shape: (B, H, N, M)

        4. Apply attention to values:
           out = A @ V  shape: (B, H, N, d)

        5. Reshape and project:
           out: (B, N, H*d) -> (B, N, embed_dim)
        """
        B, N, _ = x.shape
        M = context.shape[1]
        H = self.num_heads
        d = self.head_dim

        # Project
        q = self.q_proj(x)        # (B, N, embed_dim)
        k = self.k_proj(context)  # (B, M, embed_dim)
        v = self.v_proj(context)  # (B, M, embed_dim)

        # Reshape for multi-head attention
        q = q.view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        k = k.view(B, M, H, d).transpose(1, 2)  # (B, H, M, d)
        v = v.view(B, M, H, d).transpose(1, 2)  # (B, H, M, d)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, M)

        # Apply mask for padding tokens
        if context_mask is not None:
            # context_mask: (B, M) -> (B, 1, 1, M)
            mask = context_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, H, N, d)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, self.embed_dim)  # (B, N, embed_dim)
        out = self.out_proj(out)

        return out


class TextConditionedDiTBlock(nn.Module):
    """
    DiT block with both self-attention and cross-attention to text.

    Architecture
    ------------
    Each block has three main components:

    1. Self-Attention (with adaLN):
       - Image patches attend to each other
       - Captures spatial relationships

    2. Cross-Attention (with adaLN):
       - Image patches attend to text tokens
       - Injects text conditioning

    3. MLP (with adaLN):
       - Per-patch nonlinear transformation

    Block Diagram
    -------------
    ```
                                    +------------------+
                                    | Text Embeddings  |
                                    |   Z in R^{M x D} |
                                    +---------+--------+
                                              |
    x --> adaLN --> Self-Attn --> + --> adaLN --> Cross-Attn --> + --> adaLN --> MLP --> + --> out
    |                             |   |                          |   |                   |
    +-------(residual)------------+   +-------(residual)---------+   +----(residual)-----+
    ```

    Mathematical Formulation
    ------------------------
    Let x be the input, c be the timestep conditioning, Z be text embeddings:

    x' = x + SelfAttn(adaLN(x, c))
    x'' = x' + CrossAttn(adaLN(x', c), Z)
    out = x'' + MLP(adaLN(x'', c))

    The adaLN modulates each sublayer based on timestep, while
    cross-attention modulates based on text content.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        cond_dim: int,
        context_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            embed_dim: Dimension of patch embeddings.
            num_heads: Number of attention heads.
            cond_dim: Dimension of timestep conditioning.
            context_dim: Dimension of text embeddings.
            mlp_ratio: MLP hidden dimension = embed_dim * mlp_ratio.
            dropout: Dropout rate.
        """
        super().__init__()

        # Self-attention with adaLN
        self.norm1 = AdaLN(embed_dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout, batch_first=True
        )

        # Cross-attention with adaLN
        self.norm2 = AdaLN(embed_dim, cond_dim)
        self.cross_attn = CrossAttention(
            embed_dim, context_dim, num_heads, dropout
        )

        # MLP with adaLN
        self.norm3 = AdaLN(embed_dim, cond_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass through the text-conditioned DiT block.

        Args:
            x: Patch embeddings of shape (B, N, embed_dim).
            cond: Timestep conditioning of shape (B, cond_dim).
            context: Text token embeddings of shape (B, M, context_dim).
            context_mask: Optional mask of shape (B, M).

        Returns:
            Transformed embeddings of shape (B, N, embed_dim).
        """
        # Self-attention block with residual
        x_norm = self.norm1(x, cond)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Cross-attention block with residual
        x_norm = self.norm2(x, cond)
        cross_out = self.cross_attn(x_norm, context, context_mask)
        x = x + cross_out

        # MLP block with residual
        x_norm = self.norm3(x, cond)
        x = x + self.mlp(x_norm)

        return x


class TextConditionalDiT(nn.Module):
    """
    Diffusion Transformer with text conditioning via cross-attention.

    This is the full text-to-image architecture. Given:
    - Noisy image x_t
    - Timestep t
    - Text prompt "a photo of a cat"

    The model predicts the velocity field v(x_t, t, text) that guides
    generation toward images matching the text description.

    Architecture Overview
    ---------------------
    ```
    Text Prompt                    Noisy Image x_t          Timestep t
         |                              |                       |
         v                              v                       v
    +---------+                  +-----------+           +-----------+
    |  CLIP   |                  | Patchify  |           | Time Emb  |
    | Encoder |                  | + PosEmb  |           |   (MLP)   |
    +----+----+                  +-----+-----+           +-----+-----+
         |                             |                       |
         | Z in R^{M x D_text}         | X in R^{N x D}        | c in R^{D_cond}
         |                             |                       |
         |                             v                       |
         |                    +-----------------+              |
         +------------------->| TextConditioned |<-------------+
                              |    DiT Blocks   |
                              +--------+--------+
                                       |
                                       v
                              +-----------------+
                              |   Unpatchify    |
                              +--------+--------+
                                       |
                                       v
                              Velocity v in R^{C x H x W}
    ```

    Key Differences from ConditionalDiT
    ------------------------------------
    | Aspect          | ConditionalDiT         | TextConditionalDiT      |
    |-----------------|------------------------|-------------------------|
    | Conditioning    | Class labels (0-9)     | Text prompts            |
    | Embedding       | Learnable table        | CLIP encoder (frozen)   |
    | Integration     | Added to timestep      | Cross-attention         |
    | Flexibility     | Fixed vocabulary       | Open vocabulary         |
    | Representation  | D_cond vector          | M x D_text sequence     |

    The cross-attention mechanism allows:
    1. Per-token attention: "RED dog" -> color info from "RED"
    2. Compositional understanding: "blue banana" works
    3. Variable-length prompts: Short or long descriptions

    Classifier-Free Guidance with Text
    ----------------------------------
    CFG works similarly to class-conditional:

    v_cfg = v_uncond + w * (v_cond - v_uncond)

    where:
    - v_cond = model(x, t, text_embedding)
    - v_uncond = model(x, t, null_text_embedding)

    The "null text" is typically an empty string "", whose CLIP embedding
    serves as the unconditional baseline.

    Reference: "High-Resolution Image Synthesis with Latent Diffusion Models"
    (Rombach et al., 2022) - Stable Diffusion uses this architecture.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        context_dim: int = 512,
    ):
        """
        Args:
            img_size: Input image size (assumes square).
            patch_size: Size of each patch.
            in_channels: Number of input channels (3 for RGB).
            embed_dim: Transformer embedding dimension.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dim = embed_dim * mlp_ratio.
            dropout: Dropout rate.
            context_dim: Dimension of text embeddings (512 for CLIP ViT-B).
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.context_dim = context_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        grid_size = self.patch_embed.grid_size

        # Positional embedding
        self.pos_embed = SinusoidalPosEmb2D(embed_dim, grid_size)

        # Timestep embedding
        cond_dim = embed_dim * 4
        self.time_embed = TimestepEmbedding(embed_dim, cond_dim)

        # Text projection: map CLIP dim to model's context dim
        # This allows flexibility in context_dim used in cross-attention
        self.text_proj = nn.Linear(context_dim, context_dim)

        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList([
            TextConditionedDiTBlock(
                embed_dim, num_heads, cond_dim, context_dim, mlp_ratio, dropout
            )
            for _ in range(depth)
        ])

        # Final layers
        self.final_norm = nn.LayerNorm(embed_dim)
        patch_dim = patch_size * patch_size * in_channels
        self.final_proj = nn.Linear(embed_dim, patch_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following DiT paper recommendations."""
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

        x = x.view(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, c, h * p, w * p)

        return x

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Predict velocity field given noisy image, timestep, and text.

        Args:
            x: Noisy images of shape (B, C, H, W).
            t: Timesteps in [0, 1] of shape (B,).
            context: Text embeddings of shape (B, M, context_dim).
                These come from CLIP text encoder.
            context_mask: Optional attention mask of shape (B, M).
                True for real tokens, False for padding.

        Returns:
            Predicted velocity of shape (B, C, H, W).

        Forward Pass Details
        --------------------
        1. Patchify image: (B, C, H, W) -> (B, N, embed_dim)
        2. Add positional embeddings
        3. Embed timestep: t -> c in R^{cond_dim}
        4. Project text: context -> context' (optional dim change)
        5. Process through DiT blocks with cross-attention
        6. Final norm and projection
        7. Unpatchify: (B, N, patch_dim) -> (B, C, H, W)
        """
        # Patchify and embed
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional embeddings
        x = self.pos_embed(x)

        # Get timestep conditioning
        cond = self.time_embed(t)  # (B, cond_dim)

        # Project text embeddings (identity if dims match, else learned projection)
        context = self.text_proj(context)  # (B, M, context_dim)

        # Process through transformer blocks with cross-attention
        for block in self.blocks:
            x = block(x, cond, context, context_mask)

        # Final projection
        x = self.final_norm(x)
        x = self.final_proj(x)  # (B, num_patches, patch_dim)

        # Unpatchify to image
        x = self.unpatchify(x)  # (B, C, H, W)

        return x
