"""
Variational Autoencoder (VAE) for Latent Diffusion

Phase 5 introduces latent space diffusion, which allows scaling to higher resolutions
by performing flow matching in a compressed latent space instead of pixel space.

Why Latent Space?
-----------------
Pixel-space diffusion has computational cost O(H × W × C) per step:
- 32×32×3 = 3,072 dimensions (CIFAR-10)
- 256×256×3 = 196,608 dimensions (64× more!)
- 512×512×3 = 786,432 dimensions

Latent space diffusion compresses images first:
- 256×256×3 image → 32×32×4 latent (48× compression)
- Flow matching operates on the small latent
- Final latent is decoded back to pixels

This is the key innovation behind Stable Diffusion.

VAE Architecture Overview
-------------------------
The VAE has two main components:

1. Encoder: Image → Latent Distribution
   - Downsamples spatial dimensions (e.g., 64×64 → 8×8)
   - Outputs mean μ and log-variance log(σ²) of a Gaussian
   - Uses reparameterization trick: z = μ + σ × ε, where ε ~ N(0,1)

2. Decoder: Latent → Image
   - Upsamples spatial dimensions (e.g., 8×8 → 64×64)
   - Reconstructs the original image from latent z

Training Objective
------------------
The VAE is trained with the Evidence Lower Bound (ELBO):

    L_VAE = L_recon + β × L_KL

where:
    L_recon = ||x - decode(encode(x))||²  (reconstruction loss)
    L_KL = KL(q(z|x) || p(z))              (regularization)
         = -0.5 × sum(1 + log(σ²) - μ² - σ²)

The KL term encourages the latent distribution to be close to N(0, I),
which is important for sampling during generation.

For latent diffusion, we typically use β < 1 (e.g., β = 0.0001) to
prioritize reconstruction quality over strict regularization.

Reference: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
          "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm and SiLU activation.

    Architecture:
        x → Conv → GroupNorm → SiLU → Conv → GroupNorm → + → SiLU → out
        |                                                 |
        +-------------------(shortcut)-------------------+

    GroupNorm is preferred over BatchNorm for VAEs because:
    1. Works well with small batch sizes
    2. More stable during training
    3. No dependency on batch statistics during inference
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        # Shortcut connection if channels change
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H, W).
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return self.activation(h + self.shortcut(x))


class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing long-range dependencies.

    Used in the bottleneck of the VAE where spatial resolution is smallest.
    At low resolutions (e.g., 8×8), attention is computationally feasible
    and helps capture global image structure.

    Mathematical Formulation
    ------------------------
    Standard self-attention with queries, keys, values:

        Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V

    where Q, K, V are linear projections of the input.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        """
        Args:
            channels: Number of channels (also embedding dimension).
            num_heads: Number of attention heads.
        """
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply self-attention with residual connection.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape

        # Normalize
        h = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        # (B, C, H, W) -> (B, num_heads, head_dim, H*W) -> (B, num_heads, H*W, head_dim)
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        # Attention: (B, heads, HW, d) @ (B, heads, d, HW) -> (B, heads, HW, HW)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        out = attn @ v  # (B, heads, HW, d)

        # Reshape back
        out = out.transpose(2, 3).reshape(B, C, H, W)
        out = self.proj(out)

        return x + out


class Downsample(nn.Module):
    """
    Spatial downsampling by factor of 2.

    Uses strided convolution instead of pooling to learn the downsampling.
    This preserves more information than average/max pooling.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Spatial upsampling by factor of 2.

    Uses nearest-neighbor interpolation followed by convolution.
    This avoids checkerboard artifacts that can occur with transposed convolutions.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Encoder(nn.Module):
    """
    VAE Encoder: Image → Latent Distribution Parameters (μ, log σ²)

    Architecture for 64×64 input with 4× downsampling:

        Input: (B, 3, 64, 64)
            ↓ conv_in
        (B, 64, 64, 64)
            ↓ down_block_1 + downsample
        (B, 128, 32, 32)
            ↓ down_block_2 + downsample
        (B, 256, 16, 16)
            ↓ down_block_3 + downsample
        (B, 512, 8, 8)
            ↓ mid_block + attention
        (B, 512, 8, 8)
            ↓ conv_out
        Output: (B, 2*latent_channels, 8, 8)  → split into μ and log σ²

    The encoder outputs 2× the latent channels because we need
    both mean (μ) and log-variance (log σ²) for the Gaussian.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (8,),
    ):
        """
        Args:
            in_channels: Number of input image channels (3 for RGB).
            latent_channels: Number of latent channels (4 is common).
            base_channels: Base channel count (multiplied at each level).
            channel_multipliers: Channel multiplier at each resolution level.
            num_res_blocks: Number of residual blocks per level.
            attention_resolutions: Resolutions where attention is applied.
        """
        super().__init__()

        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Calculate channel counts at each level
        channels = [base_channels * m for m in channel_multipliers]
        num_levels = len(channel_multipliers)

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        in_ch = channels[0]
        current_res = 64  # Assume 64×64 input, adjust dynamically

        for level in range(num_levels):
            out_ch = channels[level]

            # Residual blocks at this level
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch

                # Add attention if at specified resolution
                if current_res in attention_resolutions:
                    blocks.append(AttentionBlock(out_ch))

            self.down_blocks.append(blocks)

            # Downsample (except at last level)
            if level < num_levels - 1:
                self.downsamplers.append(Downsample(out_ch))
                current_res //= 2
            else:
                self.downsamplers.append(nn.Identity())

        # Middle block
        mid_channels = channels[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels)
        self.mid_attn = AttentionBlock(mid_channels)
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels)

        # Output: project to 2× latent channels (for μ and log σ²)
        self.norm_out = nn.GroupNorm(32, mid_channels)
        self.conv_out = nn.Conv2d(mid_channels, 2 * latent_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode image to latent distribution parameters.

        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            mean: Mean of latent distribution, shape (B, latent_ch, H', W').
            logvar: Log-variance of latent distribution, shape (B, latent_ch, H', W').
        """
        # Initial convolution
        h = self.conv_in(x)

        # Downsampling path
        for blocks, downsampler in zip(self.down_blocks, self.downsamplers):
            for block in blocks:
                h = block(h)
            h = downsampler(h)

        # Middle
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        # Split into mean and log-variance
        mean, logvar = h.chunk(2, dim=1)

        return mean, logvar


class Decoder(nn.Module):
    """
    VAE Decoder: Latent → Image

    Architecture (inverse of encoder):

        Input: (B, latent_channels, 8, 8)
            ↓ conv_in
        (B, 512, 8, 8)
            ↓ mid_block + attention
        (B, 512, 8, 8)
            ↓ up_block_1 + upsample
        (B, 256, 16, 16)
            ↓ up_block_2 + upsample
        (B, 128, 32, 32)
            ↓ up_block_3 + upsample
        (B, 64, 64, 64)
            ↓ conv_out
        Output: (B, 3, 64, 64)

    The decoder mirrors the encoder structure but with upsampling instead of
    downsampling. It takes a latent vector z and reconstructs the image.
    """

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (8,),
    ):
        """
        Args:
            out_channels: Number of output image channels (3 for RGB).
            latent_channels: Number of latent channels.
            base_channels: Base channel count.
            channel_multipliers: Channel multiplier at each level.
            num_res_blocks: Number of residual blocks per level.
            attention_resolutions: Resolutions where attention is applied.
        """
        super().__init__()

        self.out_channels = out_channels
        self.latent_channels = latent_channels

        # Calculate channel counts (reversed for decoder)
        channels = [base_channels * m for m in channel_multipliers]
        num_levels = len(channel_multipliers)

        # Initial convolution from latent
        self.conv_in = nn.Conv2d(latent_channels, channels[-1], kernel_size=3, padding=1)

        # Middle block
        mid_channels = channels[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels)
        self.mid_attn = AttentionBlock(mid_channels)
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels)

        # Upsampling blocks (reversed order)
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        in_ch = channels[-1]
        current_res = 64 // (2 ** (num_levels - 1))  # Start at lowest resolution

        for level in reversed(range(num_levels)):
            out_ch = channels[level]

            # Residual blocks at this level
            blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):  # +1 compared to encoder
                blocks.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch

                # Add attention if at specified resolution
                if current_res in attention_resolutions:
                    blocks.append(AttentionBlock(out_ch))

            self.up_blocks.append(blocks)

            # Upsample (except at first level which is actually last in reversed order)
            if level > 0:
                self.upsamplers.append(Upsample(out_ch))
                current_res *= 2
            else:
                self.upsamplers.append(nn.Identity())

        # Output convolution
        self.norm_out = nn.GroupNorm(32, channels[0])
        self.conv_out = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        """
        Decode latent to image.

        Args:
            z: Latent tensor of shape (B, latent_ch, H', W').

        Returns:
            Reconstructed images of shape (B, C, H, W).
        """
        # Initial convolution
        h = self.conv_in(z)

        # Middle
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        # Upsampling path
        for blocks, upsampler in zip(self.up_blocks, self.upsamplers):
            for block in blocks:
                h = block(h)
            h = upsampler(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class VAE(nn.Module):
    """
    Variational Autoencoder for Latent Diffusion.

    The VAE compresses images to a lower-dimensional latent space where
    flow matching can operate more efficiently.

    Mathematical Framework
    ----------------------
    The VAE defines a generative model:

        z ~ p(z) = N(0, I)           # Prior (standard normal)
        x ~ p(x|z) = Decoder(z)      # Likelihood (decoder output)

    And an inference model (encoder):

        q(z|x) = N(μ(x), σ²(x))      # Approximate posterior

    Training Objective (ELBO)
    -------------------------
    We maximize the Evidence Lower Bound:

        log p(x) >= E_q[log p(x|z)] - KL(q(z|x) || p(z))
                 = -L_recon - L_KL

    Reparameterization Trick
    ------------------------
    To backpropagate through the sampling operation z ~ q(z|x):

        z = μ + σ × ε,  where ε ~ N(0, I)

    This makes the random sampling differentiable with respect to μ and σ.

    Scaling Factor
    --------------
    The latent space is typically scaled to have unit variance. We compute
    a scaling factor from the training data:

        z_scaled = z / scale_factor

    This helps with training stability when the flow matching model
    operates on the latent space.

    Usage for Latent Diffusion
    --------------------------
    1. Train VAE on images (or use pretrained)
    2. Encode training images: z = encode(x)
    3. Train flow matching on latents: v(z_t, t)
    4. Generate: sample z from flow, decode: x = decode(z)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (8,),
    ):
        """
        Args:
            in_channels: Number of input/output image channels.
            latent_channels: Number of latent channels.
            base_channels: Base channel count for encoder/decoder.
            channel_multipliers: Channel multipliers at each resolution.
            num_res_blocks: Residual blocks per resolution level.
            attention_resolutions: Resolutions where attention is applied.
        """
        super().__init__()

        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Encoder and decoder
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
        )

        self.decoder = Decoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
        )

        # Scaling factor for latent space (initialized to 1, computed from data)
        self.register_buffer('scale_factor', torch.tensor(1.0))

    def encode(self, x: Tensor, sample: bool = True) -> Tensor:
        """
        Encode images to latent space.

        Args:
            x: Input images of shape (B, C, H, W).
            sample: If True, sample from distribution. If False, return mean.

        Returns:
            Latent codes of shape (B, latent_ch, H', W').
        """
        mean, logvar = self.encoder(x)

        if sample:
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + std * eps
        else:
            z = mean

        # Apply scaling
        z = z / self.scale_factor

        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latents to images.

        Args:
            z: Latent codes of shape (B, latent_ch, H', W').

        Returns:
            Reconstructed images of shape (B, C, H, W).
        """
        # Undo scaling
        z = z * self.scale_factor
        return self.decoder(z)

    def forward(
        self,
        x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Full forward pass: encode, sample, decode.

        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            recon: Reconstructed images of shape (B, C, H, W).
            mean: Latent mean of shape (B, latent_ch, H', W').
            logvar: Latent log-variance of shape (B, latent_ch, H', W').
        """
        # Encode
        mean, logvar = self.encoder(x)

        # Sample with reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + std * eps

        # Decode
        recon = self.decoder(z)

        return recon, mean, logvar

    @staticmethod
    def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
        """
        Compute KL divergence loss: KL(q(z|x) || p(z)).

        Mathematical Derivation
        -----------------------
        For q(z|x) = N(μ, σ²) and p(z) = N(0, 1):

            KL = ∫ q(z) log(q(z)/p(z)) dz
               = ∫ N(μ,σ²) [log N(μ,σ²) - log N(0,1)] dz
               = 0.5 × (μ² + σ² - 1 - log(σ²))

        We sum over all latent dimensions and average over batch.

        Args:
            mean: Latent mean of shape (B, C, H, W).
            logvar: Latent log-variance of shape (B, C, H, W).

        Returns:
            Scalar KL divergence loss.
        """
        # KL = 0.5 * sum(μ² + σ² - 1 - log(σ²))
        # Since logvar = log(σ²), σ² = exp(logvar)
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return kl / mean.shape[0]  # Average over batch

    @staticmethod
    def reconstruction_loss(recon: Tensor, target: Tensor) -> Tensor:
        """
        Compute reconstruction loss (MSE).

        Args:
            recon: Reconstructed images of shape (B, C, H, W).
            target: Target images of shape (B, C, H, W).

        Returns:
            Scalar reconstruction loss.
        """
        return F.mse_loss(recon, target, reduction='mean')

    def loss(
        self,
        x: Tensor,
        kl_weight: float = 0.00001
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute total VAE loss.

        Loss Function
        -------------
        L = L_recon + β × L_KL

        For latent diffusion, we use very small β (e.g., 0.00001) because:
        1. We prioritize reconstruction quality
        2. The diffusion model will handle generation
        3. We just need good encoding/decoding

        Args:
            x: Input images of shape (B, C, H, W).
            kl_weight: Weight for KL term (β). Default is very small.

        Returns:
            loss: Total scalar loss.
            metrics: Dictionary with individual loss components.
        """
        # Forward pass
        recon, mean, logvar = self.forward(x)

        # Compute losses
        recon_loss = self.reconstruction_loss(recon, x)
        kl_loss = self.kl_loss(mean, logvar)

        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss

        metrics = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
        }

        return total_loss, metrics

    @torch.no_grad()
    def compute_scale_factor(self, dataloader, num_batches: int = 100) -> float:
        """
        Compute scaling factor from training data.

        The scale factor normalizes the latent space to have approximately
        unit variance, which helps flow matching training.

        Args:
            dataloader: DataLoader with training images.
            num_batches: Number of batches to use for estimation.

        Returns:
            Computed scale factor.
        """
        self.eval()

        latent_stds = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(next(self.parameters()).device)

            # Encode without sampling (use mean)
            mean, _ = self.encoder(images)
            latent_stds.append(mean.std().item())

        # Scale factor is the average std
        scale = sum(latent_stds) / len(latent_stds)
        self.scale_factor = torch.tensor(scale)

        return scale


# =============================================================================
# Smaller VAE for CIFAR-10 / MNIST (32×32 or 28×28)
# =============================================================================


class SmallVAE(nn.Module):
    """
    Smaller VAE suitable for CIFAR-10 (32×32) or MNIST (28×28).

    For small images, we don't need as much compression. This VAE:
    - Downsamples 4× (32×32 → 8×8 or 28×28 → 7×7)
    - Uses fewer channels and layers
    - Still provides meaningful compression for demonstrating latent diffusion

    Architecture
    ------------
    For 32×32 input:
        (B, 3, 32, 32) → Encoder → (B, 4, 8, 8) → Decoder → (B, 3, 32, 32)

    Compression ratio: 32×32×3 = 3,072 → 8×8×4 = 256 (12× compression)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        hidden_channels: int = 64,
    ):
        """
        Args:
            in_channels: Number of input/output image channels.
            latent_channels: Number of latent channels.
            hidden_channels: Hidden layer channels.
        """
        super().__init__()

        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Encoder: 32→16→8
        self.encoder = nn.Sequential(
            # 32×32 → 16×16
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            ResidualBlock(hidden_channels, hidden_channels),

            # 16×16 → 8×8
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU(),
            ResidualBlock(hidden_channels * 2, hidden_channels * 2),

            # Output: mean and logvar
            nn.Conv2d(hidden_channels * 2, latent_channels * 2, 3, padding=1),
        )

        # Decoder: 8→16→32
        self.decoder = nn.Sequential(
            # 8×8
            nn.Conv2d(latent_channels, hidden_channels * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU(),
            ResidualBlock(hidden_channels * 2, hidden_channels * 2),

            # 8×8 → 16×16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            ResidualBlock(hidden_channels, hidden_channels),

            # 16×16 → 32×32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
        )

        # Scaling factor
        self.register_buffer('scale_factor', torch.tensor(1.0))

    def encode(self, x: Tensor, sample: bool = True) -> Tensor:
        """Encode images to latent space."""
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=1)

        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + std * eps
        else:
            z = mean

        return z / self.scale_factor

    def decode(self, z: Tensor) -> Tensor:
        """Decode latents to images."""
        z = z * self.scale_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Full forward pass."""
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=1)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + std * eps

        recon = self.decoder(z)

        return recon, mean, logvar

    def loss(
        self,
        x: Tensor,
        kl_weight: float = 0.00001
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute VAE loss."""
        recon, mean, logvar = self.forward(x)

        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / x.shape[0]

        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss, {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
        }

    @torch.no_grad()
    def compute_scale_factor(self, dataloader, num_batches: int = 100) -> float:
        """Compute scaling factor from data."""
        self.eval()

        latent_stds = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(next(self.parameters()).device)
            h = self.encoder(images)
            mean, _ = h.chunk(2, dim=1)
            latent_stds.append(mean.std().item())

        scale = sum(latent_stds) / len(latent_stds)
        self.scale_factor = torch.tensor(scale)

        return scale
