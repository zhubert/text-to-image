"""
Training Utilities for Flow Matching Models

Provides helper functions and a training loop for flow matching.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from text_to_image.flow import FlowMatching


def get_device() -> torch.device:
    """
    Get the best available device for training.

    Priority: MPS (Apple Silicon) > CUDA > CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer:
    """
    Training loop for flow matching models.

    Handles:
    - Device management (MPS/CUDA/CPU)
    - Optimizer and scheduler setup
    - Training loop with progress tracking
    - Checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        device: torch.device | None = None,
    ):
        """
        Args:
            model: Neural network to train.
            dataloader: DataLoader providing training batches.
            lr: Learning rate.
            weight_decay: AdamW weight decay.
            device: Device to train on (auto-detected if None).
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.dataloader = dataloader

        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.flow = FlowMatching()
        self.losses: list[float] = []

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            total_epochs: Total number of epochs (for progress display).

        Returns:
            Average loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            leave=True
        )

        for batch in pbar:
            # Handle both (images,) and (images, labels) formats
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)

            # Compute flow matching loss
            loss = self.flow.get_loss(self.model, images, self.device)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        self.losses.append(avg_loss)
        return avg_loss

    def train(self, num_epochs: int, save_path: str | None = None) -> list[float]:
        """
        Run the full training loop.

        Args:
            num_epochs: Number of epochs to train.
            save_path: Optional path to save final checkpoint.

        Returns:
            List of average losses per epoch.
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")

        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(epoch, num_epochs)
            print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

        if save_path:
            self.save_checkpoint(save_path)
            print(f"Saved checkpoint to {save_path}")

        return self.losses

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": self.losses,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.losses = checkpoint.get("losses", [])


# =============================================================================
# Phase 3: Conditional Trainer with CFG Support
# =============================================================================


class ConditionalTrainer:
    """
    Training loop for class-conditional flow matching models.

    Extends the basic Trainer with:
    - Class labels passed to the model
    - Random label dropout for classifier-free guidance (CFG)

    The key difference from unconditional training:
    - We pass labels to the model (and sometimes drop them)
    - The model learns both conditional AND unconditional generation
    - This enables CFG at inference time
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        label_drop_prob: float = 0.1,
        num_classes: int = 10,
        device: torch.device | None = None,
    ):
        """
        Args:
            model: ConditionalDiT neural network to train.
            dataloader: DataLoader providing (images, labels) batches.
            lr: Learning rate.
            weight_decay: AdamW weight decay.
            label_drop_prob: Probability of dropping class label (for CFG).
                            Default 0.1 = 10% of samples trained unconditionally.
            num_classes: Number of classes (10 for MNIST).
            device: Device to train on (auto-detected if None).
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.label_drop_prob = label_drop_prob
        self.num_classes = num_classes

        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.flow = FlowMatching()
        self.losses: list[float] = []

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        """
        Train for one epoch with class conditioning.

        Args:
            epoch: Current epoch number (0-indexed).
            total_epochs: Total number of epochs (for progress display).

        Returns:
            Average loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            leave=True
        )

        for batch in pbar:
            # Expect (images, labels) format
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Compute conditional flow matching loss with CFG dropout
            loss = self.flow.get_conditional_loss(
                model=self.model,
                x_0=images,
                labels=labels,
                device=self.device,
                label_drop_prob=self.label_drop_prob,
                num_classes=self.num_classes,
            )

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        self.losses.append(avg_loss)
        return avg_loss

    def train(self, num_epochs: int, save_path: str | None = None) -> list[float]:
        """
        Run the full training loop.

        Args:
            num_epochs: Number of epochs to train.
            save_path: Optional path to save final checkpoint.

        Returns:
            List of average losses per epoch.
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"CFG label dropout: {self.label_drop_prob*100:.0f}%")

        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(epoch, num_epochs)
            print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

        if save_path:
            self.save_checkpoint(save_path)
            print(f"Saved checkpoint to {save_path}")

        return self.losses

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": self.losses,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.losses = checkpoint.get("losses", [])


# =============================================================================
# Phase 5: VAE Trainer and Latent Diffusion Trainer
# =============================================================================


class VAETrainer:
    """
    Training loop for Variational Autoencoder.

    The VAE is trained to reconstruct images while keeping the latent space
    regularized (close to a standard normal distribution).

    Loss Function
    -------------
    L = L_recon + β × L_KL

    where:
        L_recon = ||x - decode(encode(x))||²   (reconstruction)
        L_KL = KL(q(z|x) || N(0,I))            (regularization)

    For latent diffusion, we use very small β (e.g., 0.00001) to prioritize
    reconstruction quality. The diffusion model handles generation.
    """

    def __init__(
        self,
        model,
        dataloader,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        kl_weight: float = 0.00001,
        device=None,
    ):
        """
        Args:
            model: VAE model to train.
            dataloader: DataLoader providing training batches.
            lr: Learning rate.
            weight_decay: AdamW weight decay.
            kl_weight: Weight for KL divergence term (β).
            device: Device to train on (auto-detected if None).
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.kl_weight = kl_weight

        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.losses: list[float] = []
        self.recon_losses: list[float] = []
        self.kl_losses: list[float] = []

    def train_epoch(self, epoch: int, total_epochs: int) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        pbar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            leave=True
        )

        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)

            # Compute VAE loss
            loss, metrics = self.model.loss(images, kl_weight=self.kl_weight)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track losses
            total_loss += metrics['loss']
            total_recon += metrics['recon_loss']
            total_kl += metrics['kl_loss']
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "recon": f"{metrics['recon_loss']:.4f}",
            })

        avg_metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon / num_batches,
            'kl_loss': total_kl / num_batches,
        }

        self.losses.append(avg_metrics['loss'])
        self.recon_losses.append(avg_metrics['recon_loss'])
        self.kl_losses.append(avg_metrics['kl_loss'])

        return avg_metrics

    def train(self, num_epochs: int, save_path: str | None = None) -> list[float]:
        """Run the full training loop."""
        print(f"Training VAE on {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"KL weight (β): {self.kl_weight}")

        for epoch in range(num_epochs):
            metrics = self.train_epoch(epoch, num_epochs)
            print(f"Epoch {epoch + 1}: loss={metrics['loss']:.4f}, "
                  f"recon={metrics['recon_loss']:.4f}, kl={metrics['kl_loss']:.4f}")

        # Compute scale factor after training
        print("\nComputing latent scale factor...")
        scale = self.model.compute_scale_factor(self.dataloader)
        print(f"Scale factor: {scale:.4f}")

        if save_path:
            self.save_checkpoint(save_path)
            print(f"Saved checkpoint to {save_path}")

        return self.losses

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": self.losses,
            "recon_losses": self.recon_losses,
            "kl_losses": self.kl_losses,
            "scale_factor": self.model.scale_factor.item(),
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.losses = checkpoint.get("losses", [])


class LatentDiffusionTrainer:
    """
    Training loop for latent space flow matching.

    This trainer operates on VAE-encoded latents instead of raw pixels.
    The workflow is:
        1. Encode images with frozen VAE: x → z
        2. Apply flow matching in latent space
        3. Train DiT to predict velocity in latent space

    Key Insight
    -----------
    By working in latent space, we can train on smaller tensors:
        - Pixel space: 64×64×3 = 12,288 dimensions
        - Latent space: 8×8×4 = 256 dimensions (48× smaller!)

    This enables higher resolution generation without proportionally
    increasing compute cost.
    """

    def __init__(
        self,
        model,
        vae,
        dataloader,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        device=None,
    ):
        """
        Args:
            model: DiT model for velocity prediction in latent space.
            vae: Trained VAE (frozen during training).
            dataloader: DataLoader providing training batches.
            lr: Learning rate.
            weight_decay: AdamW weight decay.
            device: Device to train on.
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.vae = vae.to(self.device)
        self.vae.eval()  # VAE is frozen
        for param in self.vae.parameters():
            param.requires_grad = False

        self.dataloader = dataloader

        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.flow = FlowMatching()
        self.losses: list[float] = []

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        """Train for one epoch in latent space."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            leave=True
        )

        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)

            # Encode to latent space (no gradient through VAE)
            with torch.no_grad():
                z_0 = self.vae.encode(images, sample=False)  # Use mean, not sample

            # Flow matching in latent space
            batch_size = z_0.shape[0]
            z_1 = torch.randn_like(z_0)
            t = torch.rand(batch_size, device=self.device)

            # Interpolate in latent space
            t_expand = t.view(-1, 1, 1, 1)
            z_t = (1 - t_expand) * z_0 + t_expand * z_1
            v_target = z_1 - z_0

            # Predict velocity
            v_pred = self.model(z_t, t)

            # MSE loss
            loss = torch.mean((v_pred - v_target) ** 2)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        self.losses.append(avg_loss)
        return avg_loss

    def train(self, num_epochs: int, save_path: str | None = None) -> list[float]:
        """Run the full training loop."""
        print(f"Training Latent Diffusion on {self.device}")
        print(f"DiT parameters: {count_parameters(self.model):,}")
        print(f"VAE parameters: {count_parameters(self.vae):,} (frozen)")

        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(epoch, num_epochs)
            print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

        if save_path:
            self.save_checkpoint(save_path)
            print(f"Saved checkpoint to {save_path}")

        return self.losses

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint (only DiT, not VAE)."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": self.losses,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.losses = checkpoint.get("losses", [])


class LatentConditionalTrainer:
    """
    Training loop for class-conditional latent diffusion.

    Combines latent space flow matching with class conditioning and CFG.
    """

    def __init__(
        self,
        model,
        vae,
        dataloader,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        label_drop_prob: float = 0.1,
        num_classes: int = 10,
        device=None,
    ):
        """
        Args:
            model: ConditionalDiT for latent space.
            vae: Trained VAE (frozen).
            dataloader: DataLoader providing (images, labels).
            lr: Learning rate.
            weight_decay: AdamW weight decay.
            label_drop_prob: Probability of dropping labels for CFG.
            num_classes: Number of classes.
            device: Device to train on.
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.vae = vae.to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        self.dataloader = dataloader
        self.label_drop_prob = label_drop_prob
        self.num_classes = num_classes

        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.losses: list[float] = []

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            leave=True
        )

        for batch in pbar:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Encode to latent space
            with torch.no_grad():
                z_0 = self.vae.encode(images, sample=False)

            batch_size = z_0.shape[0]

            # Sample noise and timesteps
            z_1 = torch.randn_like(z_0)
            t = torch.rand(batch_size, device=self.device)

            # Interpolate
            t_expand = t.view(-1, 1, 1, 1)
            z_t = (1 - t_expand) * z_0 + t_expand * z_1
            v_target = z_1 - z_0

            # Apply label dropout for CFG
            drop_mask = torch.rand(batch_size, device=self.device) < self.label_drop_prob
            labels_with_dropout = labels.clone()
            labels_with_dropout[drop_mask] = self.num_classes  # Null class

            # Predict velocity
            v_pred = self.model(z_t, t, labels_with_dropout)

            # MSE loss
            loss = torch.mean((v_pred - v_target) ** 2)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        self.losses.append(avg_loss)
        return avg_loss

    def train(self, num_epochs: int, save_path: str | None = None) -> list[float]:
        """Run the full training loop."""
        print(f"Training Latent Conditional Diffusion on {self.device}")
        print(f"DiT parameters: {count_parameters(self.model):,}")
        print(f"CFG label dropout: {self.label_drop_prob * 100:.0f}%")

        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(epoch, num_epochs)
            print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

        if save_path:
            self.save_checkpoint(save_path)

        return self.losses

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": self.losses,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.losses = checkpoint.get("losses", [])


# =============================================================================
# Phase 4: Text-Conditional Trainer
# =============================================================================


class TextConditionalTrainer:
    """
    Training loop for text-conditional flow matching models.

    This trainer extends ConditionalTrainer for text prompts instead of class labels.
    It uses a frozen CLIP text encoder to convert prompts to embeddings.

    Key Differences from ConditionalTrainer
    ---------------------------------------
    | Aspect            | ConditionalTrainer    | TextConditionalTrainer    |
    |-------------------|----------------------|---------------------------|
    | Conditioning      | Integer class labels | Text prompts (strings)    |
    | Encoder           | Learnable embedding  | Frozen CLIP               |
    | CFG null          | Learned null class   | Empty string ""           |
    | Memory            | Light                | Heavier (CLIP encoder)    |

    Training Objective
    ------------------
    The loss function is the same flow matching objective, but with text conditioning:

        L = E_{x_0, x_1, t, text}[ ||v_theta(x_t, t, text) - (x_1 - x_0)||^2 ]

    For CFG, we randomly replace text embeddings with null embeddings:

        text_train = {
            CLIP(prompt)  with probability (1 - p_drop)
            CLIP("")      with probability p_drop
        }

    Dataset Requirement
    -------------------
    The dataloader should yield (images, labels) tuples where:
    - images: Tensor of shape (B, C, H, W)
    - labels: Tensor of class indices that can be converted to captions

    We convert labels to captions using a provided caption function.

    Implementation Notes
    --------------------
    1. Text encoding happens once per batch (not per sample)
    2. CLIP weights are frozen - only DiT is trained
    3. Caption dropout is applied at the embedding level
    """

    def __init__(
        self,
        model,
        text_encoder,
        dataloader,
        caption_fn,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        text_drop_prob: float = 0.1,
        device=None,
    ):
        """
        Args:
            model: TextConditionalDiT neural network to train.
            text_encoder: CLIPTextEncoder for encoding prompts.
            dataloader: DataLoader providing (images, labels) batches.
            caption_fn: Function to convert labels to text captions.
                Signature: caption_fn(labels: Tensor) -> list[str]
                Example: lambda labels: [f"a photo of a {CLASSES[l]}" for l in labels]
            lr: Learning rate.
            weight_decay: AdamW weight decay.
            text_drop_prob: Probability of dropping text (using null) for CFG.
                Default 0.1 = 10% of samples trained unconditionally.
            device: Device to train on (auto-detected if None).
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.text_encoder = text_encoder.to(self.device)
        self.dataloader = dataloader
        self.caption_fn = caption_fn
        self.text_drop_prob = text_drop_prob

        # Only optimize DiT parameters - CLIP is frozen
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.flow = FlowMatching()
        self.losses: list[float] = []

        # Pre-compute null text embedding for CFG dropout
        with torch.no_grad():
            null_emb, null_mask = self.text_encoder([""])
            self.null_embedding = null_emb.to(self.device)
            self.null_mask = null_mask.to(self.device)

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        """
        Train for one epoch with text conditioning.

        Training Algorithm
        ------------------
        For each batch (x_0, labels):
            1. Convert labels to captions using caption_fn
            2. Encode captions with CLIP: captions -> embeddings
            3. Apply text dropout: randomly replace some embeddings with null
            4. Sample noise and timesteps: x_1 ~ N(0,I), t ~ U(0,1)
            5. Create interpolation: x_t = (1-t)*x_0 + t*x_1
            6. Compute velocity target: v_target = x_1 - x_0
            7. Predict: v_pred = model(x_t, t, text_embedding, mask)
            8. Loss: ||v_pred - v_target||^2
            9. Backprop and update

        Args:
            epoch: Current epoch number (0-indexed).
            total_epochs: Total number of epochs.

        Returns:
            Average loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            leave=True
        )

        for batch in pbar:
            # Expect (images, labels) format
            images, labels = batch
            images = images.to(self.device)
            batch_size = images.shape[0]

            # Convert labels to captions
            captions = self.caption_fn(labels)

            # Encode captions with CLIP
            with torch.no_grad():
                text_embeddings, text_mask = self.text_encoder(captions)
                text_embeddings = text_embeddings.to(self.device)
                text_mask = text_mask.to(self.device)

            # Apply text dropout for CFG
            drop_mask = torch.rand(batch_size) < self.text_drop_prob
            if drop_mask.any():
                # Expand null embedding to match batch
                null_expanded = self.null_embedding.expand(drop_mask.sum(), -1, -1)
                null_mask_expanded = self.null_mask.expand(drop_mask.sum(), -1)

                text_embeddings[drop_mask] = null_expanded
                text_mask[drop_mask] = null_mask_expanded

            # Compute text-conditional flow matching loss
            loss = self._get_text_conditional_loss(
                images, text_embeddings, text_mask
            )

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        self.losses.append(avg_loss)
        return avg_loss

    def _get_text_conditional_loss(
        self,
        x_0: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the flow matching loss with text conditioning.

        Loss Function
        -------------
        L = ||v_theta(x_t, t, text) - v_target||^2

        where:
            x_t = (1 - t) * x_0 + t * x_1    (linear interpolation)
            v_target = x_1 - x_0             (target velocity)

        Args:
            x_0: Clean images of shape (B, C, H, W).
            text_embeddings: CLIP embeddings of shape (B, M, D).
            text_mask: Attention mask of shape (B, M).

        Returns:
            Scalar loss value.
        """
        batch_size = x_0.shape[0]

        # Sample noise (x_1) and timesteps (t)
        x_1 = torch.randn_like(x_0)
        t = torch.rand(batch_size, device=x_0.device)

        # Create interpolation x_t
        t_expand = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1

        # Target velocity
        v_target = x_1 - x_0

        # Predict velocity
        v_pred = self.model(x_t, t, text_embeddings, text_mask)

        # MSE loss
        loss = torch.mean((v_pred - v_target) ** 2)

        return loss

    def train(self, num_epochs: int, save_path: str | None = None) -> list[float]:
        """
        Run the full training loop.

        Args:
            num_epochs: Number of epochs to train.
            save_path: Optional path to save final checkpoint.

        Returns:
            List of average losses per epoch.
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"CFG text dropout: {self.text_drop_prob*100:.0f}%")

        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(epoch, num_epochs)
            print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

        if save_path:
            self.save_checkpoint(save_path)
            print(f"Saved checkpoint to {save_path}")

        return self.losses

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint (only DiT, not CLIP)."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": self.losses,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.losses = checkpoint.get("losses", [])
