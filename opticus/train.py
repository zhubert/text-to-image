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

from opticus.flow import FlowMatching


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
