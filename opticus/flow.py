"""
Flow Matching Implementation

Flow matching learns to transport samples from a noise distribution to the data distribution
along straight paths. This is mathematically cleaner than DDPM's stochastic approach.

Key concepts:
- Forward process: x_t = (1-t)*x_0 + t*x_1, where x_0 is data, x_1 is noise
- Velocity field: v(x_t, t) = x_1 - x_0 (the direction of the straight path)
- Training: Learn to predict v given x_t and t
- Sampling: Start from noise (t=1), integrate backward to data (t=0)

Reference: https://arxiv.org/abs/2210.02747 (Flow Matching for Generative Modeling)
"""

import torch
from torch import Tensor


class FlowMatching:
    """
    Flow matching training utilities.

    The forward process interpolates between data x_0 and noise x_1:
        x_t = (1 - t) * x_0 + t * x_1

    The target velocity is simply:
        v = x_1 - x_0

    This creates straight paths from data to noise, which the model learns to reverse.
    """

    def __init__(self, sigma_min: float = 0.001):
        """
        Args:
            sigma_min: Small constant for numerical stability at t=0.
                       Prevents exact interpolation to data point.
        """
        self.sigma_min = sigma_min

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """
        Sample random timesteps uniformly from [0, 1].

        Args:
            batch_size: Number of timesteps to sample.
            device: Device to create tensor on.

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1].
        """
        return torch.rand(batch_size, device=device)

    def forward_process(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the noised samples x_t and target velocity v.

        The interpolation formula:
            x_t = (1 - t) * x_0 + t * x_1

        The velocity (derivative w.r.t. t):
            v = dx_t/dt = x_1 - x_0

        Args:
            x_0: Clean data samples, shape (B, C, H, W).
            x_1: Noise samples, shape (B, C, H, W).
            t: Timesteps, shape (B,).

        Returns:
            x_t: Interpolated samples at time t, shape (B, C, H, W).
            velocity: Target velocity field, shape (B, C, H, W).
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, 1, 1)
        t_expanded = t.view(-1, 1, 1, 1)

        # Linear interpolation from data (t=0) to noise (t=1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        # Velocity is constant along the straight path
        velocity = x_1 - x_0

        return x_t, velocity

    def get_loss(
        self,
        model: torch.nn.Module,
        x_0: Tensor,
        device: torch.device,
    ) -> Tensor:
        """
        Compute the flow matching loss for a batch.

        Steps:
        1. Sample noise x_1 ~ N(0, I)
        2. Sample random timesteps t ~ U[0, 1]
        3. Compute x_t via interpolation
        4. Predict velocity with model
        5. MSE loss between predicted and true velocity

        Args:
            model: Neural network that predicts velocity given (x_t, t).
            x_0: Batch of clean data, shape (B, C, H, W).
            device: Device for computations.

        Returns:
            Scalar loss tensor.
        """
        batch_size = x_0.shape[0]

        # Sample noise from standard normal
        x_1 = torch.randn_like(x_0)

        # Sample random timesteps
        t = self.sample_timesteps(batch_size, device)

        # Get interpolated samples and target velocity
        x_t, velocity_target = self.forward_process(x_0, x_1, t)

        # Model predicts velocity
        velocity_pred = model(x_t, t)

        # MSE loss
        loss = torch.mean((velocity_pred - velocity_target) ** 2)

        return loss
