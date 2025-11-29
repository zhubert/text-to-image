"""
Sampling (Generation) for Flow Matching Models

To generate images, we start from pure noise (t=1) and integrate the learned
velocity field backward to t=0 (clean data).

The ODE to solve:
    dx/dt = v(x, t)

We integrate from t=1 to t=0 using Euler method or higher-order solvers.
"""

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm


@torch.no_grad()
def sample(
    model: nn.Module,
    num_samples: int,
    image_shape: tuple[int, ...],
    num_steps: int = 50,
    device: torch.device | None = None,
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    """
    Generate samples by integrating the learned velocity field.

    Uses Euler method to solve the ODE from t=1 (noise) to t=0 (data):
        x_{t-dt} = x_t - dt * v(x_t, t)

    Args:
        model: Trained velocity prediction model.
        num_samples: Number of images to generate.
        image_shape: Shape of each image (C, H, W), e.g., (1, 28, 28) for MNIST.
        num_steps: Number of integration steps (more = better quality, slower).
        device: Device to run on (auto-detected if None).
        return_trajectory: If True, also return intermediate samples.

    Returns:
        Generated images of shape (num_samples, C, H, W).
        If return_trajectory=True, also returns list of intermediate samples.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Start from pure noise at t=1
    x = torch.randn(num_samples, *image_shape, device=device)

    # Integration timesteps from t=1 to t=0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    dt = 1.0 / num_steps

    trajectory = [x.clone()] if return_trajectory else []

    # Euler integration
    for i in tqdm(range(num_steps), desc="Sampling", leave=False):
        t = timesteps[i]

        # Current timestep for all samples
        t_batch = torch.full((num_samples,), t, device=device)

        # Predict velocity at current point
        v = model(x, t_batch)

        # Euler step (going backward in time, so we subtract)
        x = x - dt * v

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return x, trajectory
    return x


@torch.no_grad()
def sample_rk4(
    model: nn.Module,
    num_samples: int,
    image_shape: tuple[int, ...],
    num_steps: int = 20,
    device: torch.device | None = None,
) -> Tensor:
    """
    Generate samples using 4th-order Runge-Kutta integration.

    RK4 is more accurate than Euler, allowing fewer steps for same quality.
    This is useful for faster sampling with trained models.

    Args:
        model: Trained velocity prediction model.
        num_samples: Number of images to generate.
        image_shape: Shape of each image (C, H, W).
        num_steps: Number of integration steps.
        device: Device to run on.

    Returns:
        Generated images of shape (num_samples, C, H, W).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Start from pure noise at t=1
    x = torch.randn(num_samples, *image_shape, device=device)

    dt = 1.0 / num_steps

    for step in tqdm(range(num_steps), desc="Sampling (RK4)", leave=False):
        t = 1.0 - step * dt

        t_batch = torch.full((num_samples,), t, device=device)
        t_half = torch.full((num_samples,), t - 0.5 * dt, device=device)
        t_next = torch.full((num_samples,), t - dt, device=device)

        # RK4 stages (note: we're integrating backward, so negate velocities)
        k1 = -model(x, t_batch)
        k2 = -model(x + 0.5 * dt * k1, t_half)
        k3 = -model(x + 0.5 * dt * k2, t_half)
        k4 = -model(x + dt * k3, t_next)

        # RK4 update
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x
