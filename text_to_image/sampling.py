"""
Sampling (Generation) for Flow Matching Models

To generate images, we start from pure noise (t=1) and integrate the learned
velocity field backward to t=0 (clean data).

The ODE to solve:
    dx/dt = v(x, t)

We integrate from t=1 to t=0 using Euler method or higher-order solvers.

Phase 3 adds:
- Class-conditional sampling
- Classifier-Free Guidance (CFG) for stronger conditioning

Phase 4 adds:
- Text-conditional sampling with CLIP embeddings
- CFG for text prompts
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


# =============================================================================
# Phase 3: Class-Conditional Sampling with Classifier-Free Guidance
# =============================================================================


@torch.no_grad()
def sample_conditional(
    model: nn.Module,
    class_labels: Tensor | int | list[int],
    image_shape: tuple[int, ...],
    num_steps: int = 50,
    cfg_scale: float = 3.0,
    device: torch.device | None = None,
    num_classes: int = 10,
) -> Tensor:
    """
    Generate class-conditional samples using Classifier-Free Guidance (CFG).

    CFG works by running the model twice at each step:
    1. Conditional: v_cond = model(x_t, t, class_label)
    2. Unconditional: v_uncond = model(x_t, t, null_class)

    Then blending: v = v_uncond + cfg_scale * (v_cond - v_uncond)

    Intuition:
    - (v_cond - v_uncond) represents "what the class adds" to the prediction
    - Scaling this difference amplifies the class-specific features
    - cfg_scale=1.0 = pure conditional (no guidance)
    - cfg_scale>1.0 = stronger adherence to class (but may reduce diversity)
    - cfg_scale=0.0 = pure unconditional

    Typical values: cfg_scale ∈ [2.0, 7.0], with 3.0-5.0 being common.

    Args:
        model: Trained ConditionalDiT model.
        class_labels: Target class(es) to generate. Can be:
            - int: generate multiple samples of this class
            - Tensor of shape (B,): batch of class labels
            - list of ints: batch of class labels
        image_shape: Shape of each image (C, H, W).
        num_steps: Number of integration steps.
        cfg_scale: Classifier-free guidance scale.
            - 1.0 = no guidance (pure conditional)
            - >1.0 = stronger conditioning
            - 3.0-5.0 is typical
        device: Device to run on.
        num_classes: Number of classes (for null class index).

    Returns:
        Generated images of shape (B, C, H, W).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Handle different input formats for class labels
    if isinstance(class_labels, int):
        # Single class, infer batch size from model or use 1
        class_labels = torch.tensor([class_labels], device=device)
    elif isinstance(class_labels, list):
        class_labels = torch.tensor(class_labels, device=device)
    else:
        class_labels = class_labels.to(device)

    num_samples = class_labels.shape[0]

    # Create null class labels for unconditional prediction
    null_labels = torch.full(
        (num_samples,), num_classes,
        dtype=torch.long, device=device
    )

    # Start from pure noise at t=1
    x = torch.randn(num_samples, *image_shape, device=device)

    # Integration timesteps from t=1 to t=0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    dt = 1.0 / num_steps

    # Euler integration with CFG
    for i in tqdm(range(num_steps), desc="Sampling (CFG)", leave=False):
        t = timesteps[i]
        t_batch = torch.full((num_samples,), t, device=device)

        # === Classifier-Free Guidance ===
        # 1. Conditional prediction (with class label)
        v_cond = model(x, t_batch, class_labels)

        # 2. Unconditional prediction (with null class)
        v_uncond = model(x, t_batch, null_labels)

        # 3. CFG blending
        # v = v_uncond + scale * (v_cond - v_uncond)
        # When scale=1: v = v_cond (no guidance)
        # When scale>1: amplify what the class "adds"
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Euler step (going backward in time)
        x = x - dt * v

    return x


@torch.no_grad()
def sample_each_class(
    model: nn.Module,
    num_per_class: int = 1,
    image_shape: tuple[int, ...] = (1, 28, 28),
    num_steps: int = 50,
    cfg_scale: float = 3.0,
    device: torch.device | None = None,
    num_classes: int = 10,
) -> Tensor:
    """
    Generate samples for each class (useful for visualization).

    Args:
        model: Trained ConditionalDiT model.
        num_per_class: Number of samples to generate per class.
        image_shape: Shape of each image (C, H, W).
        num_steps: Number of integration steps.
        cfg_scale: Classifier-free guidance scale.
        device: Device to run on.
        num_classes: Number of classes.

    Returns:
        Generated images of shape (num_classes * num_per_class, C, H, W).
        Organized as [class_0_sample_0, class_0_sample_1, ..., class_1_sample_0, ...].
    """
    if device is None:
        device = next(model.parameters()).device

    # Create labels: [0, 0, ..., 1, 1, ..., 2, 2, ..., 9, 9, ...]
    labels = torch.repeat_interleave(
        torch.arange(num_classes, device=device),
        num_per_class
    )

    return sample_conditional(
        model=model,
        class_labels=labels,
        image_shape=image_shape,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        device=device,
        num_classes=num_classes,
    )


# =============================================================================
# Phase 4: Text-Conditional Sampling
# =============================================================================


@torch.no_grad()
def sample_text_conditional(
    model: nn.Module,
    text_encoder,
    prompts: str | list[str],
    image_shape: tuple[int, ...],
    num_steps: int = 50,
    cfg_scale: float = 7.5,
    device: torch.device | None = None,
) -> Tensor:
    """
    Generate images from text prompts using Classifier-Free Guidance.

    Mathematical Framework
    ----------------------
    Text-conditional generation uses the same CFG formulation as class-conditional,
    but with text embeddings instead of class embeddings:

        v_cfg = v_uncond + w * (v_cond - v_uncond)

    where:
        v_cond = model(x_t, t, text_embedding)
        v_uncond = model(x_t, t, null_text_embedding)

    The null text embedding comes from encoding an empty string "".

    Implementation Details
    ----------------------
    At each sampling step:

    1. Encode prompts: text -> CLIP -> token_embeddings in R^{B x M x D}
    2. Encode null text: "" -> CLIP -> null_embeddings in R^{B x M x D}
    3. Get v_cond = model(x_t, t, token_embeddings, mask)
    4. Get v_uncond = model(x_t, t, null_embeddings, null_mask)
    5. Apply CFG: v = v_uncond + scale * (v_cond - v_uncond)
    6. Euler step: x_{t-dt} = x_t - dt * v

    Guidance Scale for Text
    -----------------------
    Text-conditional models typically use higher CFG scales than class-conditional:

    | Scale | Effect |
    |-------|--------|
    | 1.0   | Pure conditional (often blurry) |
    | 3-5   | Mild guidance |
    | 7-8   | Standard for text-to-image |
    | 10-15 | Strong adherence (may oversaturate) |

    The higher scales are needed because text conditioning is more complex
    and the unconditional baseline is less informative.

    Args:
        model: Trained TextConditionalDiT model.
        text_encoder: CLIPTextEncoder instance.
        prompts: Text prompt(s) for generation.
            - str: Generate one image
            - list[str]: Generate one image per prompt
        image_shape: Shape of each image (C, H, W).
        num_steps: Number of integration steps.
        cfg_scale: Classifier-free guidance scale.
            - 7.5 is a good starting point for text-to-image
            - Higher = stronger text adherence, lower diversity
        device: Device to run on.

    Returns:
        Generated images of shape (B, C, H, W).

    Example:
        >>> samples = sample_text_conditional(
        ...     model, text_encoder,
        ...     prompts=["a photo of a cat", "a photo of a dog"],
        ...     image_shape=(3, 32, 32),
        ...     cfg_scale=7.5,
        ... )
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Handle single prompt
    if isinstance(prompts, str):
        prompts = [prompts]

    num_samples = len(prompts)

    # Encode text prompts
    text_embeddings, text_mask = text_encoder(prompts)
    text_embeddings = text_embeddings.to(device)
    text_mask = text_mask.to(device)

    # Encode null text for unconditional
    null_texts = [""] * num_samples
    null_embeddings, null_mask = text_encoder(null_texts)
    null_embeddings = null_embeddings.to(device)
    null_mask = null_mask.to(device)

    # Start from pure noise at t=1
    x = torch.randn(num_samples, *image_shape, device=device)

    # Integration timesteps from t=1 to t=0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    dt = 1.0 / num_steps

    # Euler integration with CFG
    for i in tqdm(range(num_steps), desc="Sampling (Text CFG)", leave=False):
        t = timesteps[i]
        t_batch = torch.full((num_samples,), t, device=device)

        # === Classifier-Free Guidance ===
        # 1. Conditional prediction (with text)
        v_cond = model(x, t_batch, text_embeddings, text_mask)

        # 2. Unconditional prediction (with null text)
        v_uncond = model(x, t_batch, null_embeddings, null_mask)

        # 3. CFG blending
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Euler step (going backward in time)
        x = x - dt * v

    return x


@torch.no_grad()
def sample_text_conditional_batched(
    model: nn.Module,
    text_encoder,
    prompts: str | list[str],
    image_shape: tuple[int, ...],
    num_steps: int = 50,
    cfg_scale: float = 7.5,
    device: torch.device | None = None,
) -> Tensor:
    """
    Efficient text-conditional sampling with batched CFG.

    This version batches the conditional and unconditional forward passes
    together for better GPU utilization. For a batch of N prompts, we
    run a single forward pass with 2N samples.

    Mathematical Equivalence
    ------------------------
    This computes the same result as sample_text_conditional but more efficiently:

        # Batched approach
        x_batch = [x, x]           # Duplicate inputs
        c_batch = [c_cond, c_null] # Stack embeddings
        v_batch = model(x_batch, t, c_batch)
        v_cond, v_uncond = v_batch.chunk(2)

    This reduces the number of forward passes from 2N to N (by doubling batch size).

    Computational Comparison
    ------------------------
    | Method     | Forward Passes | Memory  | Latency |
    |------------|----------------|---------|---------|
    | Sequential | 2 per step     | 1x      | 2x      |
    | Batched    | 1 per step     | 2x      | 1x      |

    Use sequential for memory-limited settings, batched for speed.

    Args:
        model: Trained TextConditionalDiT model.
        text_encoder: CLIPTextEncoder instance.
        prompts: Text prompt(s) for generation.
        image_shape: Shape of each image (C, H, W).
        num_steps: Number of integration steps.
        cfg_scale: Classifier-free guidance scale.
        device: Device to run on.

    Returns:
        Generated images of shape (B, C, H, W).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Handle single prompt
    if isinstance(prompts, str):
        prompts = [prompts]

    num_samples = len(prompts)

    # Encode text prompts and null text
    text_embeddings, text_mask = text_encoder(prompts)
    null_texts = [""] * num_samples
    null_embeddings, null_mask = text_encoder(null_texts)

    # Stack for batched forward pass
    # Shape: (2 * num_samples, max_length, embed_dim)
    context_batch = torch.cat([text_embeddings, null_embeddings], dim=0).to(device)
    mask_batch = torch.cat([text_mask, null_mask], dim=0).to(device)

    # Start from pure noise at t=1
    x = torch.randn(num_samples, *image_shape, device=device)

    # Integration timesteps from t=1 to t=0
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    dt = 1.0 / num_steps

    # Euler integration with batched CFG
    for i in tqdm(range(num_steps), desc="Sampling (Batched CFG)", leave=False):
        t = timesteps[i]

        # Duplicate x for batched forward pass
        x_batch = torch.cat([x, x], dim=0)  # (2 * num_samples, C, H, W)
        t_batch = torch.full((2 * num_samples,), t, device=device)

        # Single batched forward pass
        v_batch = model(x_batch, t_batch, context_batch, mask_batch)

        # Split into conditional and unconditional
        v_cond, v_uncond = v_batch.chunk(2, dim=0)

        # CFG blending
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Euler step
        x = x - dt * v

    return x


@torch.no_grad()
def sample_with_prompt_variations(
    model: nn.Module,
    text_encoder,
    base_prompt: str,
    variations: list[str],
    image_shape: tuple[int, ...],
    num_steps: int = 50,
    cfg_scale: float = 7.5,
    device: torch.device | None = None,
) -> Tensor:
    """
    Generate images from a base prompt with variations.

    This is useful for exploring how different modifiers affect generation:
    - "a photo of a cat" + ["sitting", "running", "sleeping"]
    - "a {color} car" where color varies

    Args:
        model: Trained TextConditionalDiT model.
        text_encoder: CLIPTextEncoder instance.
        base_prompt: Base prompt to modify.
        variations: List of variations to append/substitute.
        image_shape: Shape of each image (C, H, W).
        num_steps: Number of integration steps.
        cfg_scale: Classifier-free guidance scale.
        device: Device to run on.

    Returns:
        Generated images of shape (len(variations), C, H, W).

    Example:
        >>> samples = sample_with_prompt_variations(
        ...     model, text_encoder,
        ...     base_prompt="a photo of a {} cat",
        ...     variations=["fluffy", "striped", "black"],
        ...     image_shape=(3, 32, 32),
        ... )
    """
    # Create full prompts
    if "{}" in base_prompt:
        prompts = [base_prompt.format(v) for v in variations]
    else:
        prompts = [f"{base_prompt} {v}" for v in variations]

    return sample_text_conditional(
        model=model,
        text_encoder=text_encoder,
        prompts=prompts,
        image_shape=image_shape,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        device=device,
    )
