"""
Text-to-Image: Understanding Text-to-Image Generation

An educational project for learning Flow Matching and Diffusion Transformers.
"""

from text_to_image.flow import FlowMatching
from text_to_image.sampling import (
    sample,
    sample_conditional,
    sample_each_class,
    sample_text_conditional,
    sample_text_conditional_batched,
    sample_with_prompt_variations,
    sample_latent,
    sample_latent_conditional,
    sample_latent_each_class,
)
from text_to_image.dit import DiT, ConditionalDiT, TextConditionalDiT
from text_to_image.train import (
    Trainer,
    ConditionalTrainer,
    TextConditionalTrainer,
    VAETrainer,
    LatentDiffusionTrainer,
    LatentConditionalTrainer,
    get_device,
)
from text_to_image.text_encoder import CLIPTextEncoder, make_cifar10_captions
from text_to_image.vae import VAE, SmallVAE

__version__ = "0.1.0"
__all__ = [
    # Flow Matching
    "FlowMatching",
    # Models
    "DiT",
    "ConditionalDiT",
    "TextConditionalDiT",
    # VAE (Phase 5)
    "VAE",
    "SmallVAE",
    # Text Encoding
    "CLIPTextEncoder",
    "make_cifar10_captions",
    # Sampling
    "sample",
    "sample_conditional",
    "sample_each_class",
    "sample_text_conditional",
    "sample_text_conditional_batched",
    "sample_with_prompt_variations",
    "sample_latent",
    "sample_latent_conditional",
    "sample_latent_each_class",
    # Training
    "Trainer",
    "ConditionalTrainer",
    "TextConditionalTrainer",
    "VAETrainer",
    "LatentDiffusionTrainer",
    "LatentConditionalTrainer",
    "get_device",
]
