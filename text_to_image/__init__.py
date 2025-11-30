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
)
from text_to_image.dit import DiT, ConditionalDiT, TextConditionalDiT
from text_to_image.train import Trainer, ConditionalTrainer, TextConditionalTrainer, get_device
from text_to_image.text_encoder import CLIPTextEncoder, make_cifar10_captions

__version__ = "0.1.0"
__all__ = [
    # Flow Matching
    "FlowMatching",
    # Models
    "DiT",
    "ConditionalDiT",
    "TextConditionalDiT",
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
    # Training
    "Trainer",
    "ConditionalTrainer",
    "TextConditionalTrainer",
    "get_device",
]
