# Text-to-Image

An educational project for understanding text-to-image generation using **Flow Matching** and **Diffusion Transformers (DiT)**.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the notebooks
uv run jupyter notebook notebooks/
```

## Project Structure

```
text-to-image/
├── notebooks/
│   ├── 01_flow_matching_basics.ipynb  # Phase 1: Unconditional generation
│   ├── 02_diffusion_transformer.ipynb # Phase 2: DiT architecture
│   ├── 03_class_conditioning.ipynb    # Phase 3: Class-conditional + CFG
│   └── 04_text_conditioning.ipynb     # Phase 4: Text-conditional + CLIP
└── text_to_image/
    ├── flow.py         # Flow matching training logic
    ├── dit.py          # DiT, ConditionalDiT, TextConditionalDiT
    ├── text_encoder.py # CLIP text encoder wrapper
    ├── models.py       # CNN/U-Net architectures
    ├── sampling.py     # Image generation (unconditional, class, text CFG)
    └── train.py        # Training utilities
```

## Phases

### Phase 1: Unconditional Flow Matching

Generate random MNIST digits from pure noise.

- **Forward process**: Linear interpolation from data to noise: `x_t = (1-t)*x_0 + t*x_1`
- **Velocity field**: What the model learns to predict: `v = x_1 - x_0`
- **Sampling**: Start from noise, integrate backward following the velocity

### Phase 2: Diffusion Transformer (DiT)

Replace the CNN with a transformer architecture.

- **Patchification**: Images split into patches treated as tokens
- **Positional embeddings**: 2D sinusoidal encodings for spatial awareness
- **Adaptive Layer Norm (adaLN)**: Timestep conditions every layer dynamically
- **Self-attention**: Global receptive field for better coherence

### Phase 3: Class-Conditional Generation

Control which digit gets generated using class labels and Classifier-Free Guidance (CFG).

- **Class embeddings**: Learnable vectors for each digit (0-9)
- **Label dropout**: Train with 10% random label dropping for CFG
- **CFG sampling**: Blend conditional and unconditional predictions
- **CFG formula**: `v = v_uncond + scale × (v_cond - v_uncond)`

```python
from text_to_image import ConditionalDiT, sample_conditional

model = ConditionalDiT(num_classes=10)
# ... train model ...

# Generate specific digits with CFG
samples = sample_conditional(
    model,
    class_labels=[7, 7, 7, 7],  # Generate four 7s
    cfg_scale=4.0
)
```

### Phase 4: Text Conditioning with CLIP

Move from class labels to natural language prompts using CLIP embeddings.

- **CLIP encoder**: Frozen pretrained text encoder converts prompts to embeddings
- **Cross-attention**: Image patches attend to text tokens for fine-grained control
- **Text dropout**: 10% dropout enables CFG with null text embeddings
- **CFG formula**: Same as class-conditional, but with text embeddings

```python
from text_to_image import TextConditionalDiT, CLIPTextEncoder, sample_text_conditional

text_encoder = CLIPTextEncoder()
model = TextConditionalDiT(context_dim=512)
# ... train model ...

# Generate from text prompts with CFG
samples = sample_text_conditional(
    model, text_encoder,
    prompts=["a photo of a cat", "a photo of a dog"],
    cfg_scale=7.5
)
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- Apple Silicon (MPS), CUDA, or CPU
