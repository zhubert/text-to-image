# Opticus

An educational project for understanding text-to-image generation using **Flow Matching** and **Diffusion Transformers (DiT)**.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the Phase 1 notebook
uv run jupyter notebook notebooks/01_flow_matching_basics.ipynb
```

## Project Structure

```
opticus/
├── notebooks/
│   └── 01_flow_matching_basics.ipynb  # Phase 1: Unconditional generation
└── opticus/
    ├── flow.py      # Flow matching training logic
    ├── models.py    # Neural network architectures
    ├── sampling.py  # Image generation
    └── train.py     # Training utilities
```

## Phase 1: Unconditional Flow Matching

Generate MNIST digits from pure noise. The core concepts:

- **Forward process**: Linear interpolation from data to noise: `x_t = (1-t)*x_0 + t*x_1`
- **Velocity field**: What the model learns to predict: `v = x_1 - x_0`
- **Sampling**: Start from noise, integrate backward following the velocity

See `notebooks/01_flow_matching_basics.ipynb` for a walkthrough.

## Requirements

- Python 3.12+
- PyTorch 2.0+
- Apple Silicon (MPS), CUDA, or CPU
