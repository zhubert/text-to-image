# Text-to-Image: Understanding Text-to-Image Generation

An educational project to learn how AI generates images from text prompts, using **Flow Matching** and **Diffusion Transformers (DiT)**.

## Background

Modern text-to-image models (Stable Diffusion 3, DALL-E 3, etc.) use:
- **Diffusion Transformers (DiT)**: Transformer architecture operating on image patches
- **Flow Matching**: Clean training objective (straight paths from noise to data)
- **Text Conditioning**: CLIP embeddings guide generation via cross-attention
- **Latent Space**: VAE compression for efficiency at scale

This project builds understanding progressively, starting simple and adding complexity.

---

## Phase 1: Unconditional Flow Matching ✓

**Goal**: Generate images from noise (no conditioning yet)

### What You'll Learn
- How flow matching works (straight paths from noise → data)
- The velocity field and how to train a network to predict it
- Sampling (walking along the learned flow)

### Implementation
1. Dataset: MNIST (28×28 grayscale digits)
2. Forward process: Linear interpolation between data and noise
3. Model: Small CNN or simple transformer
4. Training: Predict velocity at random timesteps
5. Sampling: Start from noise, follow velocity field to generate image

### Outcome
Generate random (blurry) digits from pure noise.

---

## Phase 2: Add the DiT Architecture ✓

**Goal**: Replace simple CNN with a proper Diffusion Transformer

### What You'll Learn
- Patchifying images (treating image as sequence of patches)
- Positional embeddings for 2D
- Timestep conditioning via adaptive layer norm (adaLN)

### Implementation
1. Split 28×28 image into patches (e.g., 7×7 patches of 4×4 pixels)
2. Embed patches + add positional encoding
3. Transformer blocks with timestep modulation
4. Predict velocity field

### Outcome
Sharper generated digits, understand DiT architecture.

---

## Phase 3: Class-Conditional Generation ✓

**Goal**: Control what digit gets generated

### What You'll Learn
- How conditioning information guides generation
- Classifier-free guidance (CFG) - the trick that makes generations pop

### Implementation
1. Add class embedding (0-9) to timestep embedding via `ClassEmbedding`
2. During training: randomly drop class label 10% of time (for CFG)
3. During sampling: blend conditional and unconditional predictions

### Key Components
- `ConditionalDiT`: DiT with class embedding support
- `ClassEmbedding`: Learnable embeddings for each class + null class
- `get_conditional_loss()`: Training loss with label dropout
- `sample_conditional()`: CFG sampling
- `sample_each_class()`: Generate samples for all classes

### CFG Formula
```
v_guided = v_uncond + scale × (v_cond - v_uncond)
```
- `scale = 1.0`: Pure conditional (no guidance)
- `scale = 3.0-5.0`: Typical range for good results
- `scale > 7.0`: May oversaturate

### Outcome
"Generate a 7" → produces a 7.

---

## Phase 4: Text Conditioning with CLIP ✓

**Goal**: Move from class labels to actual text prompts

### What You'll Learn
- How text encoders convert words to vectors
- Cross-attention between image patches and text tokens
- Why CLIP embeddings work for conditioning

### Implementation
1. Load pretrained CLIP text encoder (frozen)
2. Add cross-attention layers to DiT blocks
3. Dataset: CIFAR-10 (32×32 color) with class names as captions
   - "a photo of a dog", "a photo of an airplane", etc.
4. Train DiT to attend to text embeddings

### Key Components
- `CLIPTextEncoder`: Wrapper around HuggingFace CLIP text encoder
- `CrossAttention`: Attention from image patches to text tokens
- `TextConditionedDiTBlock`: DiT block with self-attention + cross-attention + MLP
- `TextConditionalDiT`: Full model with cross-attention to CLIP embeddings
- `TextConditionalTrainer`: Training with text dropout for CFG
- `sample_text_conditional()`: Generation with text prompts and CFG

### Cross-Attention Formula
```
CrossAttn(X, Z) = softmax(Q @ K^T / sqrt(d)) @ V

where:
  Q = X @ W_Q  (queries from image patches)
  K = Z @ W_K  (keys from text tokens)
  V = Z @ W_V  (values from text tokens)
```

### Outcome
"a photo of a cat" → produces cat-like image.

---

## Phase 5 (Optional): Latent Space with VAE

**Goal**: Scale to larger images efficiently

### What You'll Learn
- Why pixel space doesn't scale
- How VAEs compress images
- The full Stable Diffusion-style pipeline

### Implementation
1. Train or use pretrained VAE
2. Encode images → latent space
3. Run flow matching in latent space
4. Decode generated latents → images
5. Scale to 64×64 or 128×128

### Outcome
Higher resolution generations, complete understanding of modern text-to-image.

---

## Project Structure

```
text-to-image/
├── README.md
├── PROJECT.md
├── pyproject.toml
├── notebooks/
│   ├── 01_flow_matching_basics.ipynb
│   ├── 02_diffusion_transformer.ipynb
│   ├── 03_class_conditioning.ipynb
│   ├── 04_text_conditioning.ipynb
│   └── 05_latent_diffusion.ipynb
└── text_to_image/
    ├── __init__.py
    ├── flow.py          # Flow matching logic
    ├── dit.py           # DiT architecture
    ├── text_encoder.py  # CLIP wrapper
    ├── vae.py           # VAE (Phase 5)
    ├── sampling.py      # Generation loop
    └── train.py         # Training utilities
```

Each phase has a notebook for exploration + reusable code in the package.

---

## Technical Details

### Environment
- **Framework**: PyTorch
- **Hardware**: Apple Silicon (MPS)
- **Python**: 3.11+

### Key Concepts

**Flow Matching** (vs DDPM):
- DDPM: Stochastic paths, predict noise ε
- Flow Matching: Deterministic straight paths, predict velocity v
- Flow Matching is mathematically cleaner and often faster to train

**Diffusion Transformer (DiT)**:
- Replaces U-Net with transformer blocks
- Processes image as sequence of patches
- Uses adaptive layer norm (adaLN) for timestep/class conditioning
- Scales better with compute

**Classifier-Free Guidance (CFG)**:
- Train with conditioning dropped randomly
- At inference: blend conditional and unconditional predictions
- `output = unconditional + scale * (conditional - unconditional)`
- Higher scale = stronger adherence to prompt

**VAE / Latent Space**:
- Encoder compresses image (e.g., 256×256×3 → 32×32×4)
- Diffusion happens in compact latent space
- Decoder reconstructs full image
- 48× fewer dimensions = much faster training

---

## References

- [MIT 6.S184: Flow Matching and Diffusion Models (2025)](https://diffusion.csail.mit.edu/)
- [Meta's Flow Matching Guide](https://ai.meta.com/research/publications/flow-matching-guide-and-code/)
- [Scalable Diffusion Models with Transformers (DiT paper)](https://arxiv.org/abs/2212.09748)
- [Stable Diffusion 3 Technical Report](https://stability.ai/news/stable-diffusion-3)
