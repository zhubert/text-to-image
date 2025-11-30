"""
Text Encoder using CLIP for Text-to-Image Generation

Phase 4 introduces text conditioning, moving from discrete class labels to
natural language prompts. This module wraps a pretrained CLIP text encoder
to convert text prompts into embeddings that guide image generation.

Mathematical Background
-----------------------
CLIP (Contrastive Language-Image Pre-training) learns a joint embedding space
where similar images and texts have high cosine similarity:

    sim(I, T) = cos(f_image(I), f_text(T)) = (f_image(I) · f_text(T)) / (||f_image(I)|| ||f_text(T)||)

The text encoder f_text : String → R^D maps text to a D-dimensional vector.
For CLIP ViT-B/32, D = 512.

Why CLIP Works for Conditioning
-------------------------------
1. **Semantic Alignment**: CLIP embeddings capture semantic meaning, not just syntax
   - "a photo of a dog" and "canine photograph" map to similar vectors

2. **Visual Grounding**: Trained on image-text pairs, embeddings are grounded in visual concepts
   - The model "knows" what visual features correspond to "fluffy" or "red"

3. **Compositionality**: Can handle novel combinations
   - "a blue banana" works even if never seen in training

Architecture
------------
CLIP's text encoder is a Transformer that:
1. Tokenizes text into subword tokens
2. Adds positional embeddings
3. Processes through transformer layers
4. Projects the [EOS] token embedding to the final representation

We use the token-level embeddings (before final projection) for cross-attention,
giving the DiT access to per-token information rather than just a single vector.

Reference: "Learning Transferable Visual Models From Natural Language Supervision"
https://arxiv.org/abs/2103.00020
"""

import torch
import torch.nn as nn
from torch import Tensor


class CLIPTextEncoder(nn.Module):
    """
    Wrapper around a pretrained CLIP text encoder.

    Provides:
    - Text tokenization and encoding
    - Token-level embeddings for cross-attention
    - Pooled embeddings for global conditioning
    - Frozen weights (we don't fine-tune CLIP)

    Mathematical Details
    --------------------
    Given a text prompt T with N tokens, the encoder produces:

    1. Token embeddings: Z = [z_1, z_2, ..., z_N] ∈ R^{N×D}
       - Each z_i captures the contextualized meaning of token i
       - Used for cross-attention in the DiT

    2. Pooled embedding: z_pool = z_[EOS] ∈ R^D
       - The [EOS] token aggregates the full sequence meaning
       - Used for global conditioning (like timestep embedding)

    The token embeddings enable fine-grained conditioning:
    - "a RED dog" → attention focuses on "RED" when generating color
    - "a dog RUNNING" → attention focuses on "RUNNING" for motion
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        device: torch.device | str | None = None,
    ):
        """
        Initialize the CLIP text encoder.

        Args:
            model_name: HuggingFace model identifier for CLIP.
                Options:
                - "openai/clip-vit-base-patch32" (512-dim, fastest)
                - "openai/clip-vit-base-patch16" (512-dim)
                - "openai/clip-vit-large-patch14" (768-dim, best quality)
            max_length: Maximum sequence length for tokenization.
                CLIP was trained with max_length=77.
            device: Device to load the model on.
        """
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self._device = device

        # Lazy loading - models are loaded on first use
        self._tokenizer = None
        self._encoder = None
        self._embed_dim = None

    def _load_model(self):
        """Lazy load the CLIP model and tokenizer."""
        if self._encoder is not None:
            return

        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers"
            )

        # Load tokenizer
        self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)

        # Load text encoder
        self._encoder = CLIPTextModel.from_pretrained(self.model_name)

        # Freeze all parameters - we don't train CLIP
        for param in self._encoder.parameters():
            param.requires_grad = False

        # Get embedding dimension from model config
        self._embed_dim = self._encoder.config.hidden_size

        # Move to device if specified
        if self._device is not None:
            self._encoder = self._encoder.to(self._device)

        self._encoder.eval()

    @property
    def embed_dim(self) -> int:
        """Dimension of the text embeddings (512 for ViT-B, 768 for ViT-L)."""
        self._load_model()
        return self._embed_dim

    @property
    def tokenizer(self):
        """The CLIP tokenizer."""
        self._load_model()
        return self._tokenizer

    def to(self, device):
        """Move encoder to device."""
        self._device = device
        if self._encoder is not None:
            self._encoder = self._encoder.to(device)
        return self

    @torch.no_grad()
    def encode(
        self,
        texts: str | list[str],
        return_pooled: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Encode text prompts into embeddings.

        Args:
            texts: Single text or list of texts to encode.
            return_pooled: If True, also return the pooled (sentence-level) embedding.

        Returns:
            token_embeddings: Shape (B, max_length, embed_dim)
                Per-token embeddings for cross-attention.
            pooled_embedding: Shape (B, embed_dim) [only if return_pooled=True]
                Sentence-level embedding (the [EOS] token representation).

        Mathematical Note
        -----------------
        The token embeddings Z satisfy:

            Z = Transformer(TokenEmbed(tokens) + PosEmbed)

        where TokenEmbed maps each token to its learned embedding vector,
        and PosEmbed adds positional information.
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        tokens = self._tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Move to same device as encoder
        device = next(self._encoder.parameters()).device
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Encode
        outputs = self._encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Token-level embeddings (last hidden state)
        # Shape: (batch_size, max_length, embed_dim)
        token_embeddings = outputs.last_hidden_state

        if return_pooled:
            # Pooled embedding is the [EOS] token representation
            # CLIP puts [EOS] at the position of the last actual token
            # Shape: (batch_size, embed_dim)
            pooled = outputs.pooler_output
            return token_embeddings, pooled

        return token_embeddings

    def get_attention_mask(self, texts: str | list[str]) -> Tensor:
        """
        Get the attention mask for given texts.

        The mask indicates which tokens are real (1) vs padding (0).
        This is used in cross-attention to ignore padding tokens.

        Args:
            texts: Single text or list of texts.

        Returns:
            attention_mask: Shape (B, max_length), dtype bool.
                True for real tokens, False for padding.
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        tokens = self._tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        return tokens["attention_mask"].bool()

    def forward(
        self,
        texts: str | list[str],
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass: encode texts and return both token and pooled embeddings.

        This is the main interface for training. Returns:
        - Token embeddings for cross-attention
        - Attention mask for ignoring padding

        Args:
            texts: Single text or list of texts to encode.

        Returns:
            token_embeddings: Shape (B, max_length, embed_dim)
            attention_mask: Shape (B, max_length), bool
        """
        token_embeddings = self.encode(texts, return_pooled=False)
        attention_mask = self.get_attention_mask(texts)

        # Move mask to same device as embeddings
        attention_mask = attention_mask.to(token_embeddings.device)

        return token_embeddings, attention_mask


class TextProjection(nn.Module):
    """
    Project CLIP embeddings to match the DiT conditioning dimension.

    CLIP embeddings have dimension D_clip (512 or 768), but our DiT
    may use a different conditioning dimension D_cond. This layer
    provides a learnable linear projection:

        z_proj = W @ z_clip + b

    where W ∈ R^{D_cond × D_clip} and b ∈ R^{D_cond}.

    Mathematical Justification
    --------------------------
    The projection serves two purposes:

    1. **Dimension Matching**: Align CLIP's representation space with DiT's

    2. **Domain Adaptation**: CLIP was trained on natural images; our DiT
       may be trained on different data (CIFAR-10, etc.). The projection
       layer can learn to adapt the representations.

    This is analogous to adapter layers in transfer learning.
    """

    def __init__(self, clip_dim: int, cond_dim: int):
        """
        Args:
            clip_dim: CLIP embedding dimension (512 or 768).
            cond_dim: DiT conditioning dimension.
        """
        super().__init__()
        self.proj = nn.Linear(clip_dim, cond_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Project CLIP embeddings to conditioning dimension.

        Args:
            x: CLIP embeddings of shape (..., clip_dim)

        Returns:
            Projected embeddings of shape (..., cond_dim)
        """
        return self.proj(x)


# =============================================================================
# Utility Functions
# =============================================================================


def make_cifar10_captions(class_idx: Tensor | int) -> list[str]:
    """
    Convert CIFAR-10 class indices to text captions.

    This creates the training data for text-conditional generation.
    We use simple templates that CLIP understands well.

    CIFAR-10 Classes:
        0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
        5: dog, 6: frog, 7: horse, 8: ship, 9: truck

    Args:
        class_idx: Single class index or tensor of class indices.

    Returns:
        List of text captions.

    Example:
        >>> make_cifar10_captions(torch.tensor([3, 5, 1]))
        ['a photo of a cat', 'a photo of a dog', 'a photo of an automobile']
    """
    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    # Handle article (a vs an)
    def get_article(word: str) -> str:
        return "an" if word[0] in "aeiou" else "a"

    if isinstance(class_idx, int):
        class_idx = [class_idx]
    elif isinstance(class_idx, Tensor):
        class_idx = class_idx.tolist()

    captions = []
    for idx in class_idx:
        class_name = CIFAR10_CLASSES[idx]
        article = get_article(class_name)
        captions.append(f"a photo of {article} {class_name}")

    return captions


def get_null_text_embedding(
    text_encoder: CLIPTextEncoder,
    batch_size: int = 1,
) -> tuple[Tensor, Tensor]:
    """
    Get the embedding for the "null" (empty) text prompt.

    This is used for classifier-free guidance:
    - Unconditional: model(x, t, null_text_embedding)
    - Conditional: model(x, t, text_embedding)
    - CFG: v_uncond + scale * (v_cond - v_uncond)

    The null prompt is typically just "" (empty string), which CLIP
    processes as [BOS][EOS][PAD][PAD]...

    Args:
        text_encoder: The CLIP text encoder.
        batch_size: Number of null embeddings to return.

    Returns:
        token_embeddings: Shape (batch_size, max_length, embed_dim)
        attention_mask: Shape (batch_size, max_length)
    """
    null_texts = [""] * batch_size
    return text_encoder(null_texts)
