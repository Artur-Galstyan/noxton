# CLIP

```python
from noxton.models import CLIP
from noxton.models.clip import clip_tokenize
```

CLIP (Radford et al., 2021) trains an image encoder and a text encoder jointly with a contrastive objective, enabling zero-shot image classification and image–text similarity scoring. Noxton includes the original OpenAI pretrained weights.

---

## Variants

| `model` | Image encoder | Image size | Top-1 (zero-shot) |
|---|---|---|---|
| `RN50` | ModifiedResNet-50 | 224 | 59.8% |
| `RN101` | ModifiedResNet-101 | 224 | 62.3% |
| `RN50x4` | ModifiedResNet-50×4 | 288 | 66.2% |
| `RN50x16` | ModifiedResNet-50×16 | 384 | 70.8% |
| `RN50x64` | ModifiedResNet-50×64 | 448 | 75.3% |
| `ViT-B/32` | ViT-Base / patch 32 | 224 | 63.3% |
| `ViT-B/16` | ViT-Base / patch 16 | 224 | 68.3% |
| `ViT-L/14` | ViT-Large / patch 14 | 224 | 75.5% |
| `ViT-L/14@336px` | ViT-Large / patch 14 | 336 | 76.6% |

---

## `from_pretrained`

```python
CLIP.from_pretrained(
    model: str,
    dtype = None,
) -> tuple[CLIP, eqx.nn.State]
```

Weights are the original OpenAI releases downloaded from Azure CDN and cached at `~/.noxton/`. No `key` argument is needed (weights are fully determined by the pretrained checkpoint).

---

## `__call__`

```python
clip(
    image: Array,            # (3, H, W) — single image
    text: Array,             # (77,) int32 token ids — single caption
    state: eqx.nn.State,
    key: PRNGKeyArray | None = None,
) -> tuple[Array, Array, eqx.nn.State]
    # (logits_per_image,), (logits_per_text,), state
```

Use `eqx.filter_vmap` over `text` to score one image against multiple captions (see example).

### `encode_image`

```python
clip.encode_image(
    image: Array,            # (3, H, W)
    state: eqx.nn.State,
) -> tuple[Array, eqx.nn.State]  # (embed_dim,), state
```

### `encode_text`

```python
clip.encode_text(
    text: Array,             # (77,)
    state: eqx.nn.State,
) -> tuple[Array, eqx.nn.State]  # (embed_dim,), state
```

---

## Tokenization

```python
from noxton.models.clip import clip_tokenize

tokens = clip_tokenize(["a photo of a cat", "a photo of a dog"])
# tokens.shape -> (2, 77), dtype int32
# sequences are padded / truncated to 77 tokens
```

---

## Example: zero-shot classification

```python
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from PIL import Image
from noxton.models import CLIP
from noxton.models.clip import clip_tokenize

# --- Load model ---
clip, state = CLIP.from_pretrained(model="ViT-B/32", dtype=jnp.float16)
clip, state = eqx.nn.inference_mode((clip, state))

# --- Prepare inputs ---
def preprocess_image(path, n_px=224):
    img = Image.open(path).convert("RGB")
    img = img.resize((n_px, n_px), Image.BICUBIC)
    x = jnp.array(np.array(img), dtype=jnp.float16) / 255.0
    x = jnp.transpose(x, (2, 0, 1))   # HWC -> CHW
    mean = jnp.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std  = jnp.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    return (x - mean) / std

image = preprocess_image("cat.jpg")
text  = clip_tokenize(["a photo of a human", "a photo of a cat", "a photo of a dog"])

# --- Run model: vmap over text candidates ---
logits_per_image, logits_per_text, state = eqx.filter_vmap(
    clip,
    in_axes=(None, 0, None),   # broadcast image, map over text
    out_axes=(0, 0, None),
    axis_name="batch",
)(image, text, state)

probs = jax.nn.softmax(logits_per_image)
for label, p in zip(["human", "cat", "dog"], probs):
    print(f"{label}: {p * 100:.1f}%")
```
