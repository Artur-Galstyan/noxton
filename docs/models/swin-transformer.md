# Swin Transformer

```python
from noxton.models import SwinTransformer
```

Swin Transformer (Liu et al., 2021) introduces a hierarchical vision backbone using *shifted window* attention. Windows partition the feature map so attention is computed locally, then shifted between layers to allow cross-window connections. SwinV2 (Liu et al., 2022) adds continuous relative position bias and other stabilisation improvements.

---

## Variants

### Swin V1

| `model` | Params | Input size | Top-1 |
|---|---|---|---|
| `swin_t` | 28.3M | 224 | 81.5% |
| `swin_s` | 49.6M | 224 | 83.2% |
| `swin_b` | 87.8M | 224 | 83.5% |

### Swin V2

| `model` | Params | Input size | Top-1 |
|---|---|---|---|
| `swin_v2_t` | 28.4M | 256 | 82.0% |
| `swin_v2_s` | 49.7M | 256 | 83.7% |
| `swin_v2_b` | 87.9M | 256 | 84.6% |

!!! note
    V1 and V2 models use different input resolutions. Use `232/224` crop for V1 and `260/256` for V2. See [preprocessing](#input-preprocessing) below.

---

## `from_pretrained`

```python
SwinTransformer.from_pretrained(
    model: str,
    weights: str,
    key: PRNGKeyArray,
    dtype = None,
) -> tuple[SwinTransformer, eqx.nn.State]
```

### Available weights

| `weights` | Top-1 |
|---|---|
| `swin_t_IMAGENET1K_V1` | 81.5% |
| `swin_s_IMAGENET1K_V1` | 83.2% |
| `swin_b_IMAGENET1K_V1` | 83.5% |
| `swin_v2_t_IMAGENET1K_V1` | 82.0% |
| `swin_v2_s_IMAGENET1K_V1` | 83.7% |
| `swin_v2_b_IMAGENET1K_V1` | 84.6% |

---

## `__call__`

```python
model(
    x: Array,                # (3, H, W)
    state: eqx.nn.State,
    key: PRNGKeyArray | None = None,
) -> tuple[Array, eqx.nn.State]   # (1000,), state
```

---

## Example

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from noxton.models import SwinTransformer

# V1 — input 224
model_v1, state_v1 = SwinTransformer.from_pretrained(
    model="swin_t",
    weights="swin_t_IMAGENET1K_V1",
    key=jax.random.key(0),
    dtype=jnp.float32,
)
model_v1, state_v1 = eqx.nn.inference_mode((model_v1, state_v1))

images_v1 = jnp.zeros((2, 3, 224, 224))
logits_v1, _ = eqx.filter_vmap(
    model_v1, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(images_v1, state_v1)

# V2 — input 256
model_v2, state_v2 = SwinTransformer.from_pretrained(
    model="swin_v2_t",
    weights="swin_v2_t_IMAGENET1K_V1",
    key=jax.random.key(0),
    dtype=jnp.float32,
)
model_v2, state_v2 = eqx.nn.inference_mode((model_v2, state_v2))

images_v2 = jnp.zeros((2, 3, 256, 256))
logits_v2, _ = eqx.filter_vmap(
    model_v2, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(images_v2, state_v2)
```

### Input preprocessing

**Swin V1 (224px)**

```python
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

**Swin V2 (256px)**

```python
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(260),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```
