# EfficientNet

```python
from noxton.models import EfficientNet
```

EfficientNet (Tan & Le, 2019) scales depth, width, and resolution in a compound fashion, achieving strong accuracy/efficiency trade-offs. EfficientNetV2 (Tan & Le, 2021) additionally introduces Fused-MBConv blocks for faster training.

---

## Variants

### EfficientNet (V1)

| `model` | Resolution | Params |
|---|---|---|
| `efficientnet_b0` | 224 | 5.3M |
| `efficientnet_b1` | 240 | 7.8M |
| `efficientnet_b2` | 260 | 9.2M |
| `efficientnet_b3` | 300 | 12.2M |
| `efficientnet_b4` | 380 | 19.3M |
| `efficientnet_b5` | 456 | 30.4M |
| `efficientnet_b6` | 528 | 43.0M |
| `efficientnet_b7` | 600 | 66.3M |

### EfficientNetV2

| `model` | Resolution | Params |
|---|---|---|
| `efficientnet_v2_s` | 384 | 21.5M |
| `efficientnet_v2_m` | 480 | 54.1M |
| `efficientnet_v2_l` | 480 | 118.5M |

---

## `from_pretrained`

```python
EfficientNet.from_pretrained(
    model: str,
    weights: str,
    key: PRNGKeyArray = ...,
    dtype = None,
) -> tuple[EfficientNet, eqx.nn.State]
```

### Available weights (selection)

| `weights` | Top-1 |
|---|---|
| `efficientnet_b0_IMAGENET1K_V1` | 77.7% |
| `efficientnet_b1_IMAGENET1K_V2` | 81.4% |
| `efficientnet_b2_IMAGENET1K_V1` | 80.6% |
| `efficientnet_b3_IMAGENET1K_V1` | 82.0% |
| `efficientnet_b4_IMAGENET1K_V1` | 83.4% |
| `efficientnet_b5_IMAGENET1K_V1` | 83.4% |
| `efficientnet_b6_IMAGENET1K_V1` | 84.0% |
| `efficientnet_b7_IMAGENET1K_V1` | 84.1% |
| `efficientnet_v2_s_IMAGENET1K_V1` | 84.2% |
| `efficientnet_v2_m_IMAGENET1K_V1` | 85.1% |
| `efficientnet_v2_l_IMAGENET1K_V1` | 85.7% |

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
from noxton.models import EfficientNet

model, state = EfficientNet.from_pretrained(
    "efficientnet_b0",
    weights="efficientnet_b0_IMAGENET1K_V1",
    dtype=jnp.float16,
)
model, state = eqx.nn.inference_mode((model, state))

images = jnp.zeros((4, 3, 224, 224), dtype=jnp.float16)

logits, _ = eqx.filter_vmap(
    model, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(images, state)

probs = jax.nn.softmax(logits, axis=-1)  # (4, 1000)
```

### Input preprocessing (ImageNet)

```python
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```
