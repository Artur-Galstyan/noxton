# ConvNeXt

```python
from noxton.models import ConvNeXt
```

ConvNeXt (Liu et al., 2022) modernises the standard ResNet recipe with design choices from Vision Transformers: depthwise convolutions, inverted bottlenecks, fewer normalisation layers (LayerNorm only), and GELU activations. The result is a pure-CNN architecture that is competitive with Swin Transformers.

---

## Variants

| `model` | Params | ImageNet Top-1 |
|---|---|---|
| `convnext_tiny` | 28.6M | 82.1% |
| `convnext_small` | 50.2M | 83.1% |
| `convnext_base` | 88.6M | 84.1% |
| `convnext_large` | 197.8M | 84.3% |

---

## `from_pretrained`

```python
ConvNeXt.from_pretrained(
    model: str,
    weights: str,
    key: PRNGKeyArray,
    dtype = None,
) -> tuple[ConvNeXt, eqx.nn.State]
```

### Available weights

| `weights` | Top-1 |
|---|---|
| `convnext_tiny_IMAGENET1K_V1` | 82.1% |
| `convnext_small_IMAGENET1K_V1` | 83.1% |
| `convnext_base_IMAGENET1K_V1` | 84.1% |
| `convnext_large_IMAGENET1K_V1` | 84.3% |

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
from noxton.models import ConvNeXt

model, state = ConvNeXt.from_pretrained(
    model="convnext_base",
    weights="convnext_base_IMAGENET1K_V1",
    key=jax.random.key(0),
    dtype=jnp.float32,
)
model, state = eqx.nn.inference_mode((model, state))

images = jnp.zeros((2, 3, 224, 224))

logits, _ = eqx.filter_vmap(
    model, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(images, state)

probs = jax.nn.softmax(logits, axis=-1)  # (2, 1000)
```

### Input preprocessing (ImageNet)

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
