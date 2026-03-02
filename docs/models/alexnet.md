# AlexNet

```python
from noxton.models import AlexNet
```

AlexNet is the landmark 8-layer CNN that won ImageNet 2012 (Krizhevsky et al., 2012). The architecture is:

```
Conv(11×11, s=4) → ReLU → MaxPool
Conv(5×5)        → ReLU → MaxPool
Conv(3×3)        → ReLU
Conv(3×3)        → ReLU
Conv(3×3)        → ReLU → MaxPool
AdaptiveAvgPool(6×6)
Dropout → Linear(9216, 4096) → ReLU
Dropout → Linear(4096, 4096) → ReLU
Linear(4096, n_classes)
```

Local Response Normalisation (LRN) is applied after the first and second pooling layers.

---

## Constructor

```python
AlexNet(
    *,
    n_classes: int,
    key: PRNGKeyArray,
    inference: bool = False,
    dtype = None,
)
```

| Parameter | Description |
|---|---|
| `n_classes` | Number of output classes. Use `1000` for ImageNet. |
| `key` | JAX PRNG key for parameter initialisation. |
| `inference` | Disable dropout. Default `False`. |
| `dtype` | Parameter dtype. Default: project default (`float32`). |

---

## `from_pretrained`

```python
AlexNet.from_pretrained(
    weights: str = "alexnet_IMAGENET1K_V1",
    key: PRNGKeyArray = ...,
    dtype = None,
) -> tuple[AlexNet, eqx.nn.State]
```

| `weights` | Dataset | Top-1 |
|---|---|---|
| `alexnet_IMAGENET1K_V1` | ImageNet-1K | 56.5% |

Weights are downloaded from PyTorch Hub and cached at `~/.noxton/`.

---

## `__call__`

```python
model(
    x: Array,                # (3, H, W) — single image, no batch dim
    state: eqx.nn.State,
    key: PRNGKeyArray | None = None,
) -> tuple[Array, eqx.nn.State]   # (n_classes,), state
```

---

## Example

```python
import os
import jax
import jax.numpy as jnp
import equinox as eqx
from PIL import Image
from torchvision import transforms
from noxton.models import AlexNet

# Load pretrained model
model, state = AlexNet.from_pretrained(
    weights="alexnet_IMAGENET1K_V1",
    key=jax.random.key(0),
    dtype=jnp.float32,
)
model, state = eqx.nn.inference_mode((model, state))

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
image = Image.open("cat.jpg")
x = jnp.array(preprocess(image).unsqueeze(0).numpy())   # (1, 3, 224, 224)

# Batch inference
logits, _ = eqx.filter_vmap(
    model, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(x, state)

probs = jax.nn.softmax(logits[0])
top5 = jnp.argsort(probs)[-5:][::-1]
```
