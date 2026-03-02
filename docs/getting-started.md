# Getting Started

## Installation

Install noxton from PyPI:

```bash
pip install noxton
```

### Optional dependencies

| Extra | When to install |
|---|---|
| `noxton[hub]` | Required to load ESM models (fetches weights from HuggingFace Hub) |
| `torch` | Required **only** during the first `from_pretrained()` call to convert PyTorch weights. Not needed at inference time after the first run. |

## Requirements

- Python ≥ 3.13
- JAX ≥ 0.4 (CPU, CUDA, or Metal backend)
- [Equinox](https://docs.kidger.site/equinox/)

## Basic usage

### Loading a pretrained model

All pretrained models share the same `from_pretrained()` interface:

```python
import jax.numpy as jnp
from noxton.models import ResNet

model, state = ResNet.from_pretrained(
    model="resnet50",
    weights="resnet50_IMAGENET1K_V2",
    key=jax.random.key(0),
    dtype=jnp.float32,
)
```

Weights are downloaded once to `~/.noxton/` and reused on subsequent calls.

### Inference mode

Call `eqx.nn.inference_mode` to freeze BatchNorm running statistics and disable dropout:

```python
import equinox as eqx

model, state = eqx.nn.inference_mode((model, state))
```

### Batched inference

Use `eqx.filter_vmap` to run a model over a batch. The `axis_name="batch"` argument is required for models that contain `BatchNorm`:

```python
import jax
import jax.numpy as jnp
import equinox as eqx

images = jnp.zeros((8, 3, 224, 224))  # batch of 8

logits, _ = eqx.filter_vmap(
    model,
    in_axes=(0, None),   # vmap over images, broadcast state
    out_axes=(0, None),  # stack logits, pass state through
    axis_name="batch",
)(images, state)

probs = jax.nn.softmax(logits, axis=-1)  # (8, 1000)
```

### Training

For training you typically want to:

1. Keep `inference=False` (the default) so BatchNorm tracks running statistics.
2. Split parameters from state using `eqx.partition` / `eqx.filter`.
3. Pass a `key` argument to layers that use dropout.

```python
import optax

optimizer = optax.adamw(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def train_step(model, state, opt_state, images, labels):
    def loss_fn(model):
        logits, new_state = eqx.filter_vmap(
            model, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
        )(images, state)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, new_state

    (loss, new_state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, new_state, opt_state, loss
```

## Weight cache

Downloaded weights are stored at:

```
~/.noxton/
  pytorch_weights/   # raw .pth / .pt files from the internet
  models/            # converted JAX checkpoints (per dtype)
```

Delete these directories to force a fresh download and conversion.

## dtype support

All models accept a `dtype` argument:

```python
# half-precision inference (faster on GPU/TPU)
model, state = ResNet.from_pretrained("resnet50", "resnet50_IMAGENET1K_V2",
                                      key=jax.random.key(0), dtype=jnp.float16)
```

Supported dtypes: `jnp.float32`, `jnp.float16`, `jnp.bfloat16`.
