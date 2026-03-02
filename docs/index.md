# Noxton

**Noxton** is a JAX/Equinox library of neural network building blocks and pretrained models. It provides:

- **`noxton.nn`** — low-level layers (attention, normalization, convolution, Mamba, Transformer, …) that compose naturally with [Equinox](https://docs.kidger.site/equinox/)
- **`noxton.models`** — high-level vision and language models with `from_pretrained()` loaders that download and convert PyTorch weights on demand

All modules are pure JAX — no hidden state, no magic globals. Models are `eqx.Module` subclasses so they work with `jax.jit`, `jax.vmap`, `jax.grad`, and the rest of the JAX ecosystem out of the box.

---

## Available models

| Model | Variants | Task |
|---|---|---|
| [AlexNet](models/alexnet.md) | alexnet | ImageNet classification |
| [ResNet](models/resnet.md) | resnet18/34/50/101/152, resnext50/101, wide_resnet50/101 | ImageNet classification |
| [ConvNeXt](models/convnext.md) | tiny, small, base, large | ImageNet classification |
| [EfficientNet](models/efficientnet.md) | B0–B7, V2-S/M/L | ImageNet classification |
| [Swin Transformer](models/swin-transformer.md) | swin_t/s/b, swin_v2_t/s/b | ImageNet classification |
| [CLIP](models/clip.md) | RN50, RN101, ViT-B/32, ViT-B/16, ViT-L/14 | Zero-shot image–text |
| [ESM](models/esm.md) | ESM3, ESMC | Protein language modelling |

---

## Quick example

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from noxton.models import ResNet

# Load ResNet-50 with pretrained ImageNet V2 weights
model, state = ResNet.from_pretrained(
    model="resnet50",
    weights="resnet50_IMAGENET1K_V2",
    key=jax.random.key(0),
    dtype=jnp.float32,
)

# Switch to inference mode (disables BatchNorm updates, dropout)
model, state = eqx.nn.inference_mode((model, state))

# Batch inference via filter_vmap
images = jax.random.normal(jax.random.key(1), (4, 3, 224, 224))
logits, _ = eqx.filter_vmap(
    model, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(images, state)

probs = jax.nn.softmax(logits, axis=-1)  # (4, 1000)
```

---

## Design principles

**Functional first.** Every layer is an `eqx.Module`. Stateful operations (BatchNorm running statistics, dropout keys) flow through explicit arguments — there are no hidden side effects.

**Composable.** All `noxton.nn` primitives accept a `key: PRNGKeyArray` constructor argument and a `dtype` argument. They can be freely mixed with standard Equinox and JAX primitives.

**Pretrained weights.** Weights are downloaded once to `~/.noxton/` and converted from PyTorch `.pth`/`.pt` files using `statedict2pytree`. Once converted, PyTorch is not required at inference time.
