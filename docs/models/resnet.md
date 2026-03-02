# ResNet

```python
from noxton.models import ResNet
```

ResNet (He et al., 2015) introduced residual skip connections enabling very deep networks. Noxton supports the full ResNet family including ResNeXt and Wide ResNet variants.

---

## Variants

| `model` | Depth | Block | Params |
|---|---|---|---|
| `resnet18` | 18 | BasicBlock | 11.7M |
| `resnet34` | 34 | BasicBlock | 21.8M |
| `resnet50` | 50 | Bottleneck | 25.6M |
| `resnet101` | 101 | Bottleneck | 44.5M |
| `resnet152` | 152 | Bottleneck | 60.2M |
| `resnext50_32x4d` | 50 | Bottleneck (groups=32) | 25.0M |
| `resnext101_32x8d` | 101 | Bottleneck (groups=32) | 88.8M |
| `resnext101_64x4d` | 101 | Bottleneck (groups=64) | 83.5M |
| `wide_resnet50_2` | 50 | Bottleneck (2× wide) | 68.9M |
| `wide_resnet101_2` | 101 | Bottleneck (2× wide) | 126.9M |

---

## `from_pretrained`

```python
ResNet.from_pretrained(
    model: str,
    weights: str,
    key: PRNGKeyArray,
    dtype = None,
) -> tuple[ResNet, eqx.nn.State]
```

### Available weights

| `weights` | Top-1 |
|---|---|
| `resnet18_IMAGENET1K_V1` | 69.8% |
| `resnet34_IMAGENET1K_V1` | 73.3% |
| `resnet50_IMAGENET1K_V1` | 76.1% |
| `resnet50_IMAGENET1K_V2` | 80.9% |
| `resnet101_IMAGENET1K_V1` | 77.4% |
| `resnet101_IMAGENET1K_V2` | 81.9% |
| `resnet152_IMAGENET1K_V1` | 78.3% |
| `resnet152_IMAGENET1K_V2` | 82.3% |
| `resnext50_32X4D_IMAGENET1K_V1` | 77.6% |
| `resnext50_32X4D_IMAGENET1K_V2` | 81.2% |
| `resnext101_32X8D_IMAGENET1K_V1` | 79.3% |
| `resnext101_32X8D_IMAGENET1K_V2` | 82.8% |
| `resnext101_64X4D_IMAGENET1K_V1` | 83.2% |
| `wide_resnet50_2_IMAGENET1K_V1` | 78.5% |
| `wide_resnet50_2_IMAGENET1K_V2` | 81.6% |
| `wide_resnet101_2_IMAGENET1K_V1` | 78.8% |
| `wide_resnet101_2_IMAGENET1K_V2` | 82.5% |

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
from noxton.models import ResNet

# Load ResNet-50 V2
model, state = ResNet.from_pretrained(
    model="resnet50",
    weights="resnet50_IMAGENET1K_V2",
    key=jax.random.key(0),
    dtype=jnp.float16,
)
model, state = eqx.nn.inference_mode((model, state))

# Prepare a batch of images: (B, 3, 224, 224)
images = jnp.zeros((4, 3, 224, 224), dtype=jnp.float16)

logits, _ = eqx.filter_vmap(
    model, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(images, state)

probs = jax.nn.softmax(logits, axis=-1)  # (4, 1000)
top5 = jnp.argsort(probs, axis=-1)[:, -5:][:, ::-1]
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
