# Examples

All example scripts are in the [`examples/`](https://github.com/arturgalstyan/noxton/tree/main/examples) directory. Each script loads a pretrained model, preprocesses `cat.jpg`, runs inference, and prints top-5 ImageNet predictions.

---

## ResNet inference

```python
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import requests
from PIL import Image
from torchvision import transforms
from noxton.models import ResNet


def get_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    return [line.strip() for line in response.text.splitlines()]


def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    return jnp.array(preprocess(image).unsqueeze(0).numpy())


# Load model
model, state = ResNet.from_pretrained(
    model="resnet50",
    weights="resnet50_IMAGENET1K_V2",
    key=jax.random.key(0),
    dtype=jnp.float16,
)
model, state = eqx.nn.inference_mode((model, state))

# Preprocess and run
x = preprocess_image("examples/cat.jpg").astype(jnp.float16)
logits, _ = eqx.filter_vmap(
    model, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(x, state)

probs = jax.nn.softmax(logits[0])
labels = get_imagenet_labels()
top5 = jnp.argsort(probs)[-5:][::-1]
for i, idx in enumerate(top5):
    print(f"{i+1}. {labels[int(idx)]}: {float(probs[idx]):.4f}")
```

---

## CLIP zero-shot classification

```python
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from PIL import Image
from noxton.models import CLIP
from noxton.models.clip import clip_tokenize


def preprocess_image(path, n_px=224):
    img = Image.open(path).convert("RGB").resize((n_px, n_px), Image.BICUBIC)
    x = jnp.array(np.array(img), dtype=jnp.float32) / 255.0
    x = jnp.transpose(x, (2, 0, 1))  # HWC -> CHW
    mean = jnp.array([0.48145466, 0.4578275,  0.40821073]).reshape(3, 1, 1)
    std  = jnp.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    return ((x - mean) / std).astype(jnp.float16)


image  = preprocess_image("examples/cat.jpg")
text   = clip_tokenize(["a photo of a human", "a photo of a cat", "a photo of a dog"])

clip, state = CLIP.from_pretrained(model="ViT-B/32", dtype=jnp.float16)
clip, state = eqx.nn.inference_mode((clip, state))

logits_per_image, _, state = eqx.filter_vmap(
    clip,
    in_axes=(None, 0, None),
    out_axes=(0, 0, None),
    axis_name="batch",
)(image, text, state)

probs = jax.nn.softmax(logits_per_image)
for label, p in zip(["human", "cat", "dog"], probs):
    print(f"{label}: {p * 100:.1f}%")
```

---

## EfficientNet inference

```python
import functools
import jax
import jax.numpy as jnp
import equinox as eqx
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights
from noxton.models import EfficientNet


model, state = EfficientNet.from_pretrained(
    "efficientnet_b0",
    weights="efficientnet_b0_IMAGENET1K_V1",
    dtype=jnp.float16,
)
model, state = eqx.nn.inference_mode((model, state))

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
img = Image.open("examples/cat.jpg")
x = jnp.array(preprocess(img).unsqueeze(0).numpy(), dtype=jnp.float16)

key = jax.random.key(0)
model_fn = functools.partial(model, key=key)
logits, _ = eqx.filter_vmap(
    model_fn, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(x, state)

categories = EfficientNet_B0_Weights.IMAGENET1K_V1.meta["categories"]
probs = jax.nn.softmax(logits[0])
_, top5 = jax.lax.top_k(probs, 5)
for i, idx in enumerate(top5):
    print(f"{i+1}. {categories[idx]} ({probs[idx] * 100:.2f}%)")
```

---

## Swin Transformer inference

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from torchvision import transforms
from PIL import Image
from noxton.models import SwinTransformer

# V1 uses 224px input
preprocess_v1 = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# V2 uses 256px input
preprocess_v2 = transforms.Compose([
    transforms.Resize(260),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

for model_name, weights, preprocess in [
    ("swin_t",    "swin_t_IMAGENET1K_V1",    preprocess_v1),
    ("swin_v2_t", "swin_v2_t_IMAGENET1K_V1", preprocess_v2),
]:
    model, state = SwinTransformer.from_pretrained(
        model=model_name, weights=weights,
        key=jax.random.key(0), dtype=jnp.float32,
    )
    model, state = eqx.nn.inference_mode((model, state))

    img = Image.open("examples/cat.jpg")
    x = jnp.array(preprocess(img).unsqueeze(0).numpy())

    logits, _ = eqx.filter_vmap(
        model, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(x, state)

    probs = jax.nn.softmax(logits[0])
    top1  = int(jnp.argmax(probs))
    print(f"{model_name}: top-1 class {top1} ({float(probs[top1]):.4f})")
```
