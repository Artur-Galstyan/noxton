# Transformer

```python
from noxton.nn import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    Transformer,
    VisionTransformer,
)
```

---

## TransformerEncoderLayer

Single encoder layer: multi-head self-attention + feed-forward MLP, each followed by layer normalisation and a residual connection.

### Constructor

```python
TransformerEncoderLayer(
    d_model: int,
    nhead: int,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation = jax.nn.relu,
    layer_norm_eps: float = 1e-5,
    norm_first: bool = False,
    bias: bool = True,
    inference: bool = False,
    *,
    key: PRNGKeyArray,
    dtype = None,
)
```

### `__call__`

```python
layer(
    src: Array,                 # (seq_len, d_model)
    src_mask = None,
    src_key_padding_mask = None,
    is_causal: bool = False,
    key = None,
) -> Array                      # (seq_len, d_model)
```

---

## TransformerDecoderLayer

Single decoder layer: self-attention + cross-attention + MLP, each with layer norm and residual.

### Constructor

```python
TransformerDecoderLayer(
    d_model: int,
    nhead: int,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation = jax.nn.relu,
    layer_norm_eps: float = 1e-5,
    norm_first: bool = False,
    bias: bool = True,
    inference: bool = False,
    *,
    key: PRNGKeyArray,
    dtype = None,
)
```

### `__call__`

```python
layer(
    tgt: Array,                 # (tgt_len, d_model)
    memory: Array,              # (src_len, d_model)
    tgt_mask = None,
    memory_mask = None,
    tgt_key_padding_mask = None,
    memory_key_padding_mask = None,
    tgt_is_causal: bool = False,
    memory_is_causal: bool = False,
    key = None,
) -> Array                      # (tgt_len, d_model)
```

---

## TransformerEncoder

Stack of `TransformerEncoderLayer`s with optional final `LayerNorm`.

### Constructor

```python
TransformerEncoder(
    encoder_layer: TransformerEncoderLayer,
    num_layers: int,
    norm = None,
)
```

### `__call__`

```python
encoder(
    src: Array,
    mask = None,
    src_key_padding_mask = None,
    is_causal: bool = False,
    key = None,
) -> Array
```

---

## TransformerDecoder

Stack of `TransformerDecoderLayer`s with optional final `LayerNorm`.

### Constructor

```python
TransformerDecoder(
    decoder_layer: TransformerDecoderLayer,
    num_layers: int,
    norm = None,
)
```

---

## Transformer

Full encoder-decoder architecture.

### Constructor

```python
Transformer(
    d_model: int = 512,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation = jax.nn.relu,
    layer_norm_eps: float = 1e-5,
    norm_first: bool = False,
    bias: bool = True,
    inference: bool = False,
    *,
    key: PRNGKeyArray,
    dtype = None,
)
```

### `__call__`

```python
transformer(
    src: Array,                 # (src_len, d_model)
    tgt: Array,                 # (tgt_len, d_model)
    src_mask = None,
    tgt_mask = None,
    memory_mask = None,
    src_key_padding_mask = None,
    tgt_key_padding_mask = None,
    memory_key_padding_mask = None,
    src_is_causal: bool = False,
    tgt_is_causal: bool = False,
    memory_is_causal: bool = False,
    key = None,
) -> Array                      # (tgt_len, d_model)
```

---

## VisionTransformer

Vision Transformer (ViT) with patch embedding and CLS token. Divides the input image into patches, linearly embeds each patch, prepends a learnable `[CLS]` token, adds positional embeddings, and passes the sequence through a stack of `ResidualAttentionBlock`s.

### Constructor

```python
VisionTransformer(
    image_size: int,
    patch_size: int,
    width: int,
    layers: int,
    heads: int,
    output_dim: int,
    *,
    key: PRNGKeyArray,
    dtype = None,
)
```

| Parameter | Description |
|---|---|
| `image_size` | Input image resolution (square). |
| `patch_size` | Size of each patch (square). |
| `width` | Transformer hidden dimension. |
| `layers` | Number of transformer blocks. |
| `heads` | Number of attention heads. |
| `output_dim` | Dimension of the projected output (CLS token after projection). |

### `__call__`

```python
vit(x: Array) -> Array   # (C, H, W) -> (output_dim,)
```

### Example

```python
import jax
import jax.numpy as jnp
from noxton.nn import VisionTransformer

key = jax.random.PRNGKey(0)
vit = VisionTransformer(
    image_size=224,
    patch_size=16,
    width=768,
    layers=12,
    heads=12,
    output_dim=512,
    key=key,
)

image = jax.random.normal(key, (3, 224, 224))
features = vit(image)
# features.shape -> (512,)
```
