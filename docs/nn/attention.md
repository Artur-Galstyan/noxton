# Attention

```python
from noxton.nn import MultiheadAttention, SqueezeExcitation
```

---

## MultiheadAttention

A 1-to-1 JAX/Equinox port of `torch.nn.MultiheadAttention`. Splits queries, keys and values into `num_heads` independent attention heads, computes scaled dot-product attention for each head in parallel, and projects the concatenated outputs back to `embed_dim`.

!!! note
    This implementation is intentionally API-compatible with `torch.nn.MultiheadAttention`. Unless you specifically need that compatibility, prefer `eqx.nn.MultiheadAttention`, which is more idiomatic for JAX.

### Constructor

```python
MultiheadAttention(
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.0,
    bias: bool = True,
    add_bias_kv: bool = False,
    add_zero_attn: bool = False,
    kdim: int | None = None,
    vdim: int | None = None,
    inference: bool = False,
    *,
    key: PRNGKeyArray,
    dtype = None,
)
```

| Parameter | Description |
|---|---|
| `embed_dim` | Total dimensionality of the model (query) embeddings. |
| `num_heads` | Number of attention heads. `embed_dim` must be divisible by `num_heads`. |
| `dropout` | Dropout probability on attention weights during training. Default `0.0`. |
| `bias` | Add learnable bias to input and output projections. Default `True`. |
| `add_bias_kv` | Append learnable bias vectors to key and value sequences. Default `False`. |
| `add_zero_attn` | Append a batch of zeros to key and value sequences. Default `False`. |
| `kdim` | Dimensionality of key inputs. Defaults to `embed_dim`. |
| `vdim` | Dimensionality of value inputs. Defaults to `embed_dim`. |
| `inference` | Disable dropout (eval mode). Default `False`. |
| `key` | JAX PRNG key for parameter initialisation. |
| `dtype` | Parameter dtype. Defaults to project default. |

### `__call__`

```python
mha(
    query: Array,           # (tgt_len, embed_dim)
    key: Array,             # (src_len, kdim)
    value: Array,           # (src_len, vdim)
    key_padding_mask = None,
    need_weights: bool = True,
    attn_mask = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    dropout_key = None,
) -> tuple[Array, Array | None]
```

Returns `(attn_output, attn_weights)` where `attn_output` has shape `(tgt_len, embed_dim)` and `attn_weights` is `None` when `need_weights=False`, shape `(tgt_len, src_len)` when averaged, or `(num_heads, tgt_len, src_len)` per-head.

### Example

```python
import jax
import jax.numpy as jnp
from noxton.nn import MultiheadAttention

key = jax.random.PRNGKey(0)
mha = MultiheadAttention(embed_dim=64, num_heads=4, key=key)

q = jax.random.normal(key, (10, 64))   # (seq_len, embed_dim)
out, weights = mha(q, q, q)            # self-attention
# out.shape    -> (10, 64)
# weights.shape -> (10, 10)

# Causal (decoder) attention
out, _ = mha(q, q, q, is_causal=True, need_weights=False)
```

---

## SqueezeExcitation

Squeeze-and-Excitation channel-attention block (Hu et al., 2018). Globally pools the spatial dimensions of a feature map, passes the result through two 1×1 convolutions to produce a per-channel scale vector, and multiplies it back into the input.

```
scale = sigmoid(fc2(relu(fc1(avgpool(x)))))
output = scale * x
```

### Constructor

```python
SqueezeExcitation(
    input_channels: int,
    squeeze_channels: int,
    *,
    key: PRNGKeyArray,
    dtype = None,
)
```

| Parameter | Description |
|---|---|
| `input_channels` | Number of channels in the input feature map. |
| `squeeze_channels` | Bottleneck width (typically `input_channels // reduction_ratio`). |

### `__call__`

```python
se(
    x: Array,                            # (C, H, W)
    activation = jax.nn.relu,
    scale_activation = jax.nn.sigmoid,
) -> Array                               # (C, H, W)
```

### Example

```python
import jax
from noxton.nn import SqueezeExcitation

key = jax.random.PRNGKey(0)
se = SqueezeExcitation(input_channels=64, squeeze_channels=16, key=key)

x = jax.random.normal(key, (64, 28, 28))  # (C, H, W)
out = se(x)
# out.shape -> (64, 28, 28)
```
