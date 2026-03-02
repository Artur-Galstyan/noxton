# Convolution

```python
from noxton.nn import ConvNormActivation
```

---

## ConvNormActivation

Composable Conv → Normalization → Activation block. Stacks a single `eqx.nn.Conv` layer, an optional normalisation layer, and an optional activation function into one Equinox `StatefulLayer`.

When `padding` is `None`, same-padding is computed automatically from `kernel_size` and `dilation` so that spatial dimensions are preserved (assuming `stride=1`). When `use_bias` is `None` it defaults to `True` only when no `norm_layer` is provided (normalisation makes biases redundant).

### Constructor

```python
ConvNormActivation(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    padding = None,
    groups: int = 1,
    activation_layer = jax.nn.relu,
    dilation: int | Sequence[int] = 1,
    use_bias: bool | None = None,
    *,
    norm_layer: Callable | None,
    key: PRNGKeyArray,
    dtype,
)
```

| Parameter | Description |
|---|---|
| `num_spatial_dims` | Number of spatial dimensions, e.g. `2` for images. |
| `in_channels` | Number of input channels. |
| `out_channels` | Number of output channels (filters). |
| `kernel_size` | Convolution kernel size. Default `3`. |
| `stride` | Convolution stride. Default `1`. |
| `padding` | Explicit padding or `None` for auto same-padding. |
| `groups` | Number of blocked connections (grouped convolution). Default `1`. |
| `activation_layer` | Activation callable or `None` to skip. Default `jax.nn.relu`. |
| `dilation` | Kernel dilation. Default `1`. |
| `use_bias` | Bias in convolution. `None` = `True` iff no norm layer. |
| `norm_layer` | Zero-argument factory returning an `AbstractNorm`/`AbstractNormStateful`, or `None`. |

### `__call__`

```python
block(
    x: Array,            # (in_channels, *spatial_dims)
    state: eqx.nn.State,
    key = None,
) -> tuple[Array, eqx.nn.State]
```

### Example

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from noxton.nn import ConvNormActivation, BatchNorm

key = jax.random.PRNGKey(0)
norm_factory = partial(BatchNorm, size=32, axis_name="batch")

block = ConvNormActivation(
    num_spatial_dims=2,
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    norm_layer=norm_factory,
    key=key,
    dtype=jnp.float32,
)
state = eqx.nn.State(block)

x = jax.random.normal(key, (4, 16, 28, 28))  # (batch, C, H, W)

out, state = jax.vmap(
    block, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
)(x, state)
# out.shape -> (4, 32, 28, 28)
```
