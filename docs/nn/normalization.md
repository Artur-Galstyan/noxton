# Normalization

```python
from noxton.nn import BatchNorm, LayerNorm, LocalResponseNormalization
```

---

## BatchNorm

Batch normalisation for use inside `jax.vmap` / `jax.lax.pmean`. Normalises each channel across the batch using statistics gathered via `jax.lax.pmean` over the named `axis_name`. Running mean and variance are maintained in an Equinox `State` object.

At inference time (`inference=True`) stored running statistics are used instead of batch statistics.

### Constructor

```python
BatchNorm(
    size: int,
    axis_name: Hashable | Sequence[Hashable],
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = True,
    inference: bool = False,
    dtype = None,
)
```

| Parameter | Description |
|---|---|
| `size` | Number of channels to normalise. |
| `axis_name` | The `vmap`/`pmap` axis name used for cross-batch aggregation. |
| `eps` | Numerical stability constant. Default `1e-5`. |
| `momentum` | EMA factor for running statistics. Default `0.1`. |
| `affine` | Learn per-channel scale and shift. Default `True`. |
| `inference` | Use running statistics. Default `False`. |

### `__call__`

```python
bn(
    x: Array,          # (size,) or (size, *spatial_dims)
    state: eqx.nn.State,
    key = None,
) -> tuple[Array, eqx.nn.State]
```

Must be called inside a `jax.vmap` scope that maps over `axis_name`.

### Example

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from noxton.nn import BatchNorm

bn = BatchNorm(size=16, axis_name="batch")
state = eqx.nn.State(bn)

x = jax.random.normal(jax.random.PRNGKey(0), (4, 16))  # batch of 4

out, state = jax.vmap(
    bn, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
)(x, state)
# out.shape -> (4, 16)
```

---

## LayerNorm

Layer normalisation over the last one or more dimensions. Normalises by subtracting the mean and dividing by the standard deviation, then optionally applies learnable affine parameters `weight` (scale) and `bias` (shift). Computation is performed at `float32` precision minimum and cast back to the input dtype.

### Constructor

```python
LayerNorm(
    shape: int | Sequence[int],
    eps: float = 1e-5,
    use_weight: bool = True,
    use_bias: bool = True,
    dtype = None,
)
```

### `__call__`

```python
ln(x: Array, key=None) -> Array
```

### Example

```python
import jax.numpy as jnp
from noxton.nn import LayerNorm

ln = LayerNorm(shape=64)
x = jnp.ones((10, 64))
out = ln(x)
# out.shape -> (10, 64)

# 2-D normalisation
ln2d = LayerNorm(shape=(28, 28))
x2d = jnp.ones((16, 28, 28))
out2d = ln2d(x2d)
# out2d.shape -> (16, 28, 28)
```

---

## LocalResponseNormalization

Local Response Normalisation (LRN) across adjacent channels, as used in AlexNet (Krizhevsky et al., 2012):

```
b[c,h,w] = x[c,h,w] / (k + alpha * sum_j x[j,h,w]^2)^beta
```

where the sum runs over at most `n` channels centred on `c`.

### Constructor

```python
LocalResponseNormalization(
    k: int = 2,
    n: int = 5,
    alpha: float = 1e-4,
    beta: float = 0.75,
)
```

### `__call__`

```python
lrn(x: Array) -> Array   # (C, H, W) -> (C, H, W)
```

### Example

```python
import jax.numpy as jnp
from noxton.nn import LocalResponseNormalization

lrn = LocalResponseNormalization()
x = jnp.ones((8, 14, 14))
out = lrn(x)
# out.shape -> (8, 14, 14)
```
