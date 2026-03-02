# Linear

```python
from noxton.nn import BatchedLinear
```

---

## BatchedLinear

Linear layer that natively handles arbitrarily-batched inputs without requiring an explicit outer `vmap`. Accepts inputs with any number of leading batch dimensions and reshapes internally. Supports both real and complex dtypes.

Weights and biases are initialised with a uniform distribution in `[-1/√in_features, 1/√in_features]`.

### Constructor

```python
BatchedLinear(
    in_features: int | Literal["scalar"],
    out_features: int | Literal["scalar"],
    use_bias: bool = True,
    dtype = None,
    *,
    key: PRNGKeyArray,
)
```

| Parameter | Description |
|---|---|
| `in_features` | Size of the last input dimension, or `"scalar"`. |
| `out_features` | Size of the last output dimension, or `"scalar"`. |
| `use_bias` | Add a learnable bias vector. Default `True`. |
| `dtype` | Parameter dtype. Supports complex dtypes. Default: project default. |

### `__call__`

```python
linear(
    x: Array,       # (*batch_dims, in_features)
    key = None,
) -> Array          # (*batch_dims, out_features)
```

### Example

```python
import jax
import jax.numpy as jnp
from noxton.nn import BatchedLinear

key = jax.random.PRNGKey(0)
linear = BatchedLinear(in_features=8, out_features=4, key=key)

# Arbitrary batch dimensions
x = jax.random.normal(key, (3, 5, 8))
out = linear(x)
# out.shape -> (3, 5, 4)

# Works on plain 1-D vectors too
out1d = linear(jnp.ones(8))
# out1d.shape -> (4,)
```
