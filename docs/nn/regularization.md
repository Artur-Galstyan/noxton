# Regularization

```python
from noxton.nn import StochasticDepth
```

---

## StochasticDepth

Stochastic Depth (DropPath) regularisation layer (Huang et al., 2016). During training, randomly drops the entire input tensor (`mode="batch"`) or individual rows (`mode="row"`) with probability `p`. Surviving elements are rescaled by `1 / (1 - p)` to preserve expected values. At inference the layer is a no-op.

### Constructor

```python
StochasticDepth(
    p: float,
    mode: str,           # "batch" or "row"
    inference: bool = False,
)
```

| Parameter | Description |
|---|---|
| `p` | Drop probability in `[0, 1]`. |
| `mode` | `"batch"` — single mask broadcast over whole tensor; `"row"` — independent mask per row. |
| `inference` | Always pass through unchanged. Default `False`. |

### `__call__`

```python
sd(input: Array, key: PRNGKeyArray) -> Array
```

### Example

```python
import jax
import jax.numpy as jnp
from noxton.nn import StochasticDepth

key = jax.random.PRNGKey(0)
sd = StochasticDepth(p=0.2, mode="row")

x = jnp.ones((4, 8))
out = sd(x, key)
# out.shape -> (4, 8)
# during training some rows may be zeroed out
```

!!! tip
    To use in a residual block, apply stochastic depth to the residual branch:

    ```python
    x = x + sd(residual_branch(x), key)
    ```
