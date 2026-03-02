# State Space (Mamba)

```python
from noxton.nn import SelectiveStateSpaceModel, MambaBlock, Mamba
```

Noxton implements the Mamba architecture from *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (Gu & Dao, 2023). The hierarchy is:

```
Mamba                  ← full language model (embedding + blocks + LM head)
  └─ ResidualBlock     ← RMSNorm + MambaBlock + residual
       └─ MambaBlock   ← conv + SelectiveStateSpaceModel + gating
            └─ SelectiveStateSpaceModel  ← the core SSM recurrence
```

A lower-level `SelectiveStateSpace` class in `noxton.nn` exposes the SSM with an additional output projection.

---

## SelectiveStateSpaceModel

The core SSM recurrence without an output projection. Implements input-dependent state-space dynamics:

1. Project each token to `(dt_rank + 2 * d_state)` to obtain `delta`, `B`, `C`.
2. Project `delta` from `dt_rank` → `d_inner` and apply `softplus`.
3. Run the selective scan recurrence.

### Constructor

```python
SelectiveStateSpaceModel(
    d_inner: int,
    dt_rank: int,
    d_state: int,
    use_input_proj_bias: bool = False,
    use_delta_proj_bias: bool = False,
    *,
    key: PRNGKeyArray,
    dtype = None,
)
```

### `__call__`

```python
ssm(x: Array) -> Array   # (seq_len, d_inner) -> (seq_len, d_inner)
```

---

## MambaBlock

Complete Mamba block with 1-D depthwise convolution, SSM, and output gating.

### Constructor

```python
MambaBlock(
    d_model: int,
    d_inner: int,
    dt_rank: int,
    d_state: int,
    d_conv: int,
    *,
    key: PRNGKeyArray,
    dtype = None,
)
```

### `__call__`

```python
block(x: Array) -> Array   # (seq_len, d_model) -> (seq_len, d_model)
```

---

## Mamba

Full Mamba language model: token embedding → stack of `ResidualBlock`s → LM head.

### Constructor

```python
Mamba(
    vocab_size: int,
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    n_layers: int = 4,
    *,
    key: PRNGKeyArray,
    dtype = None,
)
```

| Parameter | Description |
|---|---|
| `vocab_size` | Vocabulary size for token embedding and LM head. |
| `d_model` | Model dimensionality. |
| `d_state` | Latent state size `N`. Default `16`. |
| `d_conv` | Depthwise convolution kernel size. Default `4`. |
| `expand` | Expansion factor `d_inner = expand * d_model`. Default `2`. |
| `n_layers` | Number of residual Mamba blocks. Default `4`. |

### `__call__`

```python
model(x: Array) -> Array   # (seq_len,) int tokens -> (seq_len, vocab_size) logits
```

### Example

```python
import jax
import jax.numpy as jnp
from noxton.nn import Mamba

key = jax.random.PRNGKey(0)
model = Mamba(
    vocab_size=256,
    d_model=128,
    n_layers=4,
    key=key,
)

tokens = jnp.array([1, 5, 42, 7, 100])   # (seq_len,)
logits = model(tokens)
# logits.shape -> (5, 256)
```
