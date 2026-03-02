# Embedding

```python
from noxton.nn import EmbeddingWithPadding, EmbeddingBag
```

---

## EmbeddingWithPadding

Embedding table that zeros out embeddings for a configurable padding index. Wraps `eqx.nn.Embedding` and multiplies every looked-up embedding by a binary mask that is `0` wherever the input equals `padding_idx`.

### Constructor

```python
EmbeddingWithPadding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int = 0,
    *,
    key: PRNGKeyArray,
    dtype = None,
)
```

### `__call__`

```python
emb(x: Array) -> Array   # (seq_len,) -> (seq_len, embedding_dim)
```

### Example

```python
import jax
import jax.numpy as jnp
from noxton.nn import EmbeddingWithPadding

key = jax.random.PRNGKey(0)
emb = EmbeddingWithPadding(num_embeddings=10, embedding_dim=4, key=key)

ids = jnp.array([0, 1, 2, 0])   # 0 is padding
out = emb(ids)
# out.shape -> (4, 4)
# out[0] and out[3] are zero vectors
```

---

## EmbeddingBag

Sums a bag of token embeddings into a single vector. Looks up each index using `EmbeddingWithPadding` (padding tokens contribute zero), then reduces by summing. Analogous to `torch.nn.EmbeddingBag(mode="sum")`.

### Constructor

```python
EmbeddingBag(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int = 0,
    *,
    key: PRNGKeyArray,
    dtype = None,
)
```

### `__call__`

```python
bag(x: Array) -> Array   # (bag_size,) -> (embedding_dim,)
```

### Example

```python
import jax
import jax.numpy as jnp
from noxton.nn import EmbeddingBag

key = jax.random.PRNGKey(0)
bag = EmbeddingBag(num_embeddings=10, embedding_dim=4, key=key)

ids = jnp.array([1, 3, 5])
out = bag(ids)
# out.shape -> (4,)
```
