import equinox as eqx
import jax
import jax.numpy as jnp

from noxton.nn import mLSTMCell

max_seq_len = 4
embed_dim = 16
num_heads = 8
seq_len = 4


cell = mLSTMCell(embed_dim, num_heads, key=jax.random.key(22), max_seq_len=4)

q, k, v = (
    jnp.ones(shape=(seq_len, embed_dim)),
    jnp.ones(shape=(seq_len, embed_dim)),
    jnp.ones(shape=(seq_len, embed_dim)),
)


cell(q, k, v)


q, k, v = (
    jnp.ones(shape=(1, embed_dim)),
    jnp.ones(shape=(1, embed_dim)),
    jnp.ones(shape=(1, embed_dim)),
)

step_fn_jit = eqx.filter_jit(cell.step)
step_fn_jit(q, k, v)
step_fn_jit(q, k, v)
step_fn_jit(q, k, v)
