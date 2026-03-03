import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, Float, PRNGKeyArray

from noxton.nn import ResidualLayerNorm


class mLSTMCell(eqx.Module):
    embedding_dim: int
    num_heads: int

    igate: eqx.nn.Linear
    fgate: eqx.nn.Linear

    outnorm: ResidualLayerNorm

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        key, ikey, fkey = jax.random.split(key, 3)

        igate = eqx.nn.Linear(3 * embedding_dim, num_heads, key=ikey, dtype=dtype)
        igate = eqx.tree_at(
            lambda l: l.weight, igate, jnp.zeros_like(igate.weight, dtype=dtype)
        )
        self.igate = eqx.tree_at(
            lambda l: l.bias,
            igate,
            jnp.linspace(start=3.0, stop=6.0, num=len(igate.bias), dtype=dtype),
        )

        fgate = eqx.nn.Linear(3 * embedding_dim, num_heads, key=fkey, dtype=dtype)
        fgate = eqx.tree_at(
            lambda l: l.weight, fgate, jnp.zeros_like(fgate.weight, dtype=dtype)
        )
        key, subkey = jax.random.split(key)
        self.fgate = eqx.tree_at(
            lambda l: l.bias,
            fgate,
            jnp.sqrt(0.1) * jax.random.normal(key=subkey, shape=fgate.bias.shape),
        )

        self.outnorm = ResidualLayerNorm(embedding_dim, use_bias=False, dtype=dtype)

    def __call__(
        self,
        q: Float[Array, "seq_len embed_dim"],
        k: Float[Array, "seq_len embed_dim"],
        v: Float[Array, "seq_len embed_dim"],
    ):
        seq_len, _ = q.shape
        if_gate_input = jnp.concatenate((q, k, v), axis=1)
        head_dim = self.embedding_dim // self.num_heads
        q = jnp.reshape(q, shape=(seq_len, self.num_heads, head_dim)).transpose(1, 0, 2)
        k = jnp.reshape(k, shape=(seq_len, self.num_heads, head_dim)).transpose(1, 0, 2)
        v = jnp.reshape(v, shape=(seq_len, self.num_heads, head_dim)).transpose(1, 0, 2)

        igate_preact = self.igate(if_gate_input)
        igate_preact = jnp.expand_dims(igate_preact.T, axis=-1)

        fgate_preact = self.fgate(if_gate_input)
        fgate_preact = jnp.expand_dims(fgate_preact.T, axis=-1)

        print(f"{igate_preact.shape=}")
        print(f"{fgate_preact.shape=}")


class mLSTMLayer(eqx.Module):
    pass


class sLSTMCell(eqx.Module):
    pass


class sLSTMLayer(eqx.Module):
    pass
