import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, Float, PRNGKeyArray

from noxton.nn import ResidualLayerNorm


def parallel_stabilized_simple(
    queries: Float[Array, "num_heads seq_len head_dim"],
    keys: Float[Array, "num_heads seq_len head_dim"],
    values: Float[Array, "num_heads seq_len head_dim"],
    igate_preact: Float[Array, "num_heads seq_len"],
    fgate_preact: Float[Array, "num_heads seq_len"],
    lower_triangular_matrix: Float[Array, "seq_len seq_len"] | None = None,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
    **kwargs,
) -> Array:
    NH, S, DH = queries.shape

    log_fgates = jax.nn.log_sigmoid(fgate_preact)
    if lower_triangular_matrix is None or lower_triangular_matrix.shape[0] < S:
        lower_triangular_matrix = jnp.tril(jnp.ones(shape=(S, S), dtype=jnp.bool))

    assert lower_triangular_matrix is not None

    log_fgates_cumsum = jnp.concatenate(
        (jnp.zeros((NH, 1, 1)), jnp.cumsum(log_fgates, axis=1)), axis=1
    )
    rep_log_fgates_cumsum = jnp.tile(log_fgates_cumsum, (1, 1, S + 1))

    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(0, 2, 1)
    log_fg_matrix = jnp.where(
        lower_triangular_matrix, _log_fg_matrix[:, 1:, 1:], -float("inf")
    )
    log_D_matrix = log_fg_matrix + igate_preact.transpose(0, 2, 1)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D = jnp.max(log_D_matrix, axis=-1, keepdims=True)
    else:
        max_log_D = jnp.expand_dims(
            jnp.max(log_D_matrix.reshape(NH, -1), axis=-1, keepdims=True), axis=-1
        )

    log_D_matrix_stabilized = log_D_matrix - max_log_D
    D_matrix = jnp.exp(log_D_matrix_stabilized)

    keys_scaled = keys / jnp.sqrt(DH)

    qk_matrix = queries @ keys_scaled.transpose(0, 2, 1)
    C_matrix = qk_matrix * D_matrix
    normalizer = jnp.maximum(
        jnp.abs(C_matrix.sum(axis=-1, keepdims=True)), jnp.exp(-max_log_D)
    )
    C_matrix_normalized = C_matrix / (normalizer + eps)
    h_tilde_state = C_matrix_normalized @ values

    return h_tilde_state


class mLSTMCell(eqx.Module):
    max_seq_len: int
    embedding_dim: int
    num_heads: int

    igate: eqx.nn.Linear
    fgate: eqx.nn.Linear

    outnorm: ResidualLayerNorm

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_seq_len: int,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
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

        igate_preact = eqx.filter_vmap(self.igate)(if_gate_input)
        igate_preact = jnp.expand_dims(igate_preact.T, axis=-1)

        fgate_preact = eqx.filter_vmap(self.fgate)(if_gate_input)
        fgate_preact = jnp.expand_dims(fgate_preact.T, axis=-1)

        ltr = jnp.tril(
            jnp.ones(shape=(self.max_seq_len, self.max_seq_len), dtype=jnp.bool)
        )

        h_state = parallel_stabilized_simple(
            q, k, v, igate_preact, fgate_preact, lower_triangular_matrix=ltr
        )
        h_state = h_state.transpose(1, 0, 2).reshape(seq_len, -1)
        h_state_norm = eqx.filter_vmap(self.outnorm)(h_state)
        return h_state_norm


class mLSTMLayer(eqx.Module):
    pass


class sLSTMCell(eqx.Module):
    pass


class sLSTMLayer(eqx.Module):
    pass
