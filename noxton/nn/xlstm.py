import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, Float, PRNGKeyArray

from noxton.nn import CausalConv1d, LinearHeadwiseExpand, ResidualLayerNorm


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


def recurrent_step_stabilized_simple(
    c_state: Array,
    n_state: Array,
    m_state: Array,
    q: Float[Array, "num_heads 1 head_dim"],
    k: Float[Array, "num_heads 1 head_dim"],
    v: Float[Array, "num_heads 1 head_dim"],
    igate_preact: Array,
    fgate_preact: Array,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[Array, tuple[Array, Array, Array]]:
    NH, S, DH = q.shape

    q = q.reshape(NH, DH, 1)
    k = k.reshape(NH, DH, 1)
    v = v.reshape(NH, DH, 1)

    log_fg_act = jax.nn.log_sigmoid(fgate_preact)

    # update rule
    m_state_new = jnp.maximum(log_fg_act + m_state, igate_preact)

    fg_act = jnp.exp(log_fg_act + m_state - m_state_new)
    ig_act = jnp.exp(igate_preact - m_state_new)

    k_scaled = k / jnp.sqrt(DH)

    c_state_new = fg_act * c_state + ig_act * (k_scaled @ v.transpose(0, 2, 1))
    n_state_new = fg_act * n_state + ig_act * k_scaled

    h_num = q.transpose(0, 2, 1) @ c_state_new

    qn_dotproduct = q.transpose(0, 2, 1) @ n_state_new
    max_val = jnp.exp(-m_state_new)
    h_denom = jnp.maximum(jnp.abs(qn_dotproduct), max_val) + eps
    h = h_num / h_denom

    return h, (c_state_new, n_state_new, m_state_new)


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

    def step(
        self,
        q: Float[Array, "seq_len embed_dim"],
        k: Float[Array, "seq_len embed_dim"],
        v: Float[Array, "seq_len embed_dim"],
        mlstm_state: tuple[Array, Array, Array] | None = None,
        **kwargs,
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        S, _ = q.shape  # (S, H)
        assert S == 1, (
            f"mLSTMCell.step only supports sequence length S=1, but got S={S}."
        )

        if_gate_input = jnp.concatenate([q, k, v], axis=-1)
        q = q.reshape(S, self.num_heads, -1)  # (S, NH, DH)
        k = k.reshape(S, self.num_heads, -1)  # (S, NH, DH)
        v = v.reshape(S, self.num_heads, -1)  # (S, NH, DH)

        _, NH, DH = q.shape

        q = q.transpose(1, 0, 2)  # (NH, S, DH)
        k = k.transpose(1, 0, 2)  # (NH, S, DH)
        v = v.transpose(1, 0, 2)  # (NH, S, DH)

        igate_preact = eqx.filter_vmap(self.igate)(if_gate_input)  # (S, NH)
        igate_preact = jnp.expand_dims(igate_preact, axis=-1)  # (S, NH, 1)
        igate_preact = igate_preact.transpose(1, 0, 2)  # (NH, S, 1)

        fgate_preact = eqx.filter_vmap(self.fgate)(if_gate_input)  # (S, NH)
        fgate_preact = jnp.expand_dims(fgate_preact, axis=-1)  # (S, NH, 1)
        fgate_preact = fgate_preact.transpose(1, 0, 2)  # (NH, S, 1)

        if mlstm_state is None:
            c_state = jnp.zeros(shape=(NH, DH, DH))
            n_state = jnp.zeros(shape=(NH, DH, 1))
            m_state = jnp.zeros(shape=(NH, 1, 1))
        else:
            c_state, n_state, m_state = mlstm_state

        assert c_state.shape == (NH, DH, DH), (
            f"Expected c_state shape {(NH, DH, DH)}, but got {c_state.shape}."
        )
        assert n_state.shape == (NH, DH, 1), (
            f"Expected n_state shape {(NH, DH, 1)}, but got {n_state.shape}."
        )
        assert m_state.shape == (NH, 1, 1), (
            f"Expected m_state shape {(NH, 1, 1)}, but got {m_state.shape}."
        )

        h_state, mlstm_state = recurrent_step_stabilized_simple(
            c_state=c_state,
            n_state=n_state,
            m_state=m_state,
            q=q,
            k=k,
            v=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
        )  # (NH, 1 DH), ((NH, DH, DH), (NH, DH, 1), (NH, 1, 1))
        h_state = h_state.transpose(1, 0, 2).reshape(
            S, -1
        )  # (NH, 1, DH) -> (1, NH, DH) -> (1, NH*DH) = (S, embedding_dim)
        h_state_norm = eqx.filter_vmap(self.outnorm)(h_state)  # vmap over S
        return h_state_norm, mlstm_state


class mLSTMLayer(eqx.Module):
    proj_up: eqx.nn.Linear
    q_proj: LinearHeadwiseExpand
    k_proj: LinearHeadwiseExpand
    v_proj: LinearHeadwiseExpand
    conv1d: CausalConv1d
    mlstm_cell: mLSTMCell
    learnable_skip: Array
    proj_down: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    inference: bool

    inner_embedding_dim: int

    def __init__(
        self,
        embedding_dim: int,
        inner_embedding_dim: int,
        qkv_proj_blocksize: int,
        use_bias: bool,
        conv1d_kernel_size: int,
        max_seq_len: int,
        num_heads: int,
        dropout_p: float,
        *,
        key: PRNGKeyArray,
        inference: bool = False,
        dtype: Any | None = None,
    ):
        key, proj_key = jax.random.split(key)
        self.proj_up = eqx.nn.Linear(
            in_features=embedding_dim,
            out_features=2 * inner_embedding_dim,
            use_bias=use_bias,
            key=proj_key,
            dtype=dtype,
        )

        num_proj_heads = round(inner_embedding_dim // qkv_proj_blocksize)

        key, qkey = jax.random.split(key)
        self.q_proj = LinearHeadwiseExpand(
            in_features=inner_embedding_dim,
            out_features=inner_embedding_dim,
            num_heads=num_proj_heads,
            use_bias=use_bias,
            dtype=dtype,
            key=qkey,
        )
        key, kkey = jax.random.split(key)
        self.k_proj = LinearHeadwiseExpand(
            in_features=inner_embedding_dim,
            out_features=inner_embedding_dim,
            num_heads=num_proj_heads,
            use_bias=use_bias,
            dtype=dtype,
            key=kkey,
        )
        key, vkey = jax.random.split(key)
        self.v_proj = LinearHeadwiseExpand(
            in_features=inner_embedding_dim,
            out_features=inner_embedding_dim,
            num_heads=num_proj_heads,
            use_bias=use_bias,
            dtype=dtype,
            key=vkey,
        )

        key, convkey = jax.random.split(key)
        self.conv1d = CausalConv1d(
            feature_dim=inner_embedding_dim,
            kernel_size=conv1d_kernel_size,
            key=convkey,
            channel_mixing=False,
            dtype=dtype,
        )
        key, cellkey = jax.random.split(key)
        self.mlstm_cell = mLSTMCell(
            max_seq_len=max_seq_len,
            embedding_dim=inner_embedding_dim,
            num_heads=num_heads,
            key=cellkey,
            dtype=dtype,
        )

        self.learnable_skip = jnp.ones(inner_embedding_dim)

        key, proj_down_key = jax.random.split(key)
        self.proj_down = eqx.nn.Linear(
            in_features=inner_embedding_dim,
            out_features=embedding_dim,
            use_bias=use_bias,
            key=proj_down_key,
            dtype=dtype,
        )
        self.dropout = eqx.nn.Dropout(dropout_p)
        self.inference = inference

        self.inner_embedding_dim = inner_embedding_dim

    def __call__(self, x: Float[Array, "seq_len embed_dim"], key: PRNGKeyArray | None):
        S, _ = x.shape

        # up-projection
        x_inner = eqx.filter_vmap(self.proj_up)(x)
        x_mlstm, z = jnp.split(x_inner, [self.inner_embedding_dim], axis=-1)

        # mlstm branch
        x_mlstm_conv = self.conv1d(x_mlstm)
        assert isinstance(x_mlstm_conv, Array)
        x_mlstm_conv_act = jax.nn.silu(x_mlstm_conv)

        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)

        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * jax.nn.silu(z)

        # down-projection
        y = self.dropout(eqx.filter_vmap(self.proj_down)(h_state), key=key)
        return y

    def step(
        self,
        x: Float[Array, "1 embed_dim"],
        mlstm_state: tuple[Array, Array, Array] | None = None,
        conv_state: tuple[Array] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Array, tuple[tuple[Array, Array, Array], tuple[Array] | None]]:
        x_inner = eqx.filter_vmap(self.proj_up)(x)
        x_mlstm, z = jnp.split(x_inner, [self.inner_embedding_dim], axis=-1)

        x_mlstm_conv, conv_state = self.conv1d.step(x_mlstm, conv_state=conv_state)
        print(f"{x_mlstm_conv.shape=}")
        x_mlstm_conv_act = jax.nn.silu(x_mlstm_conv)

        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        h_tilde_state, mlstm_state = self.mlstm_cell.step(
            q=q, k=k, v=v, mlstm_state=mlstm_state
        )
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        h_state = h_tilde_state_skip * jax.nn.silu(z)

        y = eqx.filter_vmap(self.proj_down)(h_state)
        y = self.dropout(y, key=key, inference=self.inference)

        return y, (mlstm_state, conv_state)


class sLSTMCell(eqx.Module):
    pass


class sLSTMLayer(eqx.Module):
    pass
