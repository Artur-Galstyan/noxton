import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, Float, PRNGKeyArray

from noxton.nn import (
    CausalConv1d,
    GatedFeedForward,
    LinearHeadwiseExpand,
    ResidualLayerNorm,
)
from noxton.utils import default_floating_dtype


def slstm_pointwise(
    raw: Float[Array, "hidden_dim_x4"],
    states: Float[Array, "4 hidden_dim"],
) -> tuple[Float[Array, "4 hidden_dim"], Float[Array, "4 hidden_dim"]]:
    """Computes the sLSTM pointwise gates and updates the cell states.

    This function implements the core pointwise operations of the scalar LSTM (sLSTM) cell.
    It takes the raw input projections and previous states to compute the input, forget,
    and output gates, as well as the new cell state (c), normalizer state (n), and memory
    state (m).

    Args:
        raw: The raw input projections of shape (hidden_dim * 4,).
        states: A tuple containing the previous states (y, c, n, m).

    Returns:
        A tuple containing the new states (y, c, n, m) and the computed gates.
    """
    y, c, n, m = states

    iraw, fraw, zraw, oraw = jnp.split(raw, 4)

    logfplusm = m + jax.nn.log_sigmoid(fraw)
    mnew = jnp.where(
        jnp.all(n == 0.0),
        iraw,
        jnp.maximum(iraw, logfplusm),
    )

    ogate = jax.nn.sigmoid(oraw)
    igate = jnp.minimum(jnp.exp(iraw - mnew), jnp.ones_like(iraw))
    fgate = jnp.minimum(jnp.exp(logfplusm - mnew), jnp.ones_like(iraw))

    cnew = fgate * c + igate * jnp.tanh(zraw)
    nnew = fgate * n + igate
    ynew = ogate * cnew / nnew

    new_states = jnp.stack([ynew, cnew, nnew, mnew])
    gates = jnp.stack([igate, fgate, zraw, ogate])

    return new_states, gates


def slstm_recurrent_step(
    Wx_t: Float[Array, "gates_x_hidden"],
    states: Float[Array, "4 hidden_dim"],
    R: Float[Array, "num_heads gates_x_head_dim head_dim"],
    b: Float[Array, "gates_x_hidden"],
) -> tuple[Float[Array, "4 hidden_dim"], Float[Array, "4 hidden_dim"]]:
    """Executes a single recurrent step of the sLSTM cell.

    This function computes the linear recurrent step for the sLSTM cell given the inputs
    at the current timestep, the recurrent weights, biases, and previous states. It applies
    the recurrent projection and then delegates to `slstm_pointwise` to update the cell states.

    Args:
        Wx_t: The input projection for the current timestep.
        states: The previous states of the cell.
        R: The recurrent kernel weights.
        b: The recurrent biases.

    Returns:
        A tuple containing the updated cell states and the output of the cell at this timestep.
    """
    num_heads = R.shape[0]
    head_dim = R.shape[2]
    num_gates_r = R.shape[1] // head_dim

    y_prev = states[0].reshape(num_heads, 1, head_dim)
    R_t = R.transpose(0, 2, 1).reshape(num_heads, head_dim, num_gates_r * head_dim)
    Ry = (y_prev @ R_t).reshape(num_heads, num_gates_r, head_dim)
    Ry = Ry.transpose(1, 0, 2).reshape(-1)

    raw = Wx_t + Ry + b
    return slstm_pointwise(raw, states)


def slstm_forward_scan(
    Wx_seq: Float[Array, "seq_len gates_x_hidden"],
    states: Float[Array, "4 hidden_dim"],
    R: Float[Array, "num_heads gates_x_head_dim head_dim"],
    b: Float[Array, "gates_x_hidden"],
) -> tuple[Float[Array, "seq_len hidden_dim"], Float[Array, "4 hidden_dim"]]:
    """Applies the sLSTM recurrent step across an entire sequence.

    This function utilizes `jax.lax.scan` to efficiently apply the `slstm_recurrent_step`
    across a sequence of inputs. This is the preferred method for processing full sequences
    during training or batch evaluation.

    Args:
        Wx_seq: The sequence of input projections over time.
        states: The initial states of the cell.
        R: The recurrent kernel weights.
        b: The recurrent biases.

    Returns:
        A tuple containing the sequence of outputs and the final cell states after processing
        the entire sequence.
    """

    def scan_fn(states, Wx_t):
        new_states, gates = slstm_recurrent_step(Wx_t, states, R, b)
        return new_states, new_states[0]

    final_states, y_seq = jax.lax.scan(scan_fn, states, Wx_seq)
    return y_seq, final_states


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
) -> Float[Array, "num_heads seq_len head_dim"]:
    """Computes the parallel stabilized matrix LSTM (mLSTM) mixing operation.

    This function performs the parallel mixing of queries, keys, and values using the
    stabilized formulation of the mLSTM cell. It avoids sequential recurrence by computing
    the mixing via associative prefix sums or cumulative operations on the gates, heavily
    relying on parallel hardware for efficient execution.

    Args:
        queries: The projected query vectors.
        keys: The projected key vectors.
        values: The projected value vectors.
        igate_preact: Pre-activations for the input gate.
        fgate_preact: Pre-activations for the forget gate.
        lower_triangular_matrix: Optional causal mask matrix.
        stabilize_rowwise: Whether to apply row-wise stabilization to avoid overflow.
        eps: Small constant for numerical stability.
        **kwargs: Additional keyword arguments.

    Returns:
        The output sequence after applying the parallel mLSTM mixing.
    """
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
    c_state: Float[Array, "num_heads head_dim head_dim"],
    n_state: Float[Array, "num_heads head_dim 1"],
    m_state: Float[Array, "num_heads 1 1"],
    q: Float[Array, "num_heads 1 head_dim"],
    k: Float[Array, "num_heads 1 head_dim"],
    v: Float[Array, "num_heads 1 head_dim"],
    igate_preact: Float[Array, "num_heads 1 1"],
    fgate_preact: Float[Array, "num_heads 1 1"],
    eps: float = 1e-6,
    **kwargs,
) -> tuple[
    Float[Array, "num_heads 1 head_dim"],
    tuple[
        Float[Array, "num_heads head_dim head_dim"],
        Float[Array, "num_heads head_dim 1"],
        Float[Array, "num_heads 1 1"],
    ],
]:
    """Executes a single stabilized recurrent step for the matrix LSTM (mLSTM).

    This function computes a single timestep update for the mLSTM cell using the
    recurrent formulation. It updates the covariance matrix state (c_state), normalizer
    state (n_state), and maximum state (m_state) to compute the cell's output. This is
    typically used during autoregressive inference where parallel processing is not possible.

    Args:
        c_state: The covariance matrix state.
        n_state: The normalizer state.
        m_state: The maximum state used for stabilization.
        q: The projected query vector for the current timestep.
        k: The projected key vector for the current timestep.
        v: The projected value vector for the current timestep.
        igate_preact: Pre-activation for the input gate.
        fgate_preact: Pre-activation for the forget gate.
        eps: Small constant for numerical stability.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the output at this timestep and the updated tuple of states.
    """
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
    """A Matrix Long Short-Term Memory (mLSTM) cell module.

    The mLSTM cell represents the memory mechanism using a covariance matrix rather than
    a scalar cell state. It allows for highly parallelizable training and large memory
    capacity. This module manages the internal gates, layer normalizations, and can operate
    both over full sequences in parallel and step-by-step for inference.
    """

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

        igate: eqx.nn.Linear = eqx.nn.Linear(
            3 * embedding_dim, num_heads, key=ikey, dtype=dtype
        )
        igate = eqx.tree_at(
            lambda l: l.weight, igate, jnp.zeros_like(igate.weight, dtype=dtype)
        )
        self.igate = eqx.tree_at(
            lambda l: l.bias,
            igate,
            jnp.linspace(start=3.0, stop=6.0, num=len(igate.bias), dtype=dtype),
        )

        fgate: eqx.nn.Linear = eqx.nn.Linear(
            3 * embedding_dim, num_heads, key=fkey, dtype=dtype
        )
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
    ) -> Float[Array, "seq_len embed_dim"]:
        """Processes a full sequence of queries, keys, and values in parallel.

        This method invokes the parallel stabilized formulation of the mLSTM to
        efficiently process entire sequences at once, bypassing the sequential bottleneck
        of standard recurrent neural networks.

        Args:
            q: The input sequence of queries.
            k: The input sequence of keys.
            v: The input sequence of values.

        Returns:
            The sequence of cell outputs after mixing and applying layer normalization.
        """
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
        q: Float[Array, "1 embed_dim"],
        k: Float[Array, "1 embed_dim"],
        v: Float[Array, "1 embed_dim"],
        cell_state: tuple[
            Float[Array, "num_heads head_dim head_dim"],
            Float[Array, "num_heads head_dim 1"],
            Float[Array, "num_heads 1 1"],
        ]
        | None = None,
    ) -> tuple[
        Float[Array, "1 embed_dim"],
        tuple[
            Float[Array, "num_heads head_dim head_dim"],
            Float[Array, "num_heads head_dim 1"],
            Float[Array, "num_heads 1 1"],
        ],
    ]:
        """Processes a single timestep for the cell.

        This method updates the cell state and computes the output for a single input timestep.
        It is mainly utilized during autoregressive generation where inputs are provided one by one.

        Args:
            q: The query vector for a single timestep.
            k: The key vector for a single timestep.
            v: The value vector for a single timestep.
            cell_state: The previous states of the cell. If None, initializes new empty states.

        Returns:
            A tuple containing the cell output for this timestep and the updated cell states.
        """
        S, _ = q.shape
        assert S == 1

        if cell_state is None:
            cell_state = self.init_state()

        if_gate_input = jnp.concatenate([q, k, v], axis=-1)
        q = q.reshape(S, self.num_heads, -1).transpose(1, 0, 2)
        k = k.reshape(S, self.num_heads, -1).transpose(1, 0, 2)
        v = v.reshape(S, self.num_heads, -1).transpose(1, 0, 2)

        igate_preact = eqx.filter_vmap(self.igate)(if_gate_input)
        igate_preact = jnp.expand_dims(igate_preact, axis=-1).transpose(1, 0, 2)

        fgate_preact = eqx.filter_vmap(self.fgate)(if_gate_input)
        fgate_preact = jnp.expand_dims(fgate_preact, axis=-1).transpose(1, 0, 2)

        c_state, n_state, m_state = cell_state

        h_state, cell_state = recurrent_step_stabilized_simple(
            c_state=c_state,
            n_state=n_state,
            m_state=m_state,
            q=q,
            k=k,
            v=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
        )
        h_state = h_state.transpose(1, 0, 2).reshape(S, -1)
        h_state_norm = eqx.filter_vmap(self.outnorm)(h_state)
        return h_state_norm, cell_state

    def init_state(
        self,
    ) -> tuple[
        Float[Array, "num_heads head_dim head_dim"],
        Float[Array, "num_heads head_dim 1"],
        Float[Array, "num_heads 1 1"],
    ]:
        """Initializes and returns the empty states for the module.

        Returns:
            A tuple of tensors initialized to zero, representing the initial covariance matrix,
            normalizer state, and maximum state.
        """
        head_dim = self.embedding_dim // self.num_heads
        return (
            jnp.zeros((self.num_heads, head_dim, head_dim)),
            jnp.zeros((self.num_heads, head_dim, 1)),
            jnp.zeros((self.num_heads, 1, 1)),
        )


class mLSTMLayer(eqx.Module):
    """A complete mLSTM layer block.

    This module encapsulates a full mLSTM block, which includes an initial up-projection
    into an inner embedding dimension, a 1D causal convolution for local mixing,
    the core `mLSTMCell` for sequence mixing, an output gate branch, and a final
    down-projection back to the original embedding dimension. It behaves as a complete
    sub-layer within a larger transformer-like architecture.
    """

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

    def __call__(
        self, x: Float[Array, "seq_len embed_dim"], key: PRNGKeyArray | None
    ) -> Float[Array, "seq_len embed_dim"]:
        """Performs a forward pass of the layer over an entire input sequence.

        This method processes the input sequence through the up-projection, causal
        convolution, parallel mLSTM cell, and down-projection.

        Args:
            x: The input sequence of embeddings.
            key: PRNG key for dropout operations.

        Returns:
            The output sequence after passing through the full layer.
        """
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
        cell_state: tuple[
            Float[Array, "num_heads head_dim head_dim"],
            Float[Array, "num_heads head_dim 1"],
            Float[Array, "num_heads 1 1"],
        ]
        | None = None,
        conv_state: tuple[Array, ...] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[
        Float[Array, "1 embed_dim"],
        tuple[
            tuple[
                Float[Array, "num_heads head_dim head_dim"],
                Float[Array, "num_heads head_dim 1"],
                Float[Array, "num_heads 1 1"],
            ],
            tuple[Array, ...],
        ],
    ]:
        """Performs a single step forward pass.

        Executes a single step update of the layer, including the stateful 1D convolution
        and the stateful mLSTM cell. This is primarily used for autoregressive decoding.

        Args:
            x: The input embedding for a single timestep.
            cell_state: The previous states of the mLSTM cell.
            conv_state: The previous states of the 1D causal convolution.
            key: PRNG key for dropout.

        Returns:
            A tuple containing the output embedding for this timestep and the updated tuples
            of cell and convolution states.
        """
        if cell_state is None:
            cell_state = self.mlstm_cell.init_state()
        if conv_state is None:
            conv_state = self.conv1d.init_state()

        x_inner = eqx.filter_vmap(self.proj_up)(x)
        x_mlstm, z = jnp.split(x_inner, [self.inner_embedding_dim], axis=-1)

        x_mlstm_conv, conv_state = self.conv1d.step(x_mlstm, conv_state=conv_state)
        x_mlstm_conv_act = jax.nn.silu(x_mlstm_conv)

        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        h_tilde_state, cell_state = self.mlstm_cell.step(
            q=q, k=k, v=v, cell_state=cell_state
        )
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        h_state = h_tilde_state_skip * jax.nn.silu(z)

        y = eqx.filter_vmap(self.proj_down)(h_state)
        y = self.dropout(y, key=key, inference=self.inference)

        return y, (cell_state, conv_state)

    def init_state(
        self,
    ) -> tuple[
        tuple[
            Float[Array, "num_heads head_dim head_dim"],
            Float[Array, "num_heads head_dim 1"],
            Float[Array, "num_heads 1 1"],
        ],
        tuple[Array, ...],
    ]:
        """Initializes and returns the empty states for the layer.

        Returns:
            A tuple containing the initial states for both the internal `mLSTMCell` and the `CausalConv1d`.
        """
        return (self.mlstm_cell.init_state(), self.conv1d.init_state())


class sLSTMCell(eqx.Module):
    """A Scalar Long Short-Term Memory (sLSTM) cell module.

    The sLSTM cell acts as a more traditional LSTM cell but with stabilized exponential
    gating and normalization. It relies on sequential processing and maintains a scalar
    memory state per hidden dimension, updated via standard recurrence.
    """

    hidden_size: int
    num_heads: int
    num_gates: int
    recurrent_kernel: Array
    bias: Array

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_gates = 4
        head_dim = hidden_size // num_heads

        self.recurrent_kernel = jnp.zeros(
            (num_heads, head_dim, self.num_gates, head_dim),
            dtype=dtype,
        )
        self.bias = jnp.zeros(
            (num_heads, self.num_gates, head_dim),
            dtype=dtype,
        )

    def _to_internal_R(self, R: Array) -> Array:
        NH, DH, NG, DH2 = R.shape
        return R.transpose(0, 2, 3, 1).reshape(NH, NG * DH2, DH)

    def _to_internal_b(self, b: Array) -> Array:
        return b.transpose(1, 0, 2).reshape(-1)

    def __call__(
        self,
        x: Float[Array, "seq_len gates_x_hidden"],
        state: Float[Array, "4 hidden_dim"] | None = None,
    ) -> tuple[Float[Array, "seq_len hidden_dim"], Float[Array, "4 hidden_dim"]]:
        """Processes an entire sequence of inputs through the cell.

        This method utilizes `jax.lax.scan` internally to efficiently compute the
        recurrent updates across the entire sequence.

        Args:
            x: The sequence of projected inputs.
            state: The initial states of the cell. If None, initializes empty states.

        Returns:
            A tuple containing the sequence of outputs and the final cell states.
        """
        if state is None:
            state = jnp.zeros((4, self.hidden_size))

        R = self._to_internal_R(self.recurrent_kernel)
        b = self._to_internal_b(self.bias)

        y_seq, final_state = slstm_forward_scan(x, state, R, b)
        return y_seq, final_state

    def step(
        self,
        x: Float[Array, "gates_x_hidden"],
        cell_state: Float[Array, "4 hidden_dim"] | None = None,
    ) -> tuple[Float[Array, "hidden_dim"], Float[Array, "4 hidden_dim"]]:
        """Processes a single timestep for the cell.

        Updates the internal scalar memory state for a single timestep input.

        Args:
            x: The projected inputs for the current timestep.
            cell_state: The previous states of the cell. If None, initializes empty states.

        Returns:
            A tuple containing the output at this timestep and the updated states.
        """
        if cell_state is None:
            cell_state = self.init_state()

        R = self._to_internal_R(self.recurrent_kernel)
        b = self._to_internal_b(self.bias)

        new_state, gates = slstm_recurrent_step(x, cell_state, R, b)
        return new_state[0], new_state

    def init_state(self) -> Float[Array, "4 hidden_dim"]:
        """Initializes and returns the empty states for the cell.

        Returns:
            A tensor of initialized zero states for the sLSTM cell.
        """
        return jnp.zeros((4, self.hidden_size))


class sLSTMLayer(eqx.Module):
    """A complete sLSTM layer implementation.

    This layer wraps the `sLSTMCell` with optional 1D causal convolution and computes
    the necessary projections for the input, forget, output, and update gates. It serves
    as a complete sub-layer component for sequential data modeling.
    """

    conv1d: CausalConv1d | None
    fgate: LinearHeadwiseExpand
    igate: LinearHeadwiseExpand
    zgate: LinearHeadwiseExpand
    ogate: LinearHeadwiseExpand
    slstm_cell: sLSTMCell
    outnorm: ResidualLayerNorm
    dropout: eqx.nn.Dropout
    inference: bool
    embedding_dim: int
    conv1d_kernel_size: int

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        conv1d_kernel_size: int,
        dropout_p: float,
        *,
        key: PRNGKeyArray,
        inference: bool = False,
        dtype: Any | None = None,
    ):
        self.embedding_dim = embedding_dim
        self.conv1d_kernel_size = conv1d_kernel_size
        self.inference = inference

        if conv1d_kernel_size > 0:
            key, convkey = jax.random.split(key)
            self.conv1d = CausalConv1d(
                feature_dim=embedding_dim,
                kernel_size=conv1d_kernel_size,
                channel_mixing=False,
                key=convkey,
                dtype=dtype,
            )
        else:
            self.conv1d = None

        key, fkey, ikey, zkey, okey = jax.random.split(key, 5)
        self.fgate = LinearHeadwiseExpand(
            in_features=embedding_dim,
            out_features=embedding_dim,
            num_heads=num_heads,
            use_bias=False,
            key=fkey,
            dtype=dtype,
        )
        self.igate = LinearHeadwiseExpand(
            in_features=embedding_dim,
            out_features=embedding_dim,
            num_heads=num_heads,
            use_bias=False,
            key=ikey,
            dtype=dtype,
        )
        self.zgate = LinearHeadwiseExpand(
            in_features=embedding_dim,
            out_features=embedding_dim,
            num_heads=num_heads,
            use_bias=False,
            key=zkey,
            dtype=dtype,
        )
        self.ogate = LinearHeadwiseExpand(
            in_features=embedding_dim,
            out_features=embedding_dim,
            num_heads=num_heads,
            use_bias=False,
            key=okey,
            dtype=dtype,
        )

        key, cellkey = jax.random.split(key)
        self.slstm_cell = sLSTMCell(
            hidden_size=embedding_dim,
            num_heads=num_heads,
            key=cellkey,
            dtype=dtype,
        )

        self.outnorm = ResidualLayerNorm(embedding_dim, use_bias=False, dtype=dtype)
        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(
        self,
        x: Float[Array, "seq_len embed_dim"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len embed_dim"]:
        """Performs a forward pass of the layer over an entire input sequence.

        Args:
            x: The input sequence of embeddings.
            key: PRNG key for dropout operations.

        Returns:
            The output sequence after being processed by the optional convolution, gates,
            sLSTM cell, dropout, and normalization.
        """
        if self.conv1d is not None:
            x_conv = self.conv1d(x)
            assert isinstance(x_conv, Array)
            x_conv = jax.nn.silu(x_conv)
        else:
            x_conv = x

        i = self.igate(x_conv)
        f = self.fgate(x_conv)
        z = self.zgate(x)
        o = self.ogate(x)

        cell_input = jnp.concatenate([i, f, z, o], axis=-1)
        y, _ = self.slstm_cell(cell_input)

        y = self.dropout(y, key=key, inference=self.inference)
        y = eqx.filter_vmap(self.outnorm)(y)

        return y

    def step(
        self,
        x: Float[Array, "1 embed_dim"],
        cell_state: Float[Array, "4 hidden_dim"] | None = None,
        conv_state: tuple[Array, ...] = (),
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[
        Float[Array, "1 embed_dim"],
        tuple[Float[Array, "4 hidden_dim"], tuple[Array, ...]],
    ]:
        """Performs a single step forward pass.

        Executes a step of the optional 1D convolution and the sLSTM cell, updating
        and returning their respective states.

        Args:
            x: The input embedding for a single timestep.
            cell_state: The previous states of the sLSTM cell.
            conv_state: The previous states of the 1D causal convolution.
            key: PRNG key for dropout.

        Returns:
            A tuple containing the output embedding and the updated tuples of cell and convolution states.
        """
        if cell_state is None:
            cell_state = self.slstm_cell.init_state()

        if self.conv1d is not None:
            x_conv, conv_state = self.conv1d.step(
                x, conv_state=conv_state if conv_state else None
            )
            x_conv = jax.nn.silu(x_conv)
        else:
            x_conv = x

        i = self.igate(x_conv)
        f = self.fgate(x_conv)
        z = self.zgate(x)
        o = self.ogate(x)

        cell_input = jnp.concatenate([i, f, z, o], axis=-1).squeeze(0)
        y, cell_state = self.slstm_cell.step(cell_input, cell_state=cell_state)

        y = jnp.expand_dims(y, axis=0)
        y = self.dropout(y, key=key, inference=self.inference)
        y = eqx.filter_vmap(self.outnorm)(y)

        return y, (cell_state, conv_state)

    def init_state(self) -> tuple[Float[Array, "4 hidden_dim"], tuple[Array, ...]]:
        """Initializes and returns the empty states for the layer.

        Returns:
            A tuple containing the initial states for both the internal `sLSTMCell` and the optional `CausalConv1d`.
        """
        conv_st = self.conv1d.init_state() if self.conv1d is not None else ()
        return (self.slstm_cell.init_state(), conv_st)


class xLSTMBlock(eqx.Module):
    """An extended LSTM (xLSTM) residual block.

    This module encapsulates an entire xLSTM block, consisting of a core xLSTM layer
    (either an `mLSTMLayer` or an `sLSTMLayer`) followed by an optional feed-forward network (FFN).
    It applies pre-layer normalization and residual connections around both the xLSTM
    layer and the FFN, forming a complete building block for deep xLSTM architectures.
    """

    xlstm_norm: ResidualLayerNorm
    xlstm: mLSTMLayer | sLSTMLayer
    ffn_norm: ResidualLayerNorm | None
    ffn: GatedFeedForward | None

    def __init__(
        self,
        embedding_dim: int,
        xlstm_layer: mLSTMLayer | sLSTMLayer,
        ffn: GatedFeedForward | None = None,
        *,
        dtype: Any | None = None,
    ):
        self.xlstm_norm = ResidualLayerNorm(embedding_dim, use_bias=False, dtype=dtype)
        self.xlstm = xlstm_layer

        if ffn is not None:
            self.ffn_norm = ResidualLayerNorm(
                embedding_dim, use_bias=False, dtype=dtype
            )
            self.ffn = ffn
        else:
            self.ffn_norm = None
            self.ffn = None

    def __call__(
        self,
        x: Float[Array, "seq_len embed_dim"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len embed_dim"]:
        """Performs a forward pass of the block over an entire input sequence.

        Args:
            x: The input sequence of embeddings.
            key: PRNG key for dropout operations within the layer and FFN.

        Returns:
            The output sequence after passing through the residual block.
        """
        key1, key2 = (None, None) if key is None else jax.random.split(key)
        x = x + self.xlstm(eqx.filter_vmap(self.xlstm_norm)(x), key=key1)
        if self.ffn is not None:
            x = x + self.ffn(eqx.filter_vmap(self.ffn_norm)(x), key=key2)
        return x

    def step(
        self,
        x: Float[Array, "1 embed_dim"],
        xlstm_state: tuple[Any, tuple[Array, ...]] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "1 embed_dim"], tuple[Any, tuple[Array, ...]]]:
        """Performs a single step forward pass.

        Executes a single stateful step through the xLSTM layer and applies the FFN.
        Maintains and updates the underlying states of the recurrent components.

        Args:
            x: The input embedding for a single timestep.
            xlstm_state: The previous states of the underlying xLSTM layer.
            key: PRNG key for dropout.

        Returns:
            A tuple containing the output embedding and the updated states of the xLSTM layer.
        """
        if xlstm_state is None:
            xlstm_state = self.init_state()

        key1, key2 = (None, None) if key is None else jax.random.split(key)
        cell_state, conv_state = xlstm_state

        x_xlstm, xlstm_state = self.xlstm.step(
            eqx.filter_vmap(self.xlstm_norm)(x),
            cell_state=cell_state,
            conv_state=conv_state,
            key=key1,
        )
        x = x + x_xlstm
        if self.ffn is not None:
            x = x + self.ffn(eqx.filter_vmap(self.ffn_norm)(x), key=key2)
        return x, xlstm_state

    def init_state(self) -> tuple[Any, tuple[Array, ...]]:
        """Initializes and returns the underlying states for the block.

        Delegates the state initialization to the underlying xLSTM layer.

        Returns:
            The initial states for the components within the block.
        """
        return self.xlstm.init_state()
