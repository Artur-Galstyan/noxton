with open("noxton/nn/xlstm.py", "r") as f:
    content = f.read()

# 1. slstm_pointwise
content = content.replace(
    ") -> tuple[Array, Array]:\n    y, c, n, m = states",
    """) -> tuple[Float[Array, "4 hidden_dim"], Float[Array, "4 hidden_dim"]]:
    \"\"\"Compute sLSTM pointwise gates and states. Update c, n, m.\"\"\"
    y, c, n, m = states""",
)

# 2. slstm_recurrent_step
content = content.replace(
    ") -> tuple[Array, Array]:\n    num_heads = R.shape[0]",
    """) -> tuple[Float[Array, "4 hidden_dim"], Float[Array, "4 hidden_dim"]]:
    \"\"\"Run single recurrent step for sLSTM.\"\"\"
    num_heads = R.shape[0]""",
)

# 3. slstm_forward_scan
content = content.replace(
    ") -> tuple[Array, Array]:\n    def scan_fn(states, Wx_t):",
    """) -> tuple[Float[Array, "seq_len hidden_dim"], Float[Array, "4 hidden_dim"]]:
    \"\"\"Scan sLSTM recurrent step over sequence.\"\"\"
    def scan_fn(states, Wx_t):""",
)

# 4. parallel_stabilized_simple
content = content.replace(
    "    eps: float = 1e-6,\n    **kwargs,\n) -> Array:\n    NH, S, DH = queries.shape",
    """    eps: float = 1e-6,
    **kwargs,
) -> Float[Array, "num_heads seq_len head_dim"]:
    \"\"\"Compute parallel stabilized mLSTM mix.\"\"\"
    NH, S, DH = queries.shape""",
)

# 5. recurrent_step_stabilized_simple
content = content.replace(
    "    eps: float = 1e-6,\n    **kwargs,\n) -> tuple[Array, tuple[Array, Array, Array]]:\n    NH, S, DH = q.shape",
    """    eps: float = 1e-6,
    **kwargs,
) -> tuple[Float[Array, "num_heads 1 head_dim"], tuple[Float[Array, "num_heads head_dim head_dim"], Float[Array, "num_heads head_dim 1"], Float[Array, "num_heads 1 1"]]]:
    \"\"\"Run single recurrent step for mLSTM mix.\"\"\"
    NH, S, DH = q.shape""",
)
content = content.replace(
    "c_state: Array,\n    n_state: Array,\n    m_state: Array,",
    """c_state: Float[Array, "num_heads head_dim head_dim"],
    n_state: Float[Array, "num_heads head_dim 1"],
    m_state: Float[Array, "num_heads 1 1"],""",
)
content = content.replace(
    "igate_preact: Array,\n    fgate_preact: Array,",
    """igate_preact: Float[Array, "num_heads 1 1"],
    fgate_preact: Float[Array, "num_heads 1 1"],""",
)

# 6. mLSTMCell
content = content.replace(
    "class mLSTMCell(eqx.Module):\n    max_seq_len: int",
    """class mLSTMCell(eqx.Module):
    \"\"\"Matrix LSTM cell.\"\"\"
    max_seq_len: int""",
)
content = content.replace(
    '    def __call__(\n        self,\n        q: Float[Array, "seq_len embed_dim"],\n        k: Float[Array, "seq_len embed_dim"],\n        v: Float[Array, "seq_len embed_dim"],\n    ):\n        seq_len, _ = q.shape',
    """    def __call__(
        self,
        q: Float[Array, "seq_len embed_dim"],
        k: Float[Array, "seq_len embed_dim"],
        v: Float[Array, "seq_len embed_dim"],
    ) -> Float[Array, "seq_len embed_dim"]:
        \"\"\"Process full sequence in parallel.\"\"\"
        seq_len, _ = q.shape""",
)
content = content.replace(
    '    def step(\n        self,\n        q: Float[Array, "1 embed_dim"],\n        k: Float[Array, "1 embed_dim"],\n        v: Float[Array, "1 embed_dim"],\n        cell_state: tuple[Array, Array, Array] | None = None,\n    ) -> tuple[Array, tuple[Array, Array, Array]]:\n        S, _ = q.shape',
    """    def step(
        self,
        q: Float[Array, "1 embed_dim"],
        k: Float[Array, "1 embed_dim"],
        v: Float[Array, "1 embed_dim"],
        cell_state: tuple[Float[Array, "num_heads head_dim head_dim"], Float[Array, "num_heads head_dim 1"], Float[Array, "num_heads 1 1"]] | None = None,
    ) -> tuple[Float[Array, "1 embed_dim"], tuple[Float[Array, "num_heads head_dim head_dim"], Float[Array, "num_heads head_dim 1"], Float[Array, "num_heads 1 1"]]]:
        \"\"\"Process single timestep.\"\"\"
        S, _ = q.shape""",
)
content = content.replace(
    "    def init_state(self) -> tuple[Array, Array, Array]:\n        head_dim",
    """    def init_state(self) -> tuple[Float[Array, "num_heads head_dim head_dim"], Float[Array, "num_heads head_dim 1"], Float[Array, "num_heads 1 1"]]:
        \"\"\"Init empty states.\"\"\"
        head_dim""",
)

# 7. mLSTMLayer
content = content.replace(
    "class mLSTMLayer(eqx.Module):\n    proj_up: eqx.nn.Linear",
    """class mLSTMLayer(eqx.Module):
    \"\"\"Full mLSTM layer with projections and conv1d.\"\"\"
    proj_up: eqx.nn.Linear""",
)
content = content.replace(
    '    def __call__(self, x: Float[Array, "seq_len embed_dim"], key: PRNGKeyArray | None):\n        S, _ = x.shape',
    """    def __call__(self, x: Float[Array, "seq_len embed_dim"], key: PRNGKeyArray | None) -> Float[Array, "seq_len embed_dim"]:
        \"\"\"Forward pass full sequence.\"\"\"
        S, _ = x.shape""",
)
content = content.replace(
    '    def step(\n        self,\n        x: Float[Array, "1 embed_dim"],\n        cell_state: tuple[Array, Array, Array] | None = None,\n        conv_state: tuple[Array] | None = None,\n        *,\n        key: PRNGKeyArray | None = None,\n    ) -> tuple[Array, tuple[tuple[Array, Array, Array], tuple[Array]]]:\n        if cell_state is None:',
    """    def step(
        self,
        x: Float[Array, "1 embed_dim"],
        cell_state: tuple[Float[Array, "num_heads head_dim head_dim"], Float[Array, "num_heads head_dim 1"], Float[Array, "num_heads 1 1"]] | None = None,
        conv_state: tuple[Array, ...] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "1 embed_dim"], tuple[tuple[Float[Array, "num_heads head_dim head_dim"], Float[Array, "num_heads head_dim 1"], Float[Array, "num_heads 1 1"]], tuple[Array, ...]]]:
        \"\"\"Forward pass single step.\"\"\"
        if cell_state is None:""",
)
content = content.replace(
    "    def init_state(self) -> tuple[tuple, tuple]:\n        return",
    """    def init_state(self) -> tuple[tuple[Float[Array, "num_heads head_dim head_dim"], Float[Array, "num_heads head_dim 1"], Float[Array, "num_heads 1 1"]], tuple[Array, ...]]:
        \"\"\"Init empty cell and conv states.\"\"\"
        return""",
)

# 8. sLSTMCell
content = content.replace(
    "class sLSTMCell(eqx.Module):\n    hidden_size: int",
    """class sLSTMCell(eqx.Module):
    \"\"\"Scalar LSTM cell.\"\"\"
    hidden_size: int""",
)
content = content.replace(
    '    def __call__(\n        self,\n        x: Float[Array, "seq_len gates_x_hidden"],\n        state: Array | None = None,\n    ) -> tuple[Array, Array]:\n        if state is None:',
    """    def __call__(
        self,
        x: Float[Array, "seq_len gates_x_hidden"],
        state: Float[Array, "4 hidden_dim"] | None = None,
    ) -> tuple[Float[Array, "seq_len hidden_dim"], Float[Array, "4 hidden_dim"]]:
        \"\"\"Process full sequence.\"\"\"
        if state is None:""",
)
content = content.replace(
    '    def step(\n        self,\n        x: Float[Array, "gates_x_hidden"],\n        cell_state: Array | None = None,\n    ) -> tuple[Array, Array]:\n        if cell_state is None:',
    """    def step(
        self,
        x: Float[Array, "gates_x_hidden"],
        cell_state: Float[Array, "4 hidden_dim"] | None = None,
    ) -> tuple[Float[Array, "hidden_dim"], Float[Array, "4 hidden_dim"]]:
        \"\"\"Process single timestep.\"\"\"
        if cell_state is None:""",
)
content = content.replace(
    "    def init_state(self) -> Array:\n        return",
    """    def init_state(self) -> Float[Array, "4 hidden_dim"]:
        \"\"\"Init empty states.\"\"\"
        return""",
)

# 9. sLSTMLayer
content = content.replace(
    "class sLSTMLayer(eqx.Module):\n    conv1d: CausalConv1d | None",
    """class sLSTMLayer(eqx.Module):
    \"\"\"Full sLSTM layer with gates and conv1d.\"\"\"
    conv1d: CausalConv1d | None""",
)
content = content.replace(
    '    def __call__(\n        self,\n        x: Float[Array, "seq_len embed_dim"],\n        *,\n        key: PRNGKeyArray | None = None,\n    ) -> Array:\n        if self.conv1d is not None:',
    """    def __call__(
        self,
        x: Float[Array, "seq_len embed_dim"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len embed_dim"]:
        \"\"\"Forward pass full sequence.\"\"\"
        if self.conv1d is not None:""",
)
content = content.replace(
    '    def step(\n        self,\n        x: Float[Array, "1 embed_dim"],\n        cell_state: Array | None = None,\n        conv_state: tuple = (),\n        *,\n        key: PRNGKeyArray | None = None,\n    ) -> tuple[Array, tuple[Array, tuple]]:\n        if cell_state is None:',
    """    def step(
        self,
        x: Float[Array, "1 embed_dim"],
        cell_state: Float[Array, "4 hidden_dim"] | None = None,
        conv_state: tuple[Array, ...] = (),
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "1 embed_dim"], tuple[Float[Array, "4 hidden_dim"], tuple[Array, ...]]]:
        \"\"\"Forward pass single step.\"\"\"
        if cell_state is None:""",
)
content = content.replace(
    "    def init_state(self) -> tuple[Array, tuple]:\n        conv_st",
    """    def init_state(self) -> tuple[Float[Array, "4 hidden_dim"], tuple[Array, ...]]:
        \"\"\"Init empty cell and conv states.\"\"\"
        conv_st""",
)

# 10. xLSTMBlock
content = content.replace(
    "class xLSTMBlock(eqx.Module):\n    xlstm_norm: ResidualLayerNorm",
    """class xLSTMBlock(eqx.Module):
    \"\"\"xLSTM residual block.\"\"\"
    xlstm_norm: ResidualLayerNorm""",
)
content = content.replace(
    '    def __call__(\n        self,\n        x: Float[Array, "seq_len embed_dim"],\n        *,\n        key: PRNGKeyArray | None = None,\n    ) -> Array:\n        key1, key2',
    """    def __call__(
        self,
        x: Float[Array, "seq_len embed_dim"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len embed_dim"]:
        \"\"\"Forward pass full sequence.\"\"\"
        key1, key2""",
)
content = content.replace(
    '    def step(\n        self,\n        x: Float[Array, "1 embed_dim"],\n        xlstm_state: tuple | None = None,\n        *,\n        key: PRNGKeyArray | None = None,\n    ) -> tuple[Array, tuple]:\n        if xlstm_state is None:',
    """    def step(
        self,
        x: Float[Array, "1 embed_dim"],
        xlstm_state: tuple[Any, tuple[Array, ...]] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "1 embed_dim"], tuple[Any, tuple[Array, ...]]]:
        \"\"\"Forward pass single step.\"\"\"
        if xlstm_state is None:""",
)
content = content.replace(
    "    def init_state(self):\n        return self.xlstm.init_state()",
    """    def init_state(self) -> tuple[Any, tuple[Array, ...]]:
        \"\"\"Init layer states.\"\"\"
        return self.xlstm.init_state()""",
)

with open("noxton/nn/xlstm.py", "w") as f:
    f.write(content)
