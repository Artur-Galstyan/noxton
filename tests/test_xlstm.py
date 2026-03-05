import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from noxton.nn import (
    CausalConv1d,
    GatedFeedForward,
    LinearHeadwiseExpand,
    ResidualLayerNorm,
)
from noxton.nn.xlstm import (
    mLSTMCell,
    mLSTMLayer,
    parallel_stabilized_simple,
    recurrent_step_stabilized_simple,
    slstm_forward_scan,
    slstm_pointwise,
    slstm_recurrent_step,
    sLSTMCell,
    sLSTMLayer,
    xLSTMBlock,
)

EMBED_DIM = 16
INNER_EMBED_DIM = 32
NUM_HEADS = 4
SEQ_LEN = 8
HEAD_DIM = EMBED_DIM // NUM_HEADS
KEY = jax.random.key(42)


class TestResidualLayerNorm:
    def test_forward_shape(self):
        norm = ResidualLayerNorm(EMBED_DIM, use_bias=False)
        x = jnp.ones(EMBED_DIM)
        assert norm(x).shape == (EMBED_DIM,)

    def test_vmap_shape(self):
        norm = ResidualLayerNorm(EMBED_DIM, use_bias=False)
        x = jnp.ones((SEQ_LEN, EMBED_DIM))
        assert eqx.filter_vmap(norm)(x).shape == (SEQ_LEN, EMBED_DIM)


class TestLinearHeadwiseExpand:
    def test_forward_shape(self):
        layer = LinearHeadwiseExpand(
            in_features=EMBED_DIM,
            out_features=EMBED_DIM,
            num_heads=NUM_HEADS,
            key=KEY,
        )
        x = jnp.ones((SEQ_LEN, EMBED_DIM))
        assert layer(x).shape == (SEQ_LEN, EMBED_DIM)

    def test_expansion_shape(self):
        layer = LinearHeadwiseExpand(
            in_features=EMBED_DIM,
            out_features=EMBED_DIM * 2,
            num_heads=NUM_HEADS,
            key=KEY,
        )
        x = jnp.ones((SEQ_LEN, EMBED_DIM))
        assert layer(x).shape == (SEQ_LEN, EMBED_DIM * 2)


class TestCausalConv1d:
    def test_forward_shape(self):
        layer = CausalConv1d(
            feature_dim=EMBED_DIM, channel_mixing=False, kernel_size=4, key=KEY
        )
        x = jnp.ones((SEQ_LEN, EMBED_DIM))
        y = layer(x)
        assert isinstance(y, Array)
        assert y.shape == (SEQ_LEN, EMBED_DIM)

    def test_step_shape(self):
        layer = CausalConv1d(
            feature_dim=EMBED_DIM, channel_mixing=False, kernel_size=4, key=KEY
        )
        x = jnp.ones((1, EMBED_DIM))
        state = layer.init_state()
        y, state = layer.step(x, conv_state=state)
        assert y.shape == (1, EMBED_DIM)
        assert state[0].shape == (4, EMBED_DIM)

    def test_step_no_rejit(self):
        layer = CausalConv1d(
            feature_dim=EMBED_DIM, channel_mixing=False, kernel_size=4, key=KEY
        )
        x = jnp.ones((1, EMBED_DIM))
        state = layer.init_state()
        step_fn = eqx.filter_jit(layer.step)
        for _ in range(3):
            y, state = step_fn(x, conv_state=state)
        assert y.shape == (1, EMBED_DIM)

    def test_kernel_size_zero(self):
        layer = CausalConv1d(
            feature_dim=EMBED_DIM, channel_mixing=False, kernel_size=0, key=KEY
        )
        x = jnp.ones((SEQ_LEN, EMBED_DIM))
        y = layer(x)

        assert isinstance(y, Array)
        assert y.shape == (SEQ_LEN, EMBED_DIM)


class TestParallelStabilizedSimple:
    def test_output_shape(self):
        q = jnp.ones((NUM_HEADS, SEQ_LEN, HEAD_DIM))
        k = jnp.ones((NUM_HEADS, SEQ_LEN, HEAD_DIM))
        v = jnp.ones((NUM_HEADS, SEQ_LEN, HEAD_DIM))
        igate = jnp.ones((NUM_HEADS, SEQ_LEN, 1))
        fgate = jnp.ones((NUM_HEADS, SEQ_LEN, 1))
        h = parallel_stabilized_simple(q, k, v, igate, fgate)
        assert h.shape == (NUM_HEADS, SEQ_LEN, HEAD_DIM)


class TestRecurrentStepStabilizedSimple:
    def test_output_shape(self):
        q = jnp.ones((NUM_HEADS, 1, HEAD_DIM))
        k = jnp.ones((NUM_HEADS, 1, HEAD_DIM))
        v = jnp.ones((NUM_HEADS, 1, HEAD_DIM))
        igate = jnp.ones((NUM_HEADS, 1, 1))
        fgate = jnp.ones((NUM_HEADS, 1, 1))
        c = jnp.zeros((NUM_HEADS, HEAD_DIM, HEAD_DIM))
        n = jnp.zeros((NUM_HEADS, HEAD_DIM, 1))
        m = jnp.zeros((NUM_HEADS, 1, 1))
        h, (c_new, n_new, m_new) = recurrent_step_stabilized_simple(
            c, n, m, q, k, v, igate, fgate
        )
        assert h.shape == (NUM_HEADS, 1, HEAD_DIM)
        assert c_new.shape == (NUM_HEADS, HEAD_DIM, HEAD_DIM)
        assert n_new.shape == (NUM_HEADS, HEAD_DIM, 1)
        assert m_new.shape == (NUM_HEADS, 1, 1)


class TestMLSTMCell:
    @pytest.fixture
    def cell(self):
        return mLSTMCell(EMBED_DIM, NUM_HEADS, max_seq_len=SEQ_LEN, key=KEY)

    def test_forward_shape(self, cell):
        q = jnp.ones((SEQ_LEN, EMBED_DIM))
        k = jnp.ones((SEQ_LEN, EMBED_DIM))
        v = jnp.ones((SEQ_LEN, EMBED_DIM))
        y = cell(q, k, v)
        assert y.shape == (SEQ_LEN, EMBED_DIM)

    def test_step_shape(self, cell):
        q = jnp.ones((1, EMBED_DIM))
        k = jnp.ones((1, EMBED_DIM))
        v = jnp.ones((1, EMBED_DIM))
        state = cell.init_state()
        y, state = cell.step(q, k, v, cell_state=state)
        assert y.shape == (1, EMBED_DIM)
        assert state[0].shape == (NUM_HEADS, HEAD_DIM, HEAD_DIM)
        assert state[1].shape == (NUM_HEADS, HEAD_DIM, 1)
        assert state[2].shape == (NUM_HEADS, 1, 1)

    def test_step_no_rejit(self, cell):
        q = jnp.ones((1, EMBED_DIM))
        k = jnp.ones((1, EMBED_DIM))
        v = jnp.ones((1, EMBED_DIM))
        state = cell.init_state()
        step_fn = eqx.filter_jit(cell.step)
        for _ in range(3):
            y, state = step_fn(q, k, v, cell_state=state)
        assert y.shape == (1, EMBED_DIM)


class TestMLSTMLayer:
    @pytest.fixture
    def layer(self):
        return mLSTMLayer(
            embedding_dim=EMBED_DIM,
            inner_embedding_dim=INNER_EMBED_DIM,
            qkv_proj_blocksize=4,
            use_bias=False,
            conv1d_kernel_size=4,
            max_seq_len=SEQ_LEN,
            num_heads=NUM_HEADS,
            dropout_p=0.0,
            key=KEY,
        )

    def test_forward_shape(self, layer):
        x = jnp.ones((SEQ_LEN, EMBED_DIM))
        y = layer(x, key=KEY)
        assert y.shape == (SEQ_LEN, EMBED_DIM)

    def test_step_shape(self, layer):
        x = jnp.ones((1, EMBED_DIM))
        state = layer.init_state()
        cell_state, conv_state = state
        y, (cell_state, conv_state) = layer.step(
            x, cell_state=cell_state, conv_state=conv_state, key=KEY
        )
        assert y.shape == (1, EMBED_DIM)

    def test_step_no_rejit(self, layer):
        x = jnp.ones((1, EMBED_DIM))
        state = layer.init_state()
        cell_state, conv_state = state
        step_fn = eqx.filter_jit(layer.step)
        for _ in range(3):
            y, (cell_state, conv_state) = step_fn(
                x, cell_state=cell_state, conv_state=conv_state, key=KEY
            )
        assert y.shape == (1, EMBED_DIM)


class TestSLSTMPointwise:
    def test_output_shape(self):
        raw = jnp.ones(EMBED_DIM * 4)
        states = jnp.zeros((4, EMBED_DIM))
        states = states.at[2].set(1.0)
        new_states, gates = slstm_pointwise(raw, states)
        assert new_states.shape == (4, EMBED_DIM)
        assert gates.shape == (4, EMBED_DIM)


class TestSLSTMRecurrentStep:
    def test_output_shape(self):
        hidden = EMBED_DIM
        num_gates = 4
        head_dim = hidden // NUM_HEADS
        Wx = jnp.ones(num_gates * hidden)
        states = jnp.zeros((4, hidden))
        states = states.at[2].set(1.0)
        R = jnp.zeros((NUM_HEADS, num_gates * head_dim, head_dim))
        b = jnp.zeros(num_gates * hidden)
        new_states, gates = slstm_recurrent_step(Wx, states, R, b)
        assert new_states.shape == (4, hidden)
        assert gates.shape == (4, hidden)


class TestSLSTMForwardScan:
    def test_output_shape(self):
        hidden = EMBED_DIM
        num_gates = 4
        head_dim = hidden // NUM_HEADS
        Wx_seq = jnp.ones((SEQ_LEN, num_gates * hidden))
        states = jnp.zeros((4, hidden))
        states = states.at[2].set(1.0)
        R = jnp.zeros((NUM_HEADS, num_gates * head_dim, head_dim))
        b = jnp.zeros(num_gates * hidden)
        y_seq, final_states = slstm_forward_scan(Wx_seq, states, R, b)
        assert y_seq.shape == (SEQ_LEN, hidden)
        assert final_states.shape == (4, hidden)


class TestSLSTMCell:
    @pytest.fixture
    def cell(self):
        return sLSTMCell(hidden_size=EMBED_DIM, num_heads=NUM_HEADS, key=KEY)

    def test_forward_shape(self, cell):
        x = jnp.ones((SEQ_LEN, 4 * EMBED_DIM))
        state = cell.init_state()
        y, final_state = cell(x, state=state)
        assert y.shape == (SEQ_LEN, EMBED_DIM)
        assert final_state.shape == (4, EMBED_DIM)

    def test_step_shape(self, cell):
        x = jnp.ones(4 * EMBED_DIM)
        state = cell.init_state()
        y, state = cell.step(x, cell_state=state)
        assert y.shape == (EMBED_DIM,)
        assert state.shape == (4, EMBED_DIM)

    def test_step_no_rejit(self, cell):
        x = jnp.ones(4 * EMBED_DIM)
        state = cell.init_state()
        step_fn = eqx.filter_jit(cell.step)
        for _ in range(3):
            y, state = step_fn(x, cell_state=state)
        assert y.shape == (EMBED_DIM,)


class TestSLSTMLayer:
    @pytest.fixture
    def layer(self):
        return sLSTMLayer(
            embedding_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            conv1d_kernel_size=4,
            dropout_p=0.0,
            key=KEY,
        )

    def test_forward_shape(self, layer):
        x = jnp.ones((SEQ_LEN, EMBED_DIM))
        y = layer(x, key=KEY)
        assert y.shape == (SEQ_LEN, EMBED_DIM)

    def test_step_shape(self, layer):
        x = jnp.ones((1, EMBED_DIM))
        cell_state, conv_state = layer.init_state()
        y, (cell_state, conv_state) = layer.step(
            x, cell_state=cell_state, conv_state=conv_state, key=KEY
        )
        assert y.shape == (1, EMBED_DIM)

    def test_step_no_rejit(self, layer):
        x = jnp.ones((1, EMBED_DIM))
        cell_state, conv_state = layer.init_state()
        step_fn = eqx.filter_jit(layer.step)
        for _ in range(3):
            y, (cell_state, conv_state) = step_fn(
                x, cell_state=cell_state, conv_state=conv_state, key=KEY
            )
        assert y.shape == (1, EMBED_DIM)


class TestGatedFeedForward:
    def test_forward_shape(self):
        ffn = GatedFeedForward(embedding_dim=EMBED_DIM, proj_up_dim=32, key=KEY)
        x = jnp.ones((SEQ_LEN, EMBED_DIM))
        y = ffn(x, key=KEY)
        assert y.shape == (SEQ_LEN, EMBED_DIM)


class TestXLSTMBlockMLSTM:
    @pytest.fixture
    def block(self):
        key = jax.random.key(0)
        k1, k2 = jax.random.split(key)
        mlstm_layer = mLSTMLayer(
            embedding_dim=EMBED_DIM,
            inner_embedding_dim=INNER_EMBED_DIM,
            qkv_proj_blocksize=4,
            use_bias=False,
            conv1d_kernel_size=4,
            max_seq_len=SEQ_LEN,
            num_heads=NUM_HEADS,
            dropout_p=0.0,
            key=k1,
        )
        ffn = GatedFeedForward(embedding_dim=EMBED_DIM, proj_up_dim=32, key=k2)
        return xLSTMBlock(embedding_dim=EMBED_DIM, xlstm_layer=mlstm_layer, ffn=ffn)

    def test_forward_shape(self, block):
        x = jnp.ones((SEQ_LEN, EMBED_DIM))
        y = block(x, key=KEY)
        assert y.shape == (SEQ_LEN, EMBED_DIM)

    def test_step_shape(self, block):
        x = jnp.ones((1, EMBED_DIM))
        state = block.init_state()
        y, state = block.step(x, xlstm_state=state, key=KEY)
        assert y.shape == (1, EMBED_DIM)

    def test_step_no_rejit(self, block):
        x = jnp.ones((1, EMBED_DIM))
        state = block.init_state()
        step_fn = eqx.filter_jit(block.step)
        for _ in range(3):
            y, state = step_fn(x, xlstm_state=state, key=KEY)
        assert y.shape == (1, EMBED_DIM)


class TestXLSTMBlockSLSTM:
    @pytest.fixture
    def block(self):
        key = jax.random.key(0)
        slstm_layer = sLSTMLayer(
            embedding_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            conv1d_kernel_size=4,
            dropout_p=0.0,
            key=key,
        )
        return xLSTMBlock(embedding_dim=EMBED_DIM, xlstm_layer=slstm_layer)

    def test_forward_shape(self, block):
        x = jnp.ones((SEQ_LEN, EMBED_DIM))
        y = block(x, key=KEY)
        assert y.shape == (SEQ_LEN, EMBED_DIM)

    def test_step_shape(self, block):
        x = jnp.ones((1, EMBED_DIM))
        state = block.init_state()
        y, state = block.step(x, xlstm_state=state, key=KEY)
        assert y.shape == (1, EMBED_DIM)

    def test_step_no_rejit(self, block):
        x = jnp.ones((1, EMBED_DIM))
        state = block.init_state()
        step_fn = eqx.filter_jit(block.step)
        for _ in range(3):
            y, state = step_fn(x, xlstm_state=state, key=KEY)
        assert y.shape == (1, EMBED_DIM)
