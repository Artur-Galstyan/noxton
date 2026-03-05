import equinox as eqx
import jax
import jax.numpy as jnp

from noxton.nn import GatedFeedForward, mLSTMLayer, sLSTMLayer, xLSTMBlock

embed_dim = 16
seq_len = 8
key = jax.random.key(42)

key, k1, k2, k3 = jax.random.split(key, 4)
mlstm_layer = mLSTMLayer(
    embedding_dim=embed_dim,
    inner_embedding_dim=32,
    qkv_proj_blocksize=4,
    use_bias=False,
    conv1d_kernel_size=4,
    max_seq_len=seq_len,
    num_heads=4,
    dropout_p=0.0,
    key=k1,
)
ffn = GatedFeedForward(
    embedding_dim=embed_dim,
    proj_up_dim=32,
    key=k2,
)
block = xLSTMBlock(
    embedding_dim=embed_dim,
    xlstm_layer=mlstm_layer,
    ffn=ffn,
)

x = jnp.ones(shape=(seq_len, embed_dim))
y = block(x, key=k3)
print(f"mLSTM block forward: {y.shape=}")

x_step = jnp.ones(shape=(1, embed_dim))
state = None
for i in range(3):
    key, subkey = jax.random.split(key)
    y_step, state = block.step(x_step, xlstm_state=state, key=subkey)
    print(f"mLSTM block step {i}: {y_step.shape=}")

print()

key, k4, k5 = jax.random.split(key, 3)
slstm_layer = sLSTMLayer(
    embedding_dim=embed_dim,
    num_heads=4,
    conv1d_kernel_size=4,
    dropout_p=0.0,
    key=k4,
)
block_s = xLSTMBlock(
    embedding_dim=embed_dim,
    xlstm_layer=slstm_layer,
)

y = block_s(x, key=k5)
print(f"sLSTM block forward: {y.shape=}")

jitted_step = eqx.filter_jit(block_s.step)
state = block_s.init_state()
print("EXPECTING ONLY 1 JIT")
for i in range(5):
    key, subkey = jax.random.split(key)
    y_step, state = jitted_step(x_step, xlstm_state=state, key=subkey)
