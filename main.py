import jax
import jax.numpy as jnp

from noxton.nn import mLSTMLayer

embed_dim = 16
inner_embed_dim = 32
num_heads = 8
seq_len = 4
qkv_proj_blocksize = 4

layer = mLSTMLayer(
    embedding_dim=embed_dim,
    inner_embedding_dim=inner_embed_dim,
    qkv_proj_blocksize=qkv_proj_blocksize,
    use_bias=False,
    conv1d_kernel_size=4,
    max_seq_len=seq_len,
    num_heads=num_heads,
    dropout_p=0.0,
    key=jax.random.key(42),
)

x = jnp.ones(shape=(seq_len, embed_dim))
key = jax.random.key(0)
y = layer(x, key=key)
print(f"forward: {y.shape=}")

x_step = jnp.ones(shape=(1, embed_dim))
mlstm_state = None
conv_state = None

for i in range(3):
    y_step, (mlstm_state, conv_state) = layer.step(
        x_step, mlstm_state=mlstm_state, conv_state=conv_state, key=key
    )
    print(f"step {i}: {y_step.shape=}")
