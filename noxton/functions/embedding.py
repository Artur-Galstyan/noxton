import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, Float, Int

from noxton.utils import default_floating_dtype


def sinusoidal_embedding(
    t: Int[Array, ""], embedding_size: int, dtype: Any | None = None
) -> Float[Array, " embedding_size"]:
    """Compute a sinusoidal positional embedding for a scalar timestep.

    Encodes a single integer timestep ``t`` into a fixed-size vector using
    alternating sine and cosine functions at geometrically spaced frequencies,
    following the scheme from *Attention Is All You Need* (Vaswani et al., 2017)
    and commonly used in diffusion model timestep conditioning.

    The embedding frequencies are::

        freq_i = exp(-log(10000) * i / (embedding_size / 2))   for i in [0, half_dim)

    The output is the concatenation of ``sin(t * freq)`` and ``cos(t * freq)``.

    Args:
        t: Scalar integer timestep (0-d array).
        embedding_size: Length of the output embedding vector. Must be even.
        dtype: Floating-point dtype for the output. Defaults to the project's
            default floating dtype when ``None``.

    Returns:
        1-D array of shape ``(embedding_size,)`` containing the sinusoidal
        embedding for timestep ``t``.

    Raises:
        ValueError: If ``embedding_size`` is odd.

    Example:
        >>> import jax.numpy as jnp
        >>> t = jnp.array(100)
        >>> emb = sinusoidal_embedding(t, embedding_size=16)
        >>> emb.shape
        (16,)

        >>> # Embeddings for different timesteps are distinct
        >>> emb0 = sinusoidal_embedding(jnp.array(0), embedding_size=8)
        >>> emb1 = sinusoidal_embedding(jnp.array(1), embedding_size=8)
        >>> bool(jnp.any(emb0 != emb1))
        True
    """
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None
    if embedding_size % 2 != 0:
        raise ValueError(f"Embedding size must be even, but got {embedding_size}")

    half_dim = embedding_size // 2
    embedding_freqs = jnp.exp(
        -jnp.log(10000) * jnp.arange(start=0, stop=half_dim, dtype=dtype) / half_dim
    )

    time_args = t * embedding_freqs
    embedding = jnp.concatenate([jnp.sin(time_args), jnp.cos(time_args)], axis=-1)

    return embedding
