import equinox as eqx
from beartype.typing import Any
from jaxtyping import Array, PRNGKeyArray

from noxton.utils import default_floating_dtype


class EmbeddingWithPadding(eqx.Module):
    """Embedding table that zeros out embeddings for a padding index.

    Wraps ``eqx.nn.Embedding`` and multiplies every looked-up embedding by a
    binary mask that is ``0`` wherever the input index equals
    ``padding_idx`` and ``1`` elsewhere.  This ensures that padding tokens
    contribute a zero vector to downstream computations.

    Args:
        num_embeddings: Size of the vocabulary (number of embedding vectors).
        embedding_dim: Dimensionality of each embedding vector.
        padding_idx: Token index whose embedding is forced to zero.
            Defaults to ``0``.
        key: JAX PRNG key for embedding table initialisation.
        dtype: Floating-point dtype for the embedding table.  Defaults to the
            project default when ``None``.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> emb = EmbeddingWithPadding(num_embeddings=10, embedding_dim=4, key=key)
        >>> ids = jnp.array([0, 1, 2, 0])  # 0 is the padding index
        >>> out = emb(ids)
        >>> out.shape
        (4, 4)
        >>> bool(jnp.all(out[0] == 0))  # padding rows are zero
        True
        >>> bool(jnp.all(out[3] == 0))
        True
    """

    embed: eqx.nn.Embedding
    padding_idx: int = eqx.field(static=True)

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = 0,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.embed = eqx.nn.Embedding(
            num_embeddings, embedding_dim, key=key, dtype=dtype
        )
        self.padding_idx = padding_idx

    def __call__(self, x: Array):
        out = self.embed(x)
        mask = (x != self.padding_idx).astype(out.dtype)
        return out * mask[..., None]


class EmbeddingBag(eqx.Module):
    """Embedding bag that sums a bag of token embeddings into a single vector.

    Looks up each token index in the bag using ``EmbeddingWithPadding``
    (so padding tokens contribute zero), then sums the resulting embeddings
    into a single ``(embedding_dim,)`` vector.  This is analogous to
    ``torch.nn.EmbeddingBag`` with ``mode="sum"`` and ``padding_idx`` support.

    Args:
        num_embeddings: Size of the vocabulary (number of embedding vectors).
        embedding_dim: Dimensionality of each embedding vector.
        padding_idx: Token index whose embedding is forced to zero before
            summing.  Defaults to ``0``.
        key: JAX PRNG key for embedding table initialisation.
        dtype: Floating-point dtype for the embedding table.  Defaults to the
            project default when ``None``.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> bag = EmbeddingBag(num_embeddings=10, embedding_dim=4, key=key)
        >>> ids = jnp.array([1, 3, 5])  # bag of 3 token ids
        >>> out = bag(ids)
        >>> out.shape
        (4,)
    """

    embed: EmbeddingWithPadding

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = 0,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.embed = EmbeddingWithPadding(
            num_embeddings, embedding_dim, padding_idx=padding_idx, key=key, dtype=dtype
        )

    def __call__(self, x):
        looked_up = eqx.filter_vmap(self.embed)(x)
        return looked_up.sum(axis=0)
