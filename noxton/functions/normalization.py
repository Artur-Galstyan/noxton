import jax.numpy as jnp
from jaxtyping import Array


def normalize(x: Array, p: int = 2, axis: int = 1, eps: float = 1e-12) -> Array:
    """Normalize an array along an axis using an Lp norm.

    Computes ``x / max(||x||_p, eps)`` along ``axis``, where the denominator is
    clamped to at least ``eps`` to prevent division by zero.

    Args:
        x: Input array to normalize.
        p: Order of the norm. ``p=2`` gives the Euclidean (L2) norm,
            ``p=1`` gives the Manhattan (L1) norm, etc. Defaults to ``2``.
        axis: Axis along which to compute the norm and normalize.
            Defaults to ``1``.
        eps: Small constant added to the denominator for numerical stability.
            Defaults to ``1e-12``.

    Returns:
        Array of the same shape and dtype as ``x`` with unit Lp norm along
        ``axis`` (or norm equal to ``eps`` when the input norm is smaller).

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[3.0, 4.0], [0.0, 0.0]])
        >>> normalize(x)          # L2 norm along axis=1
        Array([[0.6, 0.8],
               [0. , 0. ]], dtype=float32)

        >>> normalize(x, p=1)     # L1 norm along axis=1
        Array([[0.42857143, 0.5714286 ],
               [0.        , 0.        ]], dtype=float32)
    """
    norm = jnp.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    norm = jnp.maximum(norm, eps)
    output = x / norm

    return output
