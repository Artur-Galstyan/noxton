import jax
import jax.numpy as jnp
from jaxtyping import Array


def swiglu(x: Array, axis: int = -1) -> Array:
    """Apply the SwiGLU activation function.

    Splits the input array into two halves along the specified axis, applies
    the Swish activation to the first half, and multiplies it element-wise
    with the second half. This gated activation is commonly used in
    transformer feed-forward blocks.

    Args:
        x: Input array. Its size along ``axis`` must be even.
        axis: Axis along which to split the input into two halves.
            Defaults to ``-1`` (last axis).

    Returns:
        Array of the same dtype as ``x`` with size halved along ``axis``.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1.0, 2.0, -1.0, 0.5])
        >>> swiglu(x)  # splits into [1., 2.] and [-1., 0.5]
        Array([-0.26894143,  0.9526741 ], dtype=float32)

        >>> # 2-D input, split along last axis (default)
        >>> x2d = jnp.ones((3, 4))
        >>> swiglu(x2d).shape
        (3, 2)
    """
    a, b = jnp.split(x, 2, axis=axis)
    return jax.nn.swish(a) * b
