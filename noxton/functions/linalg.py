import jax.numpy as jnp
from jaxtyping import Array


def graham_schmidt(x_axis: Array, xy_plane: Array, eps: float = 1e-12):
    """
    Constructs an orthonormal basis (rotation matrix) using the Gram-Schmidt process.

    Given a primary `x_axis` vector and a secondary `xy_plane` vector, this function
    computes a 3D orthonormal basis. The first basis vector is the normalized `x_axis`.
    The second is the orthogonalized `xy_plane` vector. The third is their cross product.

    Args:
        x_axis: An array of shape (..., 3) representing the primary axis.
        xy_plane: An array of shape (..., 3) used to define the xy-plane.
        eps: A small epsilon value for numerical stability during normalization.

    Returns:
        An array of shape (..., 3, 3) representing the resulting rotation matrices,
        where the orthonormal basis vectors are stacked along the last axis (as columns).

    Example:
        >>> import jax.numpy as jnp
        >>> x_axis = jnp.array([1.0, 0.0, 0.0])
        >>> xy_plane = jnp.array([1.0, 1.0, 0.0])
        >>> graham_schmidt(x_axis, xy_plane)
        Array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)
    """
    e1 = xy_plane
    denom = jnp.sqrt((x_axis**2).sum(axis=-1, keepdims=True) + eps)
    x_axis = x_axis / denom
    dot = (x_axis * e1).sum(axis=-1, keepdims=True)
    e1 = e1 - x_axis * dot
    denom = jnp.sqrt((e1**2).sum(axis=-1, keepdims=True) + eps)
    e1 = e1 / denom
    e2 = jnp.cross(x_axis, e1, axis=-1)
    rots = jnp.stack([x_axis, e1, e2], axis=-1)
    return rots
