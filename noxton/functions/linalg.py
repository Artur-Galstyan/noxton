import jax.numpy as jnp
from jaxtyping import Array, Float


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


def qmul(
    a: Float[Array, "w x y z"], b: Float[Array, "w x y z"]
) -> Float[Array, "w x y z"]:
    """
    Multiply two quaternions.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return jnp.stack((ow, ox, oy, oz), -1)


def qrot(q: Float[Array, "w x y z"], p: Float[Array, "x y z"]) -> Array:
    """
    Rotates p by quaternion q.

    Args:
        q: Quaternions as tensor of shape (..., 4), real part first.
        p: Points as tensor of shape (..., 3)

    Returns:
        The rotated version of p, of shape (..., 3)
    """
    aw, ax, ay, az = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    bx, by, bz = p[..., 0], p[..., 1], p[..., 2]
    # fmt: off
    ow =         - ax * bx - ay * by - az * bz
    ox = aw * bx           + ay * bz - az * by
    oy = aw * by - ax * bz           + az * bx
    oz = aw * bz + ax * by - ay * bx
    # fmt: on
    q_mul_pts = jnp.stack((ow, ox, oy, oz), -1)
    return qmul(q_mul_pts, qinv(q))[..., 1:]


def qinv(q: Array):
    return q * jnp.array([1, -1, -1, -1])
