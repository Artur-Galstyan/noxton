import jax.numpy as jnp
from jaxtyping import Array, Float


def rbf(values, v_min, v_max, n_bins=16):
    """
    Computes radial basis function (RBF) features for a given array of values.

    Args:
        values: A `jnp.ndarray` of continuous values of shape `(...)`.
        v_min: Minimum value for the RBF centers.
        v_max: Maximum value for the RBF centers.
        n_bins: Number of RBF centers (bins) to use. Default is 16.

    Returns:
        A `jnp.ndarray` of shape `(..., n_bins)` containing the evaluated RBF features.

    Example:
        >>> import jax.numpy as jnp
        >>> values = jnp.array([0.0, 1.0, 2.0])
        >>> rbf(values, v_min=0.0, v_max=2.0, n_bins=2)
        Array([[1.        , 0.01831564],
               [0.36787945, 0.36787945],
               [0.01831564, 1.        ]], dtype=float32)
    """
    rbf_centers = jnp.linspace(v_min, v_max, n_bins, dtype=values.dtype)
    rbf_centers = rbf_centers.reshape([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (jnp.expand_dims(values, axis=-1) - rbf_centers) / rbf_std
    return jnp.exp(-(z**2))


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
