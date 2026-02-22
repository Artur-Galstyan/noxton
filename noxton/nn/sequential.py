import math

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Literal
from jaxtyping import Array, Float, PRNGKeyArray

from noxton.utils import default_floating_dtype


def default_init(
    key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any, lim: float
) -> jax.Array:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        real_dtype = jnp.finfo(dtype).dtype
        rkey, ikey = jax.random.split(key, 2)
        real = jax.random.uniform(rkey, shape, real_dtype, minval=-lim, maxval=lim)
        imag = jax.random.uniform(ikey, shape, real_dtype, minval=-lim, maxval=lim)
        return real.astype(dtype) + 1j * imag.astype(dtype)
    else:
        return jax.random.uniform(key, shape, dtype, minval=-lim, maxval=lim)


class BatchedLinear(eqx.Module):
    """Linear layer that natively handles arbitrarily-batched inputs.

    Behaves like ``eqx.nn.Linear`` but accepts inputs with any number of
    leading batch dimensions and reshapes internally, avoiding the need for
    an explicit outer ``vmap``.  Supports both real and complex dtypes.

    Weights and biases are initialised with a uniform distribution in
    ``[-1/sqrt(in_features), 1/sqrt(in_features)]`` (Kaiming uniform with
    ``mode="fan_in"``).

    Args:
        in_features: Size of the last input dimension, or ``"scalar"`` to
            treat scalars as 1-D vectors of length ``1``.
        out_features: Size of the last output dimension, or ``"scalar"``.
        use_bias: If ``True``, add a learnable bias vector.  Defaults to
            ``True``.
        dtype: Floating-point (or complex) dtype for all parameters.
            Defaults to the project default when ``None``.
        key: JAX PRNG key for parameter initialisation.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> linear = BatchedLinear(in_features=8, out_features=4, key=key)
        >>> x = jax.random.normal(key, (3, 5, 8))  # (batch1, batch2, in_features)
        >>> linear(x).shape
        (3, 5, 4)

        >>> # Also works with a plain 1-D vector
        >>> linear(jnp.ones(8)).shape
        (4,)
    """

    weight: Array
    bias: Array | None
    in_features: int | Literal["scalar"] = eqx.field(static=True)
    out_features: int | Literal["scalar"] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int | Literal["scalar"],
        out_features: int | Literal["scalar"],
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        weight_key, bias_key = jax.random.split(key)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        if in_features_ == 0:
            lim = 1.0
        else:
            lim = 1 / math.sqrt(in_features_)
        wshape = (out_features_, in_features_)
        self.weight = default_init(weight_key, wshape, dtype, lim)
        bshape = (out_features_,)
        self.bias = default_init(bias_key, bshape, dtype, lim) if use_bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(
        self, x: Float[Array, "*batch in_features"], key=None
    ) -> Float[Array, "*batch out_features"]:
        """Apply the linear transformation to ``x``.

        Args:
            x: Input array whose last dimension must equal ``in_features``.
                Leading dimensions are treated as batch dimensions.
            key: Unused; present for API compatibility.  Defaults to ``None``.

        Returns:
            Array of shape ``(*batch_dims, out_features)``.

        Raises:
            AssertionError: If ``x.shape[-1] != in_features``.
        """
        input_shape = x.shape

        assert input_shape[-1] == self.weight.shape[1], (
            f"Expected last dimension to be {self.weight.shape[1]},"
            f" got {input_shape[-1]}"
        )

        if len(input_shape) > 1:
            batch_dims = input_shape[:-1]
            flattened_x = x.reshape(-1, self.in_features)
            result = flattened_x @ self.weight.T
            if self.use_bias and self.bias is not None:
                result = result + self.bias
            return result.reshape(*batch_dims, self.out_features)
        else:
            result = self.weight @ x
            if self.use_bias and self.bias is not None:
                result = result + self.bias
            return result
