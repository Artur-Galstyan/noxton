import math
from collections.abc import Callable

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


class LinearHeadwiseExpand(eqx.Module):
    """Structured headwise projection that expands the feature dimension.

    Splits the input along the feature axis into ``num_heads`` equal heads,
    applies an independent linear projection to each head, then concatenates
    the results.  The weight tensor has shape
    ``(num_heads, out_features_per_head, in_features_per_head)`` and is
    initialised with a scaled normal distribution
    (``std = sqrt(2 / (5 * in_features_per_head))``).

    Only integer expansion factors are supported: ``out_features`` must be
    divisible by ``num_heads``, and ``in_features`` must be divisible by
    ``num_heads``.

    Args:
        in_features: Total size of the input feature dimension.  Must be
            divisible by ``num_heads``.
        num_heads: Number of independent projection heads.
        out_features: Total size of the output feature dimension.  Must be
            divisible by ``num_heads``.
        use_bias: If ``True``, add a single learnable bias of shape
            ``(out_features,)`` initialised to zero.  Defaults to ``True``.
        key: JAX PRNG key for weight initialisation.
        dtype: Parameter dtype.  Defaults to the project default when ``None``.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from noxton.nn import LinearHeadwiseExpand
        >>> key = jax.random.PRNGKey(0)
        >>> layer = LinearHeadwiseExpand(in_features=64, num_heads=4, out_features=256, key=key)
        >>> x = jax.random.normal(key, (32, 64))  # (time, features)
        >>> layer(x).shape
        (32, 256)
    """

    weight: Array
    bias: Array | None

    num_heads: int

    def __init__(
        self,
        in_features: int,
        num_heads: int,
        out_features: int,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.num_heads = num_heads
        out_features_per_head = out_features // num_heads
        weight_shape = (num_heads, out_features_per_head, in_features // num_heads)
        wkey, bkey = jax.random.split(key)
        self.weight = math.sqrt(2 / 5 / weight_shape[-1]) * jax.random.normal(
            key=wkey,
            shape=weight_shape,
            dtype=dtype,
        )
        if use_bias:
            self.bias = jnp.zeros(out_features)
        else:
            self.bias = None

    def __call__(self, x: Array) -> Array:
        """Apply the headwise projection to ``x``.

        Args:
            x: Input array whose last dimension must equal ``in_features``.
                Leading dimensions are treated as batch dimensions.

        Returns:
            Array of the same shape as ``x`` except the last dimension is
            ``out_features``.
        """
        shape = x.shape
        x = x.reshape(*shape[:-1], self.num_heads, -1)
        x = jnp.einsum("...hd,hod->...ho", x, self.weight)
        x = x.reshape(*shape[:-1], -1)
        if self.bias is not None:
            x = x + self.bias
        return x


class GatedFeedForward(eqx.Module):
    proj_up: eqx.nn.Linear
    proj_down: eqx.nn.Linear
    act_fn: Callable
    dropout: eqx.nn.Dropout
    inference: bool
    proj_up_dim: int

    def __init__(
        self,
        embedding_dim: int,
        proj_up_dim: int,
        use_bias: bool = False,
        act_fn: Callable = jax.nn.gelu,
        dropout_p: float = 0.0,
        *,
        key: PRNGKeyArray,
        inference: bool = False,
        dtype: Any | None = None,
    ):
        self.proj_up_dim = proj_up_dim
        self.act_fn = act_fn
        self.inference = inference

        key, upkey, downkey = jax.random.split(key, 3)
        self.proj_up = eqx.nn.Linear(
            in_features=embedding_dim,
            out_features=2 * proj_up_dim,
            use_bias=use_bias,
            key=upkey,
            dtype=dtype,
        )
        self.proj_down = eqx.nn.Linear(
            in_features=proj_up_dim,
            out_features=embedding_dim,
            use_bias=use_bias,
            key=downkey,
            dtype=dtype,
        )
        self.dropout = eqx.nn.Dropout(dropout_p)

    def __call__(
        self,
        x: Float[Array, "seq_len embed_dim"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Array:
        x = eqx.filter_vmap(self.proj_up)(x)
        gate_preact, up_proj = jnp.split(x, [self.proj_up_dim], axis=-1)
        x = self.act_fn(gate_preact) * up_proj
        x = eqx.filter_vmap(self.proj_down)(x)
        x = self.dropout(x, key=key, inference=self.inference)
        return x
