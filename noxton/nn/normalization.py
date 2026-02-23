import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Hashable, Sequence
from equinox.nn import State
from jaxtyping import Array, Float, PRNGKeyArray

from noxton.utils import default_floating_dtype

from .abstract import AbstractNorm, AbstractNormStateful


class BatchNorm(AbstractNormStateful):
    """Batch normalisation for use inside ``jax.vmap`` / ``jax.lax.pmean``.

    Normalises each channel across the batch (and spatial dimensions when the
    input has more than one dimension) using statistics gathered via
    ``jax.lax.pmean`` / ``jax.lax.psum`` over the named ``axis_name``.
    Running mean and variance are maintained in an Equinox ``State`` object
    and updated during training.

    At inference time (``inference=True``) the stored running statistics are
    used instead of batch statistics.

    Args:
        size: Number of channels (features) to normalise.
        axis_name: The ``vmap`` / ``pmap`` axis name used for cross-device
            or cross-batch aggregation via ``jax.lax.pmean``.
        eps: Small constant added to the variance for numerical stability.
            Defaults to ``1e-5``.
        momentum: Exponential moving-average factor for updating running
            statistics.  Defaults to ``0.1``.
        affine: If ``True``, learn per-channel scale (``gamma``) and shift
            (``beta``) parameters.  Defaults to ``True``.
        inference: If ``True``, use running statistics instead of batch
            statistics.  Defaults to ``False``.
        dtype: Floating-point dtype for parameters.  Defaults to the project
            default when ``None``.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> bn = BatchNorm(size=16, axis_name="batch")
        >>> state = eqx.nn.State(bn)
        >>> x = jax.random.normal(jax.random.PRNGKey(0), (16,))
        >>> out, state = jax.vmap(
        ...     bn, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
        ... )(x[None], state)
        >>> out.shape
        (1, 16)
    """

    running_mean_var: eqx.nn.StateIndex

    gamma: Float[Array, "size"] | None
    beta: Float[Array, "size"] | None

    inference: bool
    axis_name: Hashable | Sequence[Hashable]

    size: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    affine: bool = eqx.field(static=True)

    def __init__(
        self,
        size: int,
        axis_name: Hashable | Sequence[Hashable],
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        inference: bool = False,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.size = size
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.inference = inference
        self.axis_name = axis_name

        self.gamma = jnp.ones(self.size, dtype=dtype) if self.affine else None
        self.beta = jnp.zeros(self.size, dtype=dtype) if self.affine else None

        self.running_mean_var = eqx.nn.StateIndex(
            (jnp.zeros(size, dtype=dtype), jnp.ones(size, dtype=dtype))
        )

    def __call__(
        self, x: Array, state: State, *_, key: PRNGKeyArray | None = None, **__
    ) -> tuple[Array, State]:
        """Apply batch normalisation to ``x``.

        Must be called inside a ``jax.vmap`` (or ``jax.lax.pmap``) scope that
        maps over the named ``axis_name`` so that ``jax.lax.pmean`` /
        ``jax.lax.psum`` are valid.

        Args:
            x: Input array.  Either ``(size,)`` for feature vectors or
                ``(size, *spatial_dims)`` for convolutional feature maps.
            state: Equinox ``State`` containing the running mean and variance
                buffers.
            key: Unused; present for API compatibility.  Defaults to ``None``.

        Returns:
            A ``(output, state)`` tuple where ``output`` has the same shape as
            ``x`` and ``state`` has updated running statistics (training only).
        """
        running_mean, running_var = state.get(self.running_mean_var)

        input_shape = x.shape
        ndim = len(input_shape)

        if ndim == 1:
            batch_mean = jax.lax.pmean(x, axis_name=self.axis_name)
            batch_size = jax.lax.psum(1, axis_name=self.axis_name)

            if self.inference:
                x_normalized = (x - running_mean) / jnp.sqrt(running_var + self.eps)
            else:
                xmu = x - batch_mean
                sq = xmu**2
                batch_var = jax.lax.pmean(sq, axis_name=self.axis_name)
                std = jnp.sqrt(batch_var + self.eps)
                x_normalized = xmu / std

                correction_factor = batch_size / jnp.maximum(batch_size - 1, 1)
                running_mean = (
                    1 - self.momentum
                ) * running_mean + self.momentum * batch_mean
                running_var = (1 - self.momentum) * running_var + self.momentum * (
                    batch_var * correction_factor
                )

                state = state.set(self.running_mean_var, (running_mean, running_var))
        else:
            spatial_axes = tuple(range(1, ndim))  # All dims except channel dim (0)

            if self.inference:
                x_normalized = (
                    x - running_mean.reshape((-1,) + (1,) * (ndim - 1))
                ) / jnp.sqrt(running_var.reshape((-1,) + (1,) * (ndim - 1)) + self.eps)
            else:
                spatial_mean = jnp.mean(x, axis=spatial_axes)

                batch_mean = jax.lax.pmean(spatial_mean, axis_name=self.axis_name)
                batch_size = jax.lax.psum(1, axis_name=self.axis_name)

                broadcast_shape = (-1,) + (1,) * (ndim - 1)
                batch_mean_broadcasted = batch_mean.reshape(broadcast_shape)

                xmu = x - batch_mean_broadcasted
                sq = xmu**2

                spatial_var = jnp.mean(sq, axis=spatial_axes)
                batch_var = jax.lax.pmean(spatial_var, axis_name=self.axis_name)

                batch_var_broadcasted = batch_var.reshape(broadcast_shape)
                std = jnp.sqrt(batch_var_broadcasted + self.eps)

                x_normalized = xmu / std

                spatial_size = 1
                for dim in spatial_axes:
                    spatial_size *= x.shape[dim]
                total_size = batch_size * spatial_size

                correction_factor = total_size / jnp.maximum(total_size - 1, 1)
                running_mean = (
                    1 - self.momentum
                ) * running_mean + self.momentum * batch_mean
                running_var = (1 - self.momentum) * running_var + self.momentum * (
                    batch_var * correction_factor
                )

                state = state.set(self.running_mean_var, (running_mean, running_var))

        out = x_normalized
        if self.affine and self.gamma is not None and self.beta is not None:
            if ndim > 1:
                broadcast_shape = (-1,) + (1,) * (ndim - 1)
                gamma_broadcasted = self.gamma.reshape(broadcast_shape)
                beta_broadcasted = self.beta.reshape(broadcast_shape)
                out = gamma_broadcasted * x_normalized + beta_broadcasted
            else:
                out = self.gamma * x_normalized + self.beta

        return out, state


class LocalResponseNormalization(eqx.Module):
    """Local Response Normalisation (LRN) across adjacent channels.

    Divides each spatial position by a normalisation factor derived from a
    window of ``n`` neighbouring channels, following the original AlexNet
    formulation (Krizhevsky et al., 2012)::

        b_{c,h,w} = x_{c,h,w} / (k + alpha * sum_{j} x_{j,h,w}^2)^beta

    where the sum runs over at most ``n`` channels centred on ``c``.

    Args:
        k: Bias constant in the denominator.  Defaults to ``2``.
        n: Number of adjacent channels included in the normalisation window.
            Defaults to ``5``.
        alpha: Scale factor for the squared activations.  Defaults to
            ``1e-4``.
        beta: Exponent applied to the denominator.  Defaults to ``0.75``.

    Example:
        >>> import jax.numpy as jnp
        >>> lrn = LocalResponseNormalization()
        >>> x = jnp.ones((8, 14, 14))  # (C, H, W)
        >>> lrn(x).shape
        (8, 14, 14)
    """

    k: int = eqx.field(static=True)
    n: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    def __init__(self, k=2, n=5, alpha=1e-4, beta=0.75) -> None:
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        c, _, _ = x.shape
        p = jnp.pad(x, pad_width=[(self.n // 2, self.n // 2), (0, 0), (0, 0)])

        def _body(i):
            window = jax.lax.dynamic_slice_in_dim(p, i, self.n) ** 2
            d = (jnp.einsum("ijk->jk", window) * self.alpha + self.k) ** self.beta
            b = x[i] / d
            return b

        ys = eqx.filter_vmap(_body)(jnp.arange(c))
        return ys


class LayerNorm(AbstractNorm):
    """Layer normalisation over the last one or more dimensions of an array.

    Normalises the input by subtracting the mean and dividing by the standard
    deviation computed over the trailing ``len(shape)`` axes, then optionally
    applies learnable affine parameters ``weight`` (scale) and ``bias``
    (shift).

    Computation is performed at a higher precision (at least ``float32``) and
    the result is cast back to the original dtype, matching the behaviour of
    ``torch.nn.LayerNorm``.

    Args:
        shape: The shape of the normalised sub-array, i.e. the trailing
            dimensions.  Pass a single ``int`` for the common 1-D case.
        eps: Small constant added to the variance for numerical stability.
            Defaults to ``1e-5``.
        use_weight: If ``True``, learn a per-element scale parameter
            initialised to ``1``.  Defaults to ``True``.
        use_bias: If ``True``, learn a per-element bias parameter initialised
            to ``0``.  Defaults to ``True``.
        dtype: Floating-point dtype for the affine parameters.  Defaults to
            the project default when ``None``.

    Raises:
        ValueError: If the last ``len(shape)`` dimensions of the input do not
            match ``shape``.

    Example:
        >>> import jax.numpy as jnp
        >>> ln = LayerNorm(shape=64)
        >>> x = jnp.ones((10, 64))
        >>> ln(x).shape
        (10, 64)

        >>> # 2-D normalisation (normalise over H and W together)
        >>> ln2d = LayerNorm(shape=(28, 28))
        >>> x2d = jnp.ones((16, 28, 28))
        >>> ln2d(x2d).shape
        (16, 28, 28)
    """

    shape: tuple[int, ...] = eqx.field(static=True)
    axes: tuple[int, ...] = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    use_weight: bool = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    weight: Array | None
    bias: Array | None

    def __init__(
        self,
        shape: int | Sequence[int],
        eps: float = 1e-5,
        use_weight: bool = True,
        use_bias: bool = True,
        dtype=None,
    ):
        if isinstance(shape, int):
            shape_tuple = (shape,)
        else:
            shape_tuple = tuple(shape)
        self.shape = shape_tuple
        self.axes = tuple(range(-len(self.shape), 0))
        self.eps = eps
        self.use_weight = use_weight
        self.use_bias = use_bias

        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None

        if self.use_weight:
            self.weight = jnp.ones(self.shape, dtype=dtype)
        else:
            self.weight = None
        if self.use_bias:
            self.bias = jnp.zeros(self.shape, dtype=dtype)
        else:
            self.bias = None

    def __call__(self, x: Array, *_, key: PRNGKeyArray | None = None, **__) -> Array:
        if x.shape[-len(self.shape) :] != self.shape:
            raise ValueError(f"Input shape {x.shape} must end with shape {self.shape}")

        orig_dtype = x.dtype
        with jax.numpy_dtype_promotion("standard"):
            calc_dtype_elems = [x.dtype, jnp.float32]
            if self.weight is not None:
                calc_dtype_elems.append(self.weight.dtype)
            if self.bias is not None:
                calc_dtype_elems.append(self.bias.dtype)
            calc_dtype = jnp.result_type(*calc_dtype_elems)

        x = x.astype(calc_dtype)

        mean = jnp.mean(x, axis=self.axes, keepdims=True)
        mean_keepdims = jnp.mean(x, axis=self.axes, keepdims=True)
        variance = jnp.mean(
            jnp.square(x - mean_keepdims), axis=self.axes, keepdims=True
        )
        variance = jnp.maximum(0.0, variance)
        inv = jax.lax.rsqrt(variance + self.eps)
        out = (x - mean) * inv

        if self.use_weight:
            assert self.weight is not None
            out = out * self.weight.astype(calc_dtype)
        if self.use_bias:
            assert self.bias is not None
            out = out + self.bias.astype(calc_dtype)

        out = out.astype(orig_dtype)
        return out
