from collections.abc import Iterable
from itertools import repeat

import equinox as eqx
import jax
from beartype.typing import Any, Callable, Sequence, cast
from equinox.nn import StatefulLayer
from jaxtyping import Array, Float, PRNGKeyArray

from noxton.nn import AbstractNorm, AbstractNormStateful


def make_ntuple(x: Any, n: int) -> tuple[Any, ...]:
    if isinstance(x, Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


class ConvNormActivation(StatefulLayer):
    """Composable convolution → normalisation → activation block.

    Stacks a single ``eqx.nn.Conv`` layer, an optional normalisation layer,
    and an optional activation function into one Equinox ``StatefulLayer``.
    When ``padding`` is ``None``, same-padding is computed automatically from
    ``kernel_size`` and ``dilation`` so that spatial dimensions are preserved
    (assuming ``stride=1``).

    When ``use_bias`` is ``None`` it defaults to ``True`` only when no
    ``norm_layer`` is provided (normalisation makes biases redundant).

    Args:
        num_spatial_dims: Number of spatial dimensions, e.g. ``2`` for images.
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Convolution kernel size.  Can be a single int (applied to
            all spatial dimensions) or a sequence of per-dimension ints.
            Defaults to ``3``.
        stride: Convolution stride.  Defaults to ``1``.
        padding: Explicit padding spec passed to ``eqx.nn.Conv``.  When
            ``None``, same-padding is derived automatically.
        groups: Number of blocked connections from input to output channels.
            Defaults to ``1`` (standard convolution).
        activation_layer: Callable used as the activation function, or
            ``None`` to omit activation.  Defaults to ``jax.nn.relu``.
        dilation: Kernel dilation.  Defaults to ``1``.
        use_bias: Whether to add a learnable bias to the convolution.  When
            ``None``, defaults to ``True`` iff ``norm_layer`` is ``None``.
        norm_layer: Zero-argument factory that returns an ``AbstractNorm`` or
            ``AbstractNormStateful`` instance, or ``None`` to skip
            normalisation.
        key: JAX PRNG key for convolution parameter initialisation.
        dtype: Floating-point dtype for convolution parameters.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> from functools import partial
        >>> from noxton.nn import ConvNormActivation, BatchNorm
        >>> key = jax.random.PRNGKey(0)
        >>> norm_factory = partial(BatchNorm, size=32, axis_name="batch")
        >>> block = ConvNormActivation(
        ...     2, 16, 32, kernel_size=3, norm_layer=norm_factory,
        ...     key=key, dtype=jax.numpy.float32,
        ... )
        >>> x = jax.random.normal(key, (16, 28, 28))
        >>> state = eqx.nn.State(block)
        >>> out, state = jax.vmap(
        ...     block, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
        ... )(x[None], state)
    """

    conv: eqx.nn.Conv
    norm: AbstractNorm | AbstractNormStateful | None
    activation: eqx.nn.Lambda | None

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: str | int | Sequence[int] | Sequence[tuple[int, int]] | None = None,
        groups: int = 1,
        activation_layer: Callable[..., Array] | None = jax.nn.relu,
        dilation: int | Sequence[int] = 1,
        use_bias: bool | None = None,
        *,
        norm_layer: Callable[..., AbstractNorm | AbstractNormStateful] | None,
        key: PRNGKeyArray,
        dtype: Any,
    ):
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = (
                    len(kernel_size)
                    if isinstance(kernel_size, Sequence)
                    else len(dilation)  # ty:ignore[invalid-argument-type]
                )
                kernel_size = make_ntuple(kernel_size, _conv_dim)
                dilation = make_ntuple(dilation, _conv_dim)
                padding = tuple(
                    (kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim)
                )
        if use_bias is None:
            use_bias = norm_layer is None

        key, subkey = jax.random.split(key)

        self.conv = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            dtype=dtype,
            key=subkey,
        )

        self.norm = None
        if norm_layer is not None:
            self.norm = norm_layer()

        if activation_layer is not None:
            self.activation = eqx.nn.Lambda(activation_layer)
        else:
            self.activation = None

    def __call__(
        self,
        x: Float[Array, "c *num_spatial_dims"],
        state: eqx.nn.State,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "c_out *num_spatial_dims_out"], eqx.nn.State]:
        """Apply convolution, optional normalisation, and optional activation.

        Args:
            x: Input array of shape ``(in_channels, *spatial_dims)``.
            state: Equinox ``State`` object; updated in-place when a stateful
                norm layer (e.g. ``BatchNorm``) is present.
            key: Unused; present for API compatibility with other stateful
                layers.  Defaults to ``None``.

        Returns:
            A ``(output, state)`` tuple where ``output`` has shape
            ``(out_channels, *spatial_dims_out)`` and ``state`` is the
            (possibly updated) state object.
        """
        x = self.conv(x)

        if self.norm:
            if isinstance(self.norm, StatefulLayer) and self.norm.is_stateful():
                assert state is not None
                x, state = self.norm(x, state)
            else:
                norm_fn = cast(AbstractNorm, self.norm)
                x = norm_fn(x)

        if self.activation:
            x = self.activation(x)

        return x, state
