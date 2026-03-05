from collections.abc import Iterable
from itertools import repeat

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable, Sequence, cast
from equinox.nn import StatefulLayer
from jaxtyping import Array, Float, PRNGKeyArray

from noxton.nn import AbstractNorm, AbstractNormStateful
from noxton.utils import default_floating_dtype


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


class CausalConv1d(eqx.Module):
    """Causal 1-D convolution over time-series tensors.

    Applies a Conv1d with left-only padding so each output timestep depends
    only on current and past inputs (no future leakage).  Supports both a
    parallel forward pass over a full sequence and an efficient single-step
    ``step`` method for autoregressive inference.

    Setting ``kernel_size=0`` creates an identity layer that returns the input
    unchanged without allocating any parameters.

    Args:
        feature_dim: Number of features (channels) in the input tensor.
        channel_mixing: If ``True``, use a single group (``groups=1``) so
            features are mixed across channels.  If ``False``, use depthwise
            convolution (``groups=feature_dim``) so each feature is convolved
            independently.
        kernel_size: Convolution kernel size.  Set to ``0`` to skip the
            convolution entirely (identity pass-through).
        use_bias: Whether to add a learnable bias term.  Defaults to ``True``.
        key: JAX PRNG key for parameter initialisation.
        dtype: Parameter dtype.  Defaults to the project default when ``None``.
        **conv1d_kwargs: Additional keyword arguments forwarded to
            ``eqx.nn.Conv1d``.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from noxton.nn import CausalConv1d
        >>> key = jax.random.PRNGKey(0)
        >>> layer = CausalConv1d(feature_dim=16, channel_mixing=False, kernel_size=4, key=key)
        >>> x = jax.random.normal(key, (32, 16))  # (time, features)
        >>> layer(x).shape
        (32, 16)
    """

    groups: int
    kernel_size: int
    conv: eqx.nn.Conv1d | None
    use_bias: bool
    pad: int

    def __init__(
        self,
        feature_dim: int,
        channel_mixing: bool,
        kernel_size: int,
        *,
        use_bias: bool = True,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        **conv1d_kwargs,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        self.groups = feature_dim
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        if channel_mixing:
            self.groups = 1
        if kernel_size == 0:
            self.conv = None
        else:
            self.pad = (
                kernel_size - 1
            )  # padding of this size assures temporal causality.
            self.conv = eqx.nn.Conv1d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=kernel_size,
                padding=self.pad,
                groups=self.groups,
                use_bias=use_bias,
                key=key,
                **conv1d_kwargs,
            )

    def __call__(
        self,
        x: Array,
        conv_state: Array | None = None,
        return_last_state: bool = False,
    ) -> Array | tuple[Array, Array]:
        """Apply causal convolution to a full input sequence.

        Processes the entire sequence in parallel.  Optionally prepends cached
        context from a previous chunk (``conv_state``) to support chunked /
        streaming inference without recomputing past activations.

        Args:
            x: Input tensor of shape ``(T, F)`` — time first, features last.
            conv_state: Optional cached context of shape ``(S, F)`` from a
                previous chunk, prepended to ``x`` before convolution.  When
                provided, the output is trimmed back to length ``T``.
            return_last_state: If ``True``, also return the last
                ``kernel_size - 1`` timesteps of the (possibly prepended)
                input as the new ``conv_state`` for the next chunk.

        Returns:
            - If ``return_last_state`` is ``False``: output array of shape
              ``(T, F)``.
            - If ``return_last_state`` is ``True``: a ``(output, new_state)``
              tuple where ``output`` is ``(T, F)`` and ``new_state`` is
              ``(kernel_size - 1, F)``.
        """
        if conv_state is not None:
            x = jnp.concat([conv_state, x], axis=0)

        if self.kernel_size == 0:
            return x
        y = x.T
        assert self.conv is not None
        y = self.conv(y)  # (B,F,T+pad) tensor
        if conv_state is not None:
            y = y[:, conv_state.shape[0] :]

        if return_last_state:
            return y[:, : -self.pad].T, x[:, -self.pad :]
        else:
            return y[:, : -self.pad].T

    def step(
        self,
        x: Array,
        conv_state: tuple[Array] | None = None,
    ) -> tuple[Array, tuple[Array] | None]:
        """Apply causal convolution to a single timestep (autoregressive mode).

        Maintains a sliding-window buffer of the last ``kernel_size``
        timesteps.  Efficient for token-by-token generation where re-running
        the full sequence each step would be prohibitive.

        Args:
            x: Input tensor of shape ``(1, F)`` — exactly one timestep.
            conv_state: Tuple containing one array of shape
                ``(kernel_size, F)`` — the sliding-window buffer from the
                previous step.  When ``None``, the buffer is zero-initialised
                automatically.

        Returns:
            A ``(output, new_conv_state)`` tuple where ``output`` is shape
            ``(1, F)`` and ``new_conv_state`` is a tuple containing the
            updated buffer of shape ``(kernel_size, F)``.  Returns the
            original ``conv_state`` unchanged when ``kernel_size == 0``.
        """
        if self.kernel_size == 0:
            return x, conv_state

        def _conv1d_step(
            x: Array,
            conv_state: Array,
            conv1d_weight: Array,
            conv1d_bias: Array | None = None,
        ):
            """
            S: sequence length
            D: feature dimension
            KS: kernel size
            Args:
                x (Array): (S, D)
                conv_state (Array): (KS, D)
                conv1d_weight (Array): (KS, D)
            """
            seq_len, feat_dims = x.shape
            assert feat_dims == conv_state.shape[1], (
                f"x has feature dimension {feat_dims} but conv_state has feature dimension {conv_state.shape[1]}"
            )
            assert seq_len == 1, f"x has sequence length {seq_len} but it should be 1"
            conv_state = jnp.roll(conv_state, shift=-1, axis=0)
            conv_state = conv_state.at[-1:, :].set(x.squeeze(0))
            y = jnp.sum(conv_state * conv1d_weight, axis=0, keepdims=True)
            if conv1d_bias is not None:
                y += conv1d_bias.squeeze()

            return y, conv_state

        S, D = x.shape

        if conv_state is None:
            assert self.conv is not None
            conv_state = (
                jnp.zeros(
                    shape=(self.kernel_size, D),
                    dtype=self.conv.weight.dtype,
                ),
            )

        assert self.conv is not None
        y, conv_state = _conv1d_step(
            x,
            conv_state[0],
            self.conv.weight[:, 0, :].T,
            conv1d_bias=self.conv.bias if self.use_bias else None,
        )
        return y, (conv_state,)
