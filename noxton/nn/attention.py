import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import Array, PRNGKeyArray

from noxton.functions import (
    canonical_mask,
    multi_head_attention_forward,
)
from noxton.utils import default_floating_dtype


class MultiheadAttention(eqx.Module):
    """Multi-head attention — a 1-to-1 JAX/Equinox port of ``torch.nn.MultiheadAttention``.

    Splits queries, keys, and values into ``num_heads`` independent attention
    heads, computes scaled dot-product attention for each head in parallel, and
    projects the concatenated outputs back to ``embed_dim``.

    .. note::
        This implementation is intentionally API-compatible with
        ``torch.nn.MultiheadAttention``.  Unless you specifically need that
        compatibility, prefer ``eqx.nn.MultiheadAttention``, which is more
        idiomatic for JAX.

    Args:
        embed_dim: Total dimensionality of the model (query) embeddings.
        num_heads: Number of attention heads.  ``embed_dim`` must be
            divisible by ``num_heads``.
        dropout: Dropout probability applied to the attention weights during
            training.  Defaults to ``0.0`` (no dropout).
        bias: If ``True``, add learnable bias terms to the input and output
            projections.  Defaults to ``True``.
        add_bias_kv: If ``True``, append learnable bias vectors to the key
            and value sequences.  Defaults to ``False``.
        add_zero_attn: If ``True``, append a batch of zeros to the key and
            value sequences.  Defaults to ``False``.
        kdim: Dimensionality of the key inputs.  Defaults to ``embed_dim``.
        vdim: Dimensionality of the value inputs.  Defaults to ``embed_dim``.
        inference: If ``True``, disable dropout (eval mode).  Defaults to
            ``False``.
        key: JAX PRNG key for parameter initialisation.
        dtype: Floating-point dtype for all parameters.  Defaults to the
            project default when ``None``.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> mha = MultiheadAttention(embed_dim=64, num_heads=4, key=key)
        >>> q = jax.random.normal(key, (10, 64))  # (seq_len, embed_dim)
        >>> out, weights = mha(q, q, q)
        >>> out.shape
        (10, 64)
        >>> weights.shape   # averaged over heads by default
        (10, 10)
    """

    q_proj_weight: Array | None
    k_proj_weight: Array | None
    v_proj_weight: Array | None

    in_proj_weight: Array | None

    in_proj_bias: Array | None

    out_proj: eqx.nn.Linear

    bias_k: Array | None
    bias_v: Array | None

    embed_dim: int = eqx.field(static=True)
    kdim: int = eqx.field(static=True)
    vdim: int = eqx.field(static=True)
    _qkv_same_embed_dim: bool = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    add_zero_attn: bool = eqx.field(static=True)

    inference: bool

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        inference: bool = False,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ) -> None:
        self.inference = inference
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        uniform_initializer = jax.nn.initializers.uniform(dtype=dtype)

        if not self._qkv_same_embed_dim:
            key, *subkeys = jax.random.split(key, 4)
            self.q_proj_weight = uniform_initializer(
                key=subkeys[0], shape=(embed_dim, embed_dim)
            )
            self.k_proj_weight = uniform_initializer(
                key=subkeys[1], shape=(embed_dim, self.kdim)
            )
            self.v_proj_weight = uniform_initializer(
                key=subkeys[2], shape=(embed_dim, self.vdim)
            )
            self.in_proj_weight = None
        else:
            key, subkey = jax.random.split(key)
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None
            self.in_proj_weight = uniform_initializer(
                key=subkey, shape=(3 * embed_dim, embed_dim)
            )

        if bias:
            self.in_proj_bias = jnp.empty((3 * embed_dim), dtype=dtype)
        else:
            self.in_proj_bias = None
        key, subkey = jax.random.split(key)
        out_proj = eqx.nn.Linear(
            embed_dim, embed_dim, use_bias=bias, key=subkey, dtype=dtype
        )
        if bias:
            assert out_proj.bias is not None
            new_bias = jnp.zeros_like(out_proj.bias, dtype=dtype)
            where = lambda layer: layer.bias
            self.out_proj = eqx.tree_at(where, out_proj, new_bias)
        else:
            self.out_proj = out_proj

        if add_bias_kv:
            normal_initializer = jax.nn.initializers.normal(dtype=dtype)
            key, *subkeys = jax.random.split(key, 3)
            self.bias_k = normal_initializer(key=subkeys[0], shape=(1, embed_dim))
            self.bias_v = normal_initializer(key=subkeys[0], shape=(1, embed_dim))
        else:
            self.bias_k = None
            self.bias_v = None

        self.add_zero_attn = add_zero_attn

    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        key_padding_mask: Array | None = None,
        need_weights: bool = True,
        attn_mask: Array | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        dropout_key: PRNGKeyArray | None = None,
    ) -> tuple[Array, Array | None]:
        """Compute multi-head attention over query, key, and value arrays.

        Args:
            query: Query array of shape ``(tgt_len, embed_dim)``.
            key: Key array of shape ``(src_len, kdim)``.
            value: Value array of shape ``(src_len, vdim)``.
            key_padding_mask: Boolean or float mask of shape ``(src_len,)``.
                Positions set to ``True`` (or ``-inf``) are ignored in
                attention.  Defaults to ``None``.
            need_weights: If ``True``, also return the attention weight
                matrix.  Defaults to ``True``.
            attn_mask: Additive attention bias of shape
                ``(tgt_len, src_len)`` or broadcastable.  Defaults to
                ``None``.
            average_attn_weights: If ``True``, average the attention weights
                across heads before returning.  Defaults to ``True``.
            is_causal: If ``True``, apply a causal (lower-triangular) mask
                so each query position can only attend to earlier key
                positions.  Defaults to ``False``.
            dropout_key: JAX PRNG key for attention-weight dropout.  Required
                when ``dropout > 0`` and ``inference=False``.

        Returns:
            A ``(attn_output, attn_weights)`` tuple where ``attn_output`` has
            shape ``(tgt_len, embed_dim)`` and ``attn_weights`` is either
            ``None`` (when ``need_weights=False``) or an array of shape
            ``(tgt_len, src_len)`` (averaged) or
            ``(num_heads, tgt_len, src_len)`` (per-head).
        """
        key_padding_mask = canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                inference=self.inference,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                dropout_key=dropout_key,
            )
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                inference=self.inference,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                dropout_key=dropout_key,
            )

        return attn_output, attn_output_weights


class SqueezeExcitation(eqx.Module):
    """Squeeze-and-Excitation (SE) channel-attention block.

    Globally pools the spatial dimensions of a feature map, passes the result
    through two 1×1 convolutions to produce a per-channel scale vector, and
    multiplies it back into the original input.  This recalibrates channel
    responses adaptively, as introduced in *Squeeze-and-Excitation Networks*
    (Hu et al., 2018).

    The forward pass computes::

        scale = scale_activation(fc2(activation(fc1(avgpool(x)))))
        output = scale * x

    Args:
        input_channels: Number of channels in the input feature map.
        squeeze_channels: Bottleneck width for the intermediate representation
            (typically ``input_channels // reduction_ratio``).
        key: JAX PRNG key for parameter initialisation.
        dtype: Floating-point dtype for all parameters.  Defaults to the
            project default when ``None``.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> se = SqueezeExcitation(input_channels=64, squeeze_channels=16, key=key)
        >>> x = jax.random.normal(key, (64, 28, 28))  # (C, H, W)
        >>> se(x).shape
        (64, 28, 28)
    """

    avgpool: eqx.nn.AdaptiveAvgPool2d
    fc1: eqx.nn.Conv2d
    fc2: eqx.nn.Conv2d

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ) -> None:
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.avgpool = eqx.nn.AdaptiveAvgPool2d(1)
        key, subkey = jax.random.split(key)
        self.fc1 = eqx.nn.Conv2d(
            input_channels, squeeze_channels, 1, key=key, dtype=dtype
        )
        self.fc2 = eqx.nn.Conv2d(
            squeeze_channels, input_channels, 1, key=subkey, dtype=dtype
        )

    def __call__(
        self,
        x: Array,
        activation: Callable[..., Array] = jax.nn.relu,
        scale_activation: Callable[..., Array] = jax.nn.sigmoid,
    ) -> Array:
        """Apply squeeze-and-excitation recalibration to ``x``.

        Args:
            x: Input feature map of shape ``(C, H, W)``.
            activation: Activation applied after the first 1×1 convolution.
                Defaults to ``jax.nn.relu``.
            scale_activation: Activation applied after the second 1×1
                convolution to produce the channel-wise scale in ``[0, 1]``.
                Defaults to ``jax.nn.sigmoid``.

        Returns:
            Recalibrated array of the same shape ``(C, H, W)`` as ``x``.
        """
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = activation(scale)
        scale = self.fc2(scale)
        scale = scale_activation(scale)
        return scale * x
