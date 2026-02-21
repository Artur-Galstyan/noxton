import functools

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from noxton.functions.normalization import normalize
from noxton.functions.regularization import dropout as dropout_fn
from noxton.utils.utils import default_floating_dtype


def multi_head_attention_forward(
    query: Float[Array, "tgt_len d_model"],
    key: Float[Array, "src_len d_model"],
    value: Float[Array, "src_len d_model"],
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Float[Array, "3*d_model d_model"] | None = None,
    in_proj_bias: Float[Array, "3*d_model"] | None = None,
    bias_k: Float[Array, "1 d_model"] | None = None,
    bias_v: Float[Array, "1 d_model"] | None = None,
    add_zero_attn: bool = False,
    dropout_p: float = 0.0,
    out_proj_weight: Float[Array, "d_model d_model"] | None = None,
    out_proj_bias: Float[Array, "d_model"] | None = None,
    inference: bool = False,
    key_padding_mask: Float[Array, "src_len"] | Bool[Array, "src_len"] | None = None,
    attn_mask: Float[Array, "tgt_len src_len"] | None = None,
    need_weights: bool = True,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Float[Array, "d_model d_model"] | None = None,
    k_proj_weight: Float[Array, "d_model d_model"] | None = None,
    v_proj_weight: Float[Array, "d_model d_model"] | None = None,
    static_k: Float[Array, "src_len d_model"] | None = None,
    static_v: Float[Array, "src_len d_model"] | None = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    dropout_key: PRNGKeyArray | None = None,
) -> tuple[
    Float[Array, "tgt_len d_model"],
    Float[Array, "num_heads tgt_len src_len"]
    | Float[Array, "tgt_len src_len"]
    | Float[Array, "tgt_len src_len+1"]
    | None,
]:
    """Compute scaled dot-product multi-head attention.

    This is the functional core of multi-head attention as described in
    *Attention Is All You Need* (Vaswani et al., 2017). It accepts
    pre-allocated weight matrices and returns the attended output along with
    optional attention weights.

    This function is a 1:1 mapping from the PyTorch implementation.

    The function supports several advanced options:

    - **Separate projection weights** (``use_separate_proj_weight=True``):
      pass individual ``q_proj_weight``, ``k_proj_weight``, and
      ``v_proj_weight`` instead of a single fused ``in_proj_weight``.
    - **Static key/value**: bypass the key/value projections by providing
      pre-computed ``static_k`` / ``static_v`` tensors.
    - **Extra key/value bias tokens** (``bias_k``, ``bias_v``): append
      learnable bias tokens to the key and value sequences.
    - **Zero-attention slot** (``add_zero_attn``): append a zero vector to
      keys and values, giving the model an option to attend to nothing.
    - **Causal masking** (``is_causal``): apply an upper-triangular ``-inf``
      mask so each query can only attend to earlier or equal positions.
    - **Key-padding mask** / **additive attention mask**: arbitrary masking
      via ``key_padding_mask`` and ``attn_mask``.

    Args:
        query: Query tensor of shape ``(tgt_len, d_model)``.
        key: Key tensor of shape ``(src_len, d_model)``.
        value: Value tensor of shape ``(src_len, d_model)``.
        embed_dim_to_check: Expected embedding dimension; asserted to equal
            ``d_model``.
        num_heads: Number of attention heads. Must divide ``d_model`` evenly.
        in_proj_weight: Fused input projection weight of shape
            ``(3*d_model, d_model)``. Required when
            ``use_separate_proj_weight=False``.
        in_proj_bias: Optional bias for the fused projection of shape
            ``(3*d_model,)``.
        bias_k: Optional learnable key bias of shape ``(1, d_model)`` appended
            to the key sequence.
        bias_v: Optional learnable value bias of shape ``(1, d_model)``
            appended to the value sequence. Must be provided together with
            ``bias_k``.
        add_zero_attn: If ``True``, append a zero token to keys and values.
            Defaults to ``False``.
        dropout_p: Dropout probability applied to attention weights during
            training. Defaults to ``0.0``.
        out_proj_weight: Output projection weight of shape
            ``(d_model, d_model)``. Required.
        out_proj_bias: Optional output projection bias of shape ``(d_model,)``.
        inference: If ``True``, disable dropout (eval mode).
            Defaults to ``False``.
        key_padding_mask: Optional mask of shape ``(src_len,)`` marking padded
            key positions. ``True`` / non-zero values are masked out (set to
            ``-inf`` in logits).
        attn_mask: Optional additive attention mask of shape
            ``(tgt_len, src_len)``. Values are added to attention logits
            before softmax.
        need_weights: If ``True``, also return the attention weight matrix.
            Defaults to ``True``.
        use_separate_proj_weight: If ``True``, use ``q_proj_weight``,
            ``k_proj_weight``, ``v_proj_weight`` for projections instead of
            ``in_proj_weight``. Defaults to ``False``.
        q_proj_weight: Query projection weight ``(d_model, d_model)``.
            Required when ``use_separate_proj_weight=True``.
        k_proj_weight: Key projection weight ``(d_model, d_model)``.
            Required when ``use_separate_proj_weight=True``.
        v_proj_weight: Value projection weight ``(d_model, d_model)``.
            Required when ``use_separate_proj_weight=True``.
        static_k: Pre-computed key of shape ``(src_len, d_model)``. When
            provided, bypasses the key projection.
        static_v: Pre-computed value of shape ``(src_len, d_model)``. When
            provided, bypasses the value projection.
        average_attn_weights: If ``True``, average attention weights over
            heads before returning. Defaults to ``True``.
        is_causal: If ``True``, apply a causal mask to prevent attending to
            future positions. Defaults to ``False``.
        dropout_key: JAX PRNG key for attention dropout. Required when
            ``dropout_p > 0.0`` and ``inference=False``.

    Returns:
        A tuple ``(attn_output, attn_weights)`` where:

        - ``attn_output``: Attended output of shape ``(tgt_len, d_model)``.
        - ``attn_weights``: Attention weights or ``None`` when
          ``need_weights=False``.  Shape depends on ``average_attn_weights``:
          ``(tgt_len, src_len)`` when averaged, or
          ``(num_heads, tgt_len, src_len)`` otherwise.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> d_model, num_heads, tgt_len, src_len = 8, 2, 4, 6
        >>> q = jax.random.normal(key, (tgt_len, d_model))
        >>> k = jax.random.normal(key, (src_len, d_model))
        >>> v = jax.random.normal(key, (src_len, d_model))
        >>> W = jax.random.normal(key, (3 * d_model, d_model))
        >>> W_out = jax.random.normal(key, (d_model, d_model))
        >>> out, weights = multi_head_attention_forward(
        ...     q, k, v,
        ...     embed_dim_to_check=d_model,
        ...     num_heads=num_heads,
        ...     in_proj_weight=W,
        ...     out_proj_weight=W_out,
        ...     inference=True,
        ... )
        >>> out.shape
        (4, 8)
        >>> weights.shape  # averaged over heads by default
        (4, 6)
    """
    tgt_len, d_model = query.shape
    src_len, k_dim = key.shape
    value_len, v_dim = value.shape

    assert d_model == k_dim == v_dim == embed_dim_to_check, (
        "Embedding dimensions must match"
    )

    assert src_len == value_len, "Key and value must have the same sequence length"

    head_dim = d_model // num_heads
    assert head_dim * num_heads == d_model, "embed_dim must be divisible by num_heads"

    if dropout_p > 0.0:
        assert dropout_key is not None, (
            "dropout_key must be provided if dropout_p > 0.0"
        )

    if use_separate_proj_weight:
        # When using separate projection weights for q, k, v
        assert q_proj_weight is not None, (
            "q_proj_weight should not be None when use_separate_proj_weight=True"
        )
        assert k_proj_weight is not None, (
            "k_proj_weight should not be None when use_separate_proj_weight=True"
        )
        assert v_proj_weight is not None, (
            "v_proj_weight should not be None when use_separate_proj_weight=True"
        )

        q = query @ q_proj_weight.T

        if static_k is None:
            k = key @ k_proj_weight.T
        else:
            k = static_k
            src_len, _ = k.shape

        if static_v is None:
            v = value @ v_proj_weight.T
        else:
            v = static_v
            value_len, _ = v.shape

        if in_proj_bias is not None:
            q_bias, k_bias, v_bias = jnp.split(in_proj_bias, 3)
            q = q + q_bias
            k = k + k_bias
            v = v + v_bias

    else:
        assert in_proj_weight is not None, (
            "in_proj_weight should not be None when use_separate_proj_weight=False"
        )

        q_proj_weight_part, k_proj_weight_part, v_proj_weight_part = jnp.split(
            in_proj_weight, 3
        )

        q = query @ q_proj_weight_part.T

        if static_k is None:
            k = key @ k_proj_weight_part.T
        else:
            k = static_k
            src_len, _ = static_k.shape

        if static_v is None:
            v = value @ v_proj_weight_part.T
        else:
            v = static_v
            value_len, _ = static_v.shape

        if in_proj_bias is not None:
            q_bias, k_bias, v_bias = jnp.split(in_proj_bias, 3)
            q = q + q_bias
            k = k + k_bias
            v = v + v_bias

    assert src_len == value_len

    q = q.reshape(tgt_len, num_heads, head_dim)
    k = k.reshape(src_len, num_heads, head_dim)
    v = v.reshape(src_len, num_heads, head_dim)

    if add_zero_attn:
        zero_attn_shape = (1, num_heads, head_dim)
        k_zeros = jnp.zeros(zero_attn_shape)
        v_zeros = jnp.zeros(zero_attn_shape)

        k = jnp.concatenate([k, k_zeros], axis=0)
        v = jnp.concatenate([v, v_zeros], axis=0)

        src_len += 1
        value_len += 1

    if bias_k is not None and bias_v is not None:
        bias_k = bias_k.reshape(1, num_heads, head_dim)
        bias_v = bias_v.reshape(1, num_heads, head_dim)

        k = jnp.concatenate([k, bias_k], axis=0)
        v = jnp.concatenate([v, bias_v], axis=0)

        src_len += 1
        value_len += 1

    assert src_len == value_len

    # [tgt_len, num_heads, head_dim] → [num_heads, tgt_len, head_dim]
    q = jnp.transpose(q, (1, 0, 2))

    # [src_len, num_heads, head_dim] → [num_heads, src_len, head_dim]
    k = jnp.transpose(k, (1, 0, 2))
    v = jnp.transpose(v, (1, 0, 2))

    scale = jnp.sqrt(head_dim)
    attn_output_weights = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / scale

    if key_padding_mask is not None:
        padding_mask = key_padding_mask.reshape(1, 1, src_len)
        padding_mask = jnp.repeat(padding_mask, num_heads, axis=0)
        padding_mask = jnp.repeat(padding_mask, tgt_len, axis=1)
        attn_output_weights = jnp.where(
            padding_mask, float("-inf"), attn_output_weights
        )

    if attn_mask is not None:
        # [tgt_len, src_len] -> [num_heads, tgt_len, src_len]
        mask = attn_mask.reshape(1, tgt_len, src_len)
        mask = jnp.repeat(mask, num_heads, axis=0)
        attn_output_weights = attn_output_weights + mask

    if is_causal:
        causal_mask = jnp.triu(jnp.ones((tgt_len, src_len)), k=1)
        causal_mask = (causal_mask == 1).reshape(1, tgt_len, src_len)
        causal_mask = jnp.repeat(causal_mask, num_heads, axis=0)
        attn_output_weights = jnp.where(causal_mask, float("-inf"), attn_output_weights)

    # [num_heads, tgt_len, src_len]
    attn_output_weights = jax.nn.softmax(attn_output_weights, axis=-1)

    if dropout_p > 0.0 and not inference:
        assert dropout_key is not None, (
            "dropout_key required because dropout_p > 0.0 and training"
        )
        dropout_mask = jax.random.bernoulli(
            dropout_key, 1 - dropout_p, attn_output_weights.shape
        )
        scale = 1.0 / (1.0 - dropout_p)
        attn_output_weights = attn_output_weights * dropout_mask * scale

    attn_output = jnp.matmul(attn_output_weights, v)
    attn_output = jnp.transpose(attn_output, (1, 0, 2))
    attn_output = attn_output.reshape(tgt_len, d_model)

    assert out_proj_weight is not None, "out_proj_weight must be provided"
    attn_output = attn_output @ out_proj_weight.T

    if out_proj_bias is not None:
        attn_output = attn_output + out_proj_bias

    if need_weights:
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(axis=0)
        return attn_output, attn_output_weights
    else:
        return attn_output, None


def create_attn_mask(
    pad_H: int,
    pad_W: int,
    window_size: list[int],
    shift_size: list[int],
    dtype: Any | None = None,
) -> Array:
    """Build the region-index mask used by shifted-window attention.

    Assigns each spatial position in a ``(pad_H, pad_W)`` feature map an
    integer *region index* that encodes which window it belongs to after a
    cyclic shift.  Positions in the same window share the same region index;
    positions in different windows have different indices.  The mask is used
    downstream to zero out cross-window attention scores.

    The region indices are computed by partitioning the height axis at
    ``pad_H - window_size[0]`` and ``pad_H - shift_size[0]``, and the width
    axis at ``pad_W - window_size[1]`` and ``pad_W - shift_size[1]``, then
    assigning a unique integer to each ``(row_region, col_region)`` cell.

    Args:
        pad_H: Padded feature map height (must be a multiple of
            ``window_size[0]``).
        pad_W: Padded feature map width (must be a multiple of
            ``window_size[1]``).
        window_size: Local attention window size as ``[window_H, window_W]``.
        shift_size: Cyclic shift amounts as ``[shift_H, shift_W]``.
        dtype: Output array dtype. Defaults to the project's default floating
            dtype when ``None``.

    Returns:
        Integer-region-index map of shape ``(pad_H, pad_W)`` cast to
        ``dtype``.

    Example:
        >>> mask = create_attn_mask(8, 8, window_size=[4, 4], shift_size=[2, 2])
        >>> mask.shape
        (8, 8)
    """
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None
    h_boundaries = jnp.array([pad_H - window_size[0], pad_H - shift_size[0]])
    w_boundaries = jnp.array([pad_W - window_size[1], pad_W - shift_size[1]])

    h_boundaries = jnp.sort(h_boundaries)
    w_boundaries = jnp.sort(w_boundaries)

    ii, jj = jnp.indices((pad_H, pad_W))  # ii for rows, jj for columns

    row_region_idx = jnp.searchsorted(h_boundaries, ii, side="right")
    col_region_idx = jnp.searchsorted(w_boundaries, jj, side="right")

    num_col_regions = len(w_boundaries) + 1
    attn_mask = row_region_idx * num_col_regions + col_region_idx

    return attn_mask.astype(dtype)


def shifted_window_attention(
    x: Float[Array, "H W C"],
    qkv_weight: Float[Array, "in_dim out_dim"],
    proj_weight: Float[Array, "out_dim out_dim"],
    relative_position_bias: Array,
    window_size: list[int],
    num_heads: int,
    shift_size: list[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Array | None = None,
    proj_bias: Array | None = None,
    logit_scale: Array | None = None,
    inference: bool = False,
    key: PRNGKeyArray | None = None,
) -> Float[Array, "H W C"]:
    """Apply Shifted-Window Multi-Head Self-Attention (Swin Attention).

    Implements the window-based self-attention mechanism from the Swin
    Transformer (Liu et al., 2021).  The input feature map is partitioned
    into non-overlapping local windows; attention is computed independently
    within each window.  A cyclic spatial shift (``shift_size``) is applied
    before partitioning to create cross-window connections, and a region-index
    mask is used to prevent attending across window boundaries introduced by
    the padding/shift.

    Supports two attention score variants:

    - **Scaled dot-product** (``logit_scale=None``): standard
      ``q @ k.T / sqrt(head_dim)``.
    - **Cosine attention** (when ``logit_scale`` is provided): L2-normalised
      ``q`` and ``k`` are used, and scores are multiplied by a learnable
      (but bounded) temperature ``exp(min(logit_scale, log(100)))``.

    Relative position biases are added to the attention logits before softmax.

    Args:
        x: Input feature map of shape ``(H, W, C)``.
        qkv_weight: Fused QKV projection weight of shape
            ``(in_dim, out_dim)`` where ``out_dim = 3 * C``.
        proj_weight: Output projection weight of shape ``(C, C)``.
        relative_position_bias: Relative position bias tensor added to
            attention logits; shape must broadcast with
            ``(num_windows, num_heads, window_H*window_W, window_H*window_W)``.
        window_size: Local window size as ``[window_H, window_W]``.
        num_heads: Number of attention heads. Must divide ``C`` evenly.
        shift_size: Cyclic shift amounts as ``[shift_H, shift_W]``.
            Use ``[0, 0]`` to disable shifting (regular window attention).
        attention_dropout: Dropout probability applied to attention weights.
            Defaults to ``0.0``.
        dropout: Dropout probability applied to the output projection.
            Defaults to ``0.0``.
        qkv_bias: Optional bias for the QKV projection of shape
            ``(3 * C,)``. When ``logit_scale`` is also provided, the key
            bias component is zeroed out (as in Swin V2).
        proj_bias: Optional bias for the output projection of shape ``(C,)``.
        logit_scale: Learnable log-scale scalar for cosine attention. When
            ``None``, standard scaled dot-product attention is used.
        inference: If ``True``, disable dropout. Defaults to ``False``.
        key: JAX PRNG key required when ``inference=False``.

    Returns:
        Output feature map of shape ``(H, W, C)``.

    Raises:
        ValueError: If ``inference=False`` and ``key`` is ``None``.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> H, W, C, num_heads = 8, 8, 16, 2
        >>> window_size, shift_size = [4, 4], [2, 2]
        >>> key = jax.random.PRNGKey(0)
        >>> x = jax.random.normal(key, (H, W, C))
        >>> qkv_w = jax.random.normal(key, (C, 3 * C))
        >>> proj_w = jax.random.normal(key, (C, C))
        >>> win_tokens = window_size[0] * window_size[1]
        >>> num_wins = (H // window_size[0]) * (W // window_size[1])
        >>> rpb = jnp.zeros((num_wins, num_heads, win_tokens, win_tokens))
        >>> out = shifted_window_attention(
        ...     x, qkv_w, proj_w, rpb,
        ...     window_size=window_size, num_heads=num_heads,
        ...     shift_size=shift_size, inference=True,
        ... )
        >>> out.shape
        (8, 8, 16)
    """
    if not inference and key is None:
        raise ValueError("Need key when in training mode")
    H, W, C = x.shape
    to_pad_W = (window_size[1] - W % window_size[1]) % window_size[1]
    to_pad_H = (window_size[0] - H % window_size[0]) % window_size[0]
    x = jnp.pad(x, ((0, to_pad_H), (0, to_pad_W), (0, 0)))
    pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = jnp.roll(x, shift=(-shift_size[0], -shift_size[1]), axis=(0, 1))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = jnp.reshape(
        x,
        (
            pad_H // window_size[0],
            window_size[0],
            pad_W // window_size[1],
            window_size[1],
            C,
        ),
    )
    x = jnp.transpose(x, (0, 2, 1, 3, 4)).reshape(
        num_windows, window_size[0] * window_size[1], C
    )

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        length = qkv_bias.size // 3
        qkv_bias = qkv_bias.at[length : 2 * length].set(0.0)

    def linear(x: Array, weight: Array, bias: Array | None):
        output = x @ jnp.transpose(weight)  # (in,) @ (in, out) -> (out,)
        if bias is not None:
            output = output + bias
        return output

    linear_pt = functools.partial(linear, weight=qkv_weight, bias=qkv_bias)

    qkv = eqx.filter_vmap(eqx.filter_vmap(linear_pt))(x)
    win_size, patches, _ = qkv.shape
    qkv = jnp.transpose(
        qkv.reshape(win_size, patches, 3, num_heads, C // num_heads), (2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = normalize(q, axis=-1) @ jnp.transpose(
            normalize(k, axis=-1), (0, 1, 3, 2)
        )
        # Clamp the logit scale exponent for stability
        logit_scale = jnp.exp(jnp.minimum(logit_scale, jnp.log(jnp.array(100.0))))
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        # attn = q @ (jnp.transpose(normalize(k, axis=-1), (0, 1, 3, 2))) # Incorrect
        attn = q @ jnp.transpose(k, (0, 1, 3, 2))  # Corrected: q @ k.T

    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        attn_mask = create_attn_mask(pad_H, pad_W, window_size, shift_size)
        attn_mask = attn_mask.reshape(
            pad_H // window_size[0],
            window_size[0],
            pad_W // window_size[1],
            window_size[1],
        )
        attn_mask = jnp.transpose(attn_mask, (0, 2, 1, 3)).reshape(
            num_windows, window_size[0] * window_size[1]
        )
        attn_mask = jnp.expand_dims(attn_mask, axis=1) - jnp.expand_dims(
            attn_mask, axis=2
        )
        attn_mask = jnp.where(attn_mask == 0, 0.0, -100.0)

        attn = attn + attn_mask[:, None, :, :]

    attn = jax.nn.softmax(attn, axis=-1)
    if not inference:
        assert key is not None, "key must be given if not inference"
        key, subkey = jax.random.split(key)
        attn = dropout_fn(attn, p=attention_dropout, inference=inference, key=subkey)

    x = jnp.transpose(attn @ v, (0, 2, 1, 3)).reshape(
        num_windows, window_size[0] * window_size[1], C
    )
    linear_pt_proj = functools.partial(linear, weight=proj_weight, bias=proj_bias)

    x = eqx.filter_vmap(eqx.filter_vmap(linear_pt_proj))(x)
    if not inference:
        assert key is not None, "key must be given if not inference"
        key, subkey = jax.random.split(key)
        x = dropout_fn(x, p=dropout, inference=inference, key=subkey)

    # reverse windows
    x = x.reshape(
        pad_H // window_size[0],
        pad_W // window_size[1],
        window_size[0],
        window_size[1],
        C,
    )
    x = jnp.transpose(x, (0, 2, 1, 3, 4)).reshape(pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = jnp.roll(x, shift=(shift_size[0], shift_size[1]), axis=(0, 1))

    # unpad features
    x = x[:H, :W, :]
    return x
