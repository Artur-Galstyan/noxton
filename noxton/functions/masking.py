import jax.numpy as jnp
from jaxtyping import Array, Bool


def canonical_mask(
    mask,
    mask_name,
    other_name="",
    other_type=None,
    target_type=jnp.float32,
    other_mask=None,
    check_other=True,
):
    """Convert an arbitrary mask tensor into a canonical additive float mask.

    Accepts boolean, integer, or floating-point masks and normalises them to
    a floating-point additive mask (``0.0`` for positions to attend to,
    ``-inf`` for positions to ignore) that can be directly added to attention
    logits.

    - **Boolean masks**: ``True`` → ``-inf``, ``False`` → ``0.0``
    - **Integer / float masks**: cast to ``target_type`` without modification.
    - ``None`` input returns ``None`` (no mask).

    Args:
        mask: The mask to canonicalise. May be a boolean, integer, or
            floating-point JAX array, or ``None``.
        mask_name: Human-readable name of this mask used in error messages.
        other_name: Human-readable name of the secondary mask (used in error
            messages only). Defaults to ``""``.
        other_type: Expected dtype of ``other_mask`` (currently unused in the
            implementation but kept for API compatibility). Defaults to
            ``None``.
        target_type: Target floating-point dtype for the output. Defaults to
            ``jnp.float32``.
        other_mask: A secondary mask for cross-validation purposes (currently
            unused). Defaults to ``None``.
        check_other: Whether to perform cross-mask validation (currently
            unused). Defaults to ``True``.

    Returns:
        A floating-point array of dtype ``target_type`` with the same shape
        as ``mask``, or ``None`` if ``mask`` is ``None``.

    Raises:
        TypeError: If ``mask`` has an unsupported dtype (not bool, int, or
            float).

    Example:
        >>> import jax.numpy as jnp
        >>> bool_mask = jnp.array([True, False, True])
        >>> canonical_mask(bool_mask, "attn_mask")
        Array([-inf,   0., -inf], dtype=float32)

        >>> float_mask = jnp.array([0.0, -1e9, 0.0])
        >>> canonical_mask(float_mask, "attn_mask")
        Array([ 0.e+00, -1.e+09,  0.e+00], dtype=float32)
    """
    if mask is None:
        return None
    if mask.dtype == bool:
        additive_mask = jnp.where(mask, -jnp.inf, 0.0).astype(target_type)
        return additive_mask
    elif jnp.issubdtype(mask.dtype, jnp.integer) or jnp.issubdtype(
        mask.dtype, jnp.floating
    ):
        return mask.astype(target_type)
    else:
        raise TypeError(
            f"{mask_name} must be bool, int, or float tensor, but got {mask.dtype}"
        )


def canonical_key_padding_mask(
    key_padding_mask, attn_mask=None, query_dtype=jnp.float32
):
    """Convert a key-padding mask to a canonical additive float mask.

    A convenience wrapper around :func:`canonical_mask` that applies the
    correct argument names for key-padding masks.  The resulting mask can be
    directly added to attention logits: padded positions get ``-inf`` (boolean
    ``True`` input) while real positions get ``0.0``.

    Args:
        key_padding_mask: Mask of shape ``(src_len,)`` indicating which key
            positions are padding.  May be boolean (``True`` = padded),
            integer, or float, or ``None``.
        attn_mask: The attention mask that will be combined with this mask
            (used for error reporting only). Defaults to ``None``.
        query_dtype: Target floating-point dtype for the output. Defaults to
            ``jnp.float32``.

    Returns:
        A floating-point array of dtype ``query_dtype`` or ``None``.

    Example:
        >>> import jax.numpy as jnp
        >>> kpm = jnp.array([False, False, True])  # last token is padding
        >>> canonical_key_padding_mask(kpm)
        Array([  0.,   0., -inf], dtype=float32)
    """
    return canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_name="attn_mask",
        other_mask=attn_mask,
        target_type=query_dtype,
    )


def canonical_attn_mask(attn_mask, query_dtype=jnp.float32):
    """Convert an attention mask to a canonical additive float mask.

    A convenience wrapper around :func:`canonical_mask` that applies the
    correct argument names for attention masks.  Boolean masks are converted
    so that ``True`` positions are masked out (``-inf``) and ``False``
    positions are kept (``0.0``).  Numeric masks are cast to ``query_dtype``
    without modification, allowing pre-built additive masks (e.g. causal
    masks with ``0`` and ``-inf``) to be passed through unchanged.

    Args:
        attn_mask: Attention mask of shape ``(tgt_len, src_len)``.  May be
            boolean, integer, or floating-point, or ``None``.
        query_dtype: Target floating-point dtype for the output. Defaults to
            ``jnp.float32``.

    Returns:
        A floating-point array of dtype ``query_dtype`` or ``None``.

    Example:
        >>> import jax.numpy as jnp
        >>> mask = jnp.array([[False, True], [False, False]])
        >>> canonical_attn_mask(mask)
        Array([[  0., -inf],
               [  0.,   0.]], dtype=float32)
    """
    return canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=query_dtype,
        check_other=False,
    )


def make_causal_mask(seq_len: int) -> Bool[Array, "seq_len seq_len"]:
    """Create a boolean lower-triangular causal mask.

    Position ``[i, j]`` is ``True`` when position ``j`` is allowed to attend
    to position ``i`` (i.e. ``j <= i``), and ``False`` otherwise.  Use this
    mask to prevent tokens from attending to future positions.

    Args:
        seq_len: Sequence length; the output is a square matrix of shape
            ``(seq_len, seq_len)``.

    Returns:
        Boolean array of shape ``(seq_len, seq_len)`` where the lower
        triangle (including the diagonal) is ``True`` and the upper triangle
        is ``False``.

    Example:
        >>> make_causal_mask(3)
        Array([[ True, False, False],
               [ True,  True, False],
               [ True,  True,  True]], dtype=bool)
    """
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))


def build_attention_mask(context_length: int) -> Array:
    """Create an additive causal attention mask with ``0`` and ``-inf``.

    Produces a float32 lower-triangular matrix where attended positions
    carry ``0.0`` (no change to logits) and future positions carry ``-inf``
    (zeroed out after softmax).  The mask can be added directly to attention
    logit matrices.

    Args:
        context_length: Sequence length; the output is a square matrix of
            shape ``(context_length, context_length)``.

    Returns:
        Float32 array of shape ``(context_length, context_length)`` with
        ``0.0`` on and below the diagonal and ``-inf`` above the diagonal.

    Example:
        >>> build_attention_mask(3)
        Array([[  0., -inf, -inf],
               [  0.,   0., -inf],
               [  0.,   0.,   0.]], dtype=float32)
    """
    mask = jnp.tril(jnp.zeros((context_length, context_length)))
    upper = jnp.triu(jnp.full((context_length, context_length), float("-inf")), k=1)

    mask = mask + upper
    return mask
