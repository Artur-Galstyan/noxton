import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def stochastic_depth(
    input: Array,
    p: float,
    mode: str,
    inference: bool,
    key: PRNGKeyArray,
) -> Array:
    """Apply Stochastic Depth regularization (DropPath).

    During training, randomly drops entire samples (``mode="batch"``) or
    individual rows (``mode="row"``). At inference time or when ``p=0``, the
    input is returned unchanged. Surviving elements are scaled by
    ``1 / (1 - p)`` to preserve the expected value.

    Args:
        input: Input array to apply stochastic depth to.
        p: Drop probability in ``[0, 1]``. The probability that a sample/row
            is zeroed out. ``p=0`` disables the operation.
        mode: Dropping granularity. One of:
            - ``"batch"``: a single binary mask is broadcast over the whole
              batch (all samples share the same fate).
            - ``"row"``: each sample in the batch gets its own independent
              mask.
        inference: If ``True``, skip stochastic depth and return ``input``
            unchanged (equivalent to test/eval mode).
        key: JAX PRNG key used to sample the Bernoulli noise.

    Returns:
        Array of the same shape and dtype as ``input``.

    Raises:
        ValueError: If ``p`` is outside ``[0, 1]``.
        ValueError: If ``mode`` is not ``"batch"`` or ``"row"``.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> x = jnp.ones((4, 8))
        >>> # row mode: each of the 4 samples may be dropped independently
        >>> out = stochastic_depth(x, p=0.5, mode="row", inference=False, key=key)
        >>> out.shape
        (4, 8)

        >>> # inference mode always returns the input unchanged
        >>> stochastic_depth(x, p=0.9, mode="batch", inference=True, key=key) is x
        True
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if inference or p == 0.0:
        return input
    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = jax.random.bernoulli(key, p=survival_rate, shape=size).astype(input.dtype)
    if survival_rate > 0.0:
        noise = noise / survival_rate
    return input * noise


def dropout(
    x: Array,
    p: float,
    inference: bool,
    key: PRNGKeyArray | None = None,
) -> Array:
    """Apply dropout regularization to an array.

    During training, each element is independently zeroed with probability
    ``p``. Surviving elements are scaled by ``1 / (1 - p)`` so the expected
    sum is preserved (inverted dropout). At inference time or when ``p=0``
    the input is returned unchanged.

    Args:
        x: Input array.
        p: Probability of an element being zeroed. Must be in ``[0, 1)``.
            When ``p=0`` the function is a no-op regardless of ``inference``.
        inference: If ``True``, return ``x`` unchanged (eval/test mode).
        key: JAX PRNG key. Required when ``inference=False`` and ``p > 0``.
            Defaults to ``None``.

    Returns:
        Array of the same shape and dtype as ``x``.

    Raises:
        RuntimeError: If ``inference=False``, ``p > 0``, and ``key`` is
            ``None``.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(42)
        >>> x = jnp.ones((3, 4))
        >>> out = dropout(x, p=0.5, inference=False, key=key)
        >>> out.shape
        (3, 4)

        >>> # inference mode: output equals input
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> dropout(x, p=0.5, inference=True)
        Array([1., 2., 3.], dtype=float32)
    """
    if isinstance(p, (int, float)) and p == 0:
        inference = True
    if inference:
        return x
    elif key is None:
        raise RuntimeError(
            "Dropout requires a key when running in non-deterministic mode."
        )
    else:
        q = 1 - jax.lax.stop_gradient(p)
        mask = jax.random.bernoulli(key, q, x.shape)
        return jnp.where(mask, x / q, 0)
