import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from noxton.functions import stochastic_depth


class StochasticDepth(eqx.Module):
    """Stochastic Depth (DropPath) regularisation layer.

    Wraps the ``stochastic_depth`` function as a stateful Equinox module.
    During training, randomly drops the entire input tensor (``mode="batch"``)
    or individual rows (``mode="row"``) with probability ``p``.  Surviving
    elements are rescaled by ``1 / (1 - p)`` to preserve expected values.
    At inference time the layer is a no-op.

    Args:
        p: Drop probability in ``[0, 1]``.
        mode: Dropping granularity. ``"batch"`` applies a single mask
            broadcast across the whole tensor; ``"row"`` gives each row an
            independent mask.
        inference: If ``True``, always return the input unchanged.  Defaults
            to ``False``.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> sd = StochasticDepth(p=0.2, mode="row")
        >>> x = jnp.ones((4, 8))
        >>> sd(x, key).shape
        (4, 8)

        >>> # inference mode: always passes through unchanged
        >>> sd_inf = StochasticDepth(p=0.5, mode="batch", inference=True)
        >>> sd_inf(x, key) is x
        False  # returns same values but not identical object in JAX
    """

    p: float = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    inference: bool

    def __init__(self, p: float, mode: str, inference: bool = False) -> None:
        self.p = p
        self.mode = mode
        self.inference = inference

    def __call__(self, input: Array, key: PRNGKeyArray) -> Array:
        """Apply stochastic depth to ``input``.

        Args:
            input: Input array to potentially drop.
            key: JAX PRNG key for sampling the drop mask.

        Returns:
            Array of the same shape and dtype as ``input``, either zeroed or
            rescaled depending on the sampled mask, or ``input`` unchanged
            during inference.
        """
        return stochastic_depth(input, self.p, self.mode, self.inference, key)
