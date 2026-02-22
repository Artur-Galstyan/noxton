import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTree


def kaiming_init_conv2d(model: PyTree, state: eqx.nn.State, key: PRNGKeyArray):
    """Apply Kaiming He normal initialization to all ``Conv2d`` weights in a model.

    Traverses the pytree ``model``, finds every ``eqx.nn.Conv2d`` leaf, and
    replaces its ``weight`` array with a sample drawn from the He normal
    distribution (also known as *Kaiming normal*).  Bias parameters and all
    other leaves are left untouched.

    Args:
        model: An Equinox pytree that may contain one or more
            ``eqx.nn.Conv2d`` layers.
        state: The Equinox ``State`` object associated with ``model``.
            Passed through unchanged.
        key: JAX PRNG key used to generate the new weights.  A separate
            sub-key is created for each ``Conv2d`` layer found.

    Returns:
        A ``(model, state)`` tuple where ``model`` has all ``Conv2d`` weights
        re-initialised with He normal values and ``state`` is the original
        state object unchanged.  If no ``Conv2d`` layers are found, both are
        returned unmodified.

    Example:
        >>> import jax
        >>> import equinox as eqx
        >>> key = jax.random.PRNGKey(0)
        >>> model = eqx.nn.Conv2d(3, 16, 3, key=key)
        >>> state = eqx.nn.State(model)
        >>> model, state = kaiming_init_conv2d(model, state, key)
        >>> model.weight.shape
        (16, 3, 3, 3)
    """
    # Filter function to identify Conv2d layers
    is_conv2d = lambda x: isinstance(x, eqx.nn.Conv2d)

    # Function to get weights (leaves) based on the filter
    def get_weights(model):
        return [
            x.weight for x in jax.tree.leaves(model, is_leaf=is_conv2d) if is_conv2d(x)
        ]

    # Get the list of current weights
    weights = get_weights(model)
    if not weights:  # If no Conv2d layers found
        return model, state

    # Create new weights using He initializer
    initializer = jax.nn.initializers.he_normal()
    # Split key for each weight tensor
    subkeys = jax.random.split(key, len(weights))
    new_weights = [
        initializer(subkeys[i], w.shape, w.dtype)  # Use original weight's dtype
        for i, w in enumerate(weights)
    ]

    # Replace old weights with new weights in the model pytree
    model = eqx.tree_at(get_weights, model, new_weights)

    return model, state
