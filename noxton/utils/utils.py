import os
import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import PyTree


def default_floating_dtype() -> Any:
    if jax.config.read("jax_enable_x64"):
        return jnp.float64
    else:
        return jnp.float32


def summarize_model(model: PyTree) -> str:
    params, _ = eqx.partition(model, eqx.is_array)

    param_counts = {}
    total_params = 0

    def count_params(pytree, name=""):
        nonlocal total_params
        count = 0
        if isinstance(pytree, jnp.ndarray):
            count = pytree.size
            total_params += count
            if name:
                param_counts[name] = count
        elif hasattr(pytree, "__dict__"):
            for key, value in pytree.__dict__.items():
                subname = f"{name}.{key}" if name else key
                count += count_params(value, subname)
        elif isinstance(pytree, (list, tuple)):
            for i, value in enumerate(pytree):
                subname = f"{name}[{i}]" if name else f"[{i}]"
                count += count_params(value, subname)
        elif isinstance(pytree, dict):
            for key, value in pytree.items():
                subname = f"{name}.{key}" if name else str(key)
                count += count_params(value, subname)
        return count

    count_params(params)

    # Display as table
    lines = []
    lines.append("Model Parameter Summary")
    lines.append("=" * 50)
    lines.append(f"{'Parameter Name':<30} {'Count':<15}")
    lines.append("-" * 50)

    for name, count in param_counts.items():
        lines.append(f"{name:<30} {count:<15,}")

    lines.append("-" * 50)
    lines.append(f"{'Total Parameters':<30} {total_params:<15,}")
    lines.append("=" * 50)

    return "\n".join(lines)


def default_training_dtype():
    return jnp.bfloat16


def dtype_to_str(dtype) -> str:
    if hasattr(dtype, "__name__"):
        # For simple types like jnp.float32, float, etc.
        return dtype.__name__

    # For more complex types like JaxArray with specific dtype
    dtype_str = str(dtype)

    # Remove common patterns to clean up the string
    if "<class '" in dtype_str:
        # Extract the name from "<class 'jax.numpy.float32'>" -> "float32"
        dtype_str = dtype_str.split("'")[1].split(".")[-1]

    return dtype_str


def get_cache_path(model: str):
    noxton_dir = os.path.expanduser(f"~/.noxton/models/{model}")
    os.makedirs(noxton_dir, exist_ok=True)
    return pathlib.Path(noxton_dir)
