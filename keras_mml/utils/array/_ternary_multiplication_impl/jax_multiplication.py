"""
Jax-optimized implementation of ternary multiplication.
"""

import jax
import jax.numpy as jnp


def jax_ternary_multiplication(x_quantized: jax.Array, w_quantized: jax.Array, w_scale: float) -> jax.Array:
    """
    Applies the ternary multiplication algorithm.

    Args:
        x_quantized: Quantized activation values.
        w_quantized: Quantized kernel matrix without scaling applied.
        w_scale: Scale factor for the kernel matrix.

    Returns:
        Multiplied matrix.
    """

    # TODO: Optimize
    return jnp.matmul(x_quantized, w_quantized / w_scale)  # The `matmul` should just involve addition and subtraction
