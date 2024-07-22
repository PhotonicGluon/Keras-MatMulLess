"""
Jax implementation of the core algorithm in the matmul-less Dense layer.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from keras_mml.layers.core._dense_impl.base_dense import EPSILON, HUGE, BaseDenseMML


@jax.jit
def _compute_kernel_scale(w: jax.Array) -> float:
    """
    Computes the scale factor of the kernel matrix.

    Args:
        w: Kernel matrix.

    Returns:
        Scale factor.
    """

    return 1.0 / jnp.clip(jnp.mean(jnp.abs(w)), EPSILON, HUGE)


@jax.jit
def _quantize_kernel(w: jax.Array, scale: float) -> jax.Array:
    """
    Quantizes the kernel values to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

    Args:
        w: Kernel matrix.
        scale: Scaling factor.

    Returns:
        The quantized kernel without scaling applied.
    """

    return jnp.clip(jnp.round(w * scale), -1, 1)


@jax.jit
def _get_w_quantized(w: jax.Array, scale: float) -> jax.Array:
    """
    Gets the quantized kernel matrix, with support for the backward direction by using STE gradient
    bypass.

    We use a Straight-Through Estimator (STE) trick by stopping gradient propagation.

    Args:
        w: Kernel matrix.
        scale: Scale factor.

    Returns:
        Quantized kernel matrix without scaling applied.
    """

    return w + jax.lax.stop_gradient(_quantize_kernel(w, scale) - w)


class JaxDenseMML(BaseDenseMML):
    """
    Implementation of the Dense layer using the Jax backend.
    """

    @staticmethod
    def _compute_kernel_scale(w: jax.Array) -> float:
        return _compute_kernel_scale(w)

    @staticmethod
    def _quantize_kernel(w: jax.Array, scale: float) -> jax.Array:
        return _quantize_kernel(w, scale)

    def _get_quantized_arrays(self) -> Tuple[jax.Array, float]:
        if self._kernel_scale:
            return self._kernel.value, self._kernel_scale

        scale = _compute_kernel_scale(self._kernel.value)
        return _get_w_quantized(self._kernel.value, scale), scale
