"""
Jax implementation of the core algorithm in the matmul-less Dense layer.
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from keras_mml.layers.core._dense_impl.base_dense import EPSILON, HUGE, BaseDenseMML


@jax.jit
def _activations_quantization(x: jax.Array) -> jax.Array:
    """
    Quantizes the activations to 8-bit precision using absmax quantization.

    Args:
        x: Array of quantization values.

    Returns:
        The quantized activation values.
    """

    scale = 127.0 / jnp.expand_dims(jnp.clip(jnp.max(jnp.abs(x), axis=-1), EPSILON, HUGE), -1)
    y = jnp.clip(jnp.round(x * scale), -128, 127) / scale
    return y


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
def _kernel_quantization_for_training(w: jax.Array) -> jax.Array:
    """
    Quantizes the kernel values to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

    Args:
        w: Kernel matrix.

    Returns:
        The quantized kernel with the scaling applied.
    """

    scale = _compute_kernel_scale(w)
    u = jnp.clip(jnp.round(w * scale), -1, 1)
    return u / scale


@jax.jit
def _kernel_quantization_for_saving(w: jax.Array) -> Tuple[jax.Array, float]:
    """
    Quantizes the kernel values to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

    Args:
        w: Kernel matrix.

    Returns:
        Both the quantized kernel and the scale will be returned, with the scale **not** applied to
        the quantized kernel.
    """

    scale = _compute_kernel_scale(w)
    u = jnp.clip(jnp.round(w * scale), -1, 1)
    return u, scale


@jax.jit
def _get_x_quantized(x_norm: jax.Array) -> jax.Array:
    """
    Gets the quantized activations, with support for the backward direction by using STE gradient
    bypass.

    We use a Straight-Through Estimator (STE) trick by stopping gradient propagation.

    Args:
        x_norm: Normalized activation values.

    Returns:
        Quantized activation values.
    """

    return x_norm + jax.lax.stop_gradient(_activations_quantization(x_norm) - x_norm)


@jax.jit
def _get_w_quantized(w: jax.Array) -> jax.Array:
    """
    Gets the quantized kernel matrix, with support for the backward direction by using STE gradient
    bypass.

    We use a Straight-Through Estimator (STE) trick by stopping gradient propagation.

    Args:
        w: Kernel matrix.

    Returns:
        Quantized kernel matrix.
    """

    return w + jax.lax.stop_gradient(_kernel_quantization_for_training(w) - w)


@jax.jit
def _get_quantized_arrays_for_training(x_norm, w):
    """
    Gets the quantized activation and kernel values for training the model.

    Args:
        x_norm: Normalized activation values.
        w: Kernel matrix.

    Returns:
        A tuple. The first value is the quantized activation values. The second is the quantized
        kernel values.
    """

    x_quantized = _get_x_quantized(x_norm)
    w_quantized = _get_w_quantized(w)
    return x_quantized, w_quantized


@jax.jit
def _get_quantized_arrays_for_inference(x_norm, w, w_scale):
    """
    Gets the quantized activation and kernel values for inference.

    Args:
        x_norm: Normalized activation values.
        w: Kernel matrix.
        w_scale: Scaling factor for the kernel matrix.

    Returns:
        A tuple. The first value is the quantized activation values. The second is the quantized
        kernel values.
    """

    x_quantized = _get_x_quantized(x_norm)
    w_quantized = w / w_scale
    return x_quantized, w_quantized


class JaxDenseMML(BaseDenseMML):
    @staticmethod
    def _activations_quantization(x):
        return _activations_quantization(x)

    @staticmethod
    def _compute_kernel_scale(w) -> float:
        return _compute_kernel_scale(w)

    @staticmethod
    def _kernel_quantization_for_training(w):
        return _kernel_quantization_for_training(w)

    @staticmethod
    def _kernel_quantization_for_saving(w) -> Tuple[Any, float]:
        return _kernel_quantization_for_saving(w)

    def _get_quantized_arrays(self, x_norm) -> Tuple[Any, Any]:
        if self._kernel_scale:
            return _get_quantized_arrays_for_inference(x_norm, self._kernel, self._kernel_scale)
        else:
            return _get_quantized_arrays_for_training(x_norm, self._kernel)
