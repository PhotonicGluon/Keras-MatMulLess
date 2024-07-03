"""
Tensorflow implementation of the core algorithm in the matmul-less Dense layer.
"""

from typing import Tuple

import tensorflow as tf

from keras_mml.layers.core._dense_impl.base_dense import EPSILON, HUGE, BaseDenseMML


@tf.function(jit_compile=True)
def _activations_quantization(x: tf.Tensor) -> tf.Tensor:
    """
    Quantizes the activations to 8-bit precision using absmax quantization.

    Args:
        x: Array of quantization values.

    Returns:
        The quantized activation values.
    """

    scale = 127.0 / tf.clip_by_value(tf.reduce_max(tf.abs(x), axis=-1, keepdims=True), EPSILON, HUGE)
    y = tf.clip_by_value(tf.round(x * scale), -128, 127) / scale
    return y


@tf.function(jit_compile=True)
def _compute_kernel_scale(w: tf.Tensor) -> float:
    """
    Computes the scale factor of the kernel matrix.

    Args:
        w: Kernel matrix.

    Returns:
        Scale factor.
    """

    return 1.0 / tf.clip_by_value(tf.reduce_mean(tf.abs(w)), EPSILON, HUGE)


@tf.function(jit_compile=True)
def _quantize_kernel(w: tf.Tensor, scale: float) -> tf.Tensor:
    """
    Quantizes the kernel values to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

    Args:
        w: Kernel matrix.
        scale: Scaling factor.

    Returns:
        The quantized kernel without scaling applied.
    """

    return tf.clip_by_value(tf.round(w * scale), -1, 1)


@tf.function(jit_compile=True)
def _get_x_quantized(x_norm: tf.Tensor) -> tf.Tensor:
    """
    Gets the quantized activations, with support for the backward direction by using STE gradient
    bypass.

    We use a Straight-Through Estimator (STE) trick by stopping gradient propagation.

    Args:
        x_norm: Normalized activation values.

    Returns:
        Quantized activation values.
    """

    return x_norm + tf.stop_gradient(_activations_quantization(x_norm) - x_norm)


@tf.function(jit_compile=True)
def _get_w_quantized(w: tf.Tensor, scale: float) -> tf.Tensor:
    """
    Gets the quantized kernel matrix, with support for the backward direction by using STE gradient
    bypass.

    We use a Straight-Through Estimator (STE) trick by stopping gradient propagation.

    Args:
        w: Kernel matrix.
        scale: Scaling factor.

    Returns:
        Quantized kernel matrix without scaling applied.
    """

    return w + tf.stop_gradient(_quantize_kernel(w, scale) - w)


class TensorflowDenseMML(BaseDenseMML):
    """
    Implementation of the Dense layer using the Tensorflow backend.
    """

    @staticmethod
    def _compute_kernel_scale(w: tf.Tensor) -> float:
        return _compute_kernel_scale(w)

    @staticmethod
    def _quantize_kernel(w: tf.Tensor, scale: float) -> tf.Tensor:
        return _quantize_kernel(w, scale)

    def _get_quantized_arrays(self, x_norm: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self._kernel_scale:
            return _get_x_quantized(x_norm), self._kernel.value, self._kernel_scale

        # Need this to avoid nasty "Called a function referencing variables which have been deleted" error
        w = tf.identity(self._kernel.value)
        scale = _compute_kernel_scale(w)
        return _get_x_quantized(x_norm), _get_w_quantized(w, scale), scale
