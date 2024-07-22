"""
Tensorflow implementation for the quantized RMSNorm layer.
"""

import keras
import tensorflow as tf
from jaxtyping import Float

from keras_mml.layers.normalizations._quant_rms_norm_impl.base_quant_rms_norm import EPSILON, HUGE, BaseQuantRMSNorm


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


@keras.saving.register_keras_serializable(package="keras_mml")
class TensorflowQuantRMSNorm(BaseQuantRMSNorm):
    """
    Tensorflow implementation of Root Mean Square Normalization (RMSNorm) with 8-bit quantization.
    """

    def call(self, inputs: Float[tf.Tensor, "batch_size *dims"]) -> Float[tf.Tensor, "batch_size *dims"]:
        x_norm = super().call(inputs)
        return _get_x_quantized(x_norm)
