"""
Tensorflow implementation of the core algorithm in the matmul-less Dense layer.
"""

from typing import Tuple

import tensorflow as tf
from overrides import override

from keras_mml.layers._dense_impl.base_dense import EPSILON, HUGE, BaseDenseMML


class TensorflowDenseMML(BaseDenseMML):
    """
    Dense layer without matrix multiplication, implemented in Tensorflow.
    """

    @staticmethod
    def _activations_quantization(x: tf.Tensor) -> tf.Tensor:
        """
        Quantizes the activations to 8-bit precision using absmax quantization.

        Args:
            x: Tensor of quantization values.

        Returns:
            The quantized activation values.
        """

        scale = 127.0 / tf.clip_by_value(tf.reduce_max(tf.abs(x), axis=-1, keepdims=True), EPSILON, HUGE)
        y = tf.clip_by_value(tf.round(x * scale), -128, 127) / scale
        return y

    @staticmethod
    def _weights_quantization(w: tf.Tensor) -> tf.Tensor:
        """
        Quantizes the weights to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

        Args:
            w: Array of weights.

        Returns:
            The quantized weights.
        """

        scale = 1.0 / tf.clip_by_value(tf.reduce_mean(tf.abs(w)), EPSILON, HUGE)
        u = tf.clip_by_value(tf.round(w * scale), -1, 1) / scale
        return u

    @override(check_signature=False)
    def _get_quantized_arrays(self, x_norm: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Get the quantized activations and weights
        x_quantized = x_norm + tf.stop_gradient(
            self._activations_quantization(x_norm) - x_norm
        )  # STE trick by stopping gradient propagation

        if self._weight_scale:
            # Weights should have been pre-quantized
            w_quantized = self.w / self._weight_scale
        else:
            w: tf.Tensor = self.w.value
            w_quantized = w + tf.stop_gradient(self._weights_quantization(w) - w)

        return x_quantized, w_quantized
