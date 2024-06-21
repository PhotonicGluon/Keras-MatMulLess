"""
Tensorflow implementation of the core algorithm in the matmul-less Dense layer.
"""

from typing import Tuple

import tensorflow as tf
from overrides import override

from keras_mml.layers._dense_impl.base_dense import BaseDenseMML


class TensorflowDenseMML(BaseDenseMML):
    """
    Dense layer without matrix multiplication, implemented in Tensorflow.
    """

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
