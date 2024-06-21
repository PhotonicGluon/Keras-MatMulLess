"""
Base class for all matmul-less dense layers.
"""

from typing import Any, Tuple

import keras
from keras import ops

EPSILON = 1e-5
HUGE = 1e9


class BaseDenseMML:
    """
    Base dense layer that exposes methods that needs to be overridden.
    """

    def __init__(self):
        """
        Initialization method.
        """

        #: Variable storing the weights matrix.
        self.w: keras.Variable = None

        self._weight_scale = None  #: Used for when the layer is loaded from file

    # Helper methods
    @staticmethod
    def _activations_quantization(x):
        """
        Quantizes the activations to 8-bit precision using absmax quantization.

        Args:
            x: Array of quantization values.

        Returns:
            The quantized activation values.
        """

        scale = 127.0 / ops.clip(ops.max(ops.abs(x), axis=-1, keepdims=True), EPSILON, HUGE)
        y = ops.clip(ops.round(x * scale), -128, 127) / scale
        return y

    @staticmethod
    def _weights_quantization(w) -> Any:
        """
        Quantizes the weights to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

        Args:
            w: Array of weights.

        Returns:
            The quantized weights.
        """

        scale = 1.0 / ops.clip(ops.mean(ops.abs(w)), EPSILON, HUGE)
        u = ops.clip(ops.round(w * scale), -1, 1) / scale
        return u

    def _get_quantized_arrays(self, x_norm) -> Tuple[Any, Any]:
        """
        Gets the quantized activation and weight values.

        Args:
            x_norm: Normalized activation values.

        Returns:
            A tuple. The first value is the quantized activation values. The second is the quantized
            weight values.
        """

        raise NotImplementedError
