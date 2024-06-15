"""
Implements the BitLinear layer.
"""

import keras
from keras import ops

EPSILON = 1e-5
HUGE = 1e9


@keras.saving.register_keras_serializable(package="keras_mml")
class BitLinear(keras.Layer):
    """
    TODO: ADD DOCS
    """

    @staticmethod
    def _activations_quantization(x):
        """
        Quantizes the activations to 8-bit precision using absmax quantization.

        This is equation (4) in the aforementioned paper.

        Also pre-undoes part of the scaling in (11) by dividing the clipped values by the
        scale.

        Args:
            x: Array of quantization values.

        Returns:
            Quantized activation values.
        """

        absolutes = ops.abs(x)
        maxima = ops.max(absolutes, axis=-1, keepdims=True)
        clipped = ops.clip(maxima, EPSILON, HUGE)  # This is gamma in the paper
        scale = 127.0 / clipped

        rescaled = x * scale
        rounded = ops.round(rescaled)
        clipped = ops.clip(rounded, -128, 127)

        y = clipped / scale  # Perform part of the undoing in eq. (11)
        return y

    @staticmethod
    def _weights_quantization(w):
        """
        Quantizes the weights to 1-bit precision.

        This is equation (1) in the aforementioned paper.

        Also pre-undoes part of the scaling in (11) by multiplying the weights by the scale.

        Args:
            w: Array of weights.

        Returns:
            Quantized weights.
        """

        absolutes = ops.abs(w)
        scale = ops.mean(absolutes)
        alpha = ops.mean(w)

        signs = ops.sign(w - alpha)

        u = signs * scale
        return u
