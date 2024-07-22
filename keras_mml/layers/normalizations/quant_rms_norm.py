"""
Quantized Root Mean Square Normalization (RMSNorm) implementation.
"""

import keras
import numpy as np
from jaxtyping import Float
from keras import ops

from keras_mml.layers.normalizations.rms_norm import RMSNorm

EPSILON = 1e-5
HUGE = 1e9


@keras.saving.register_keras_serializable(package="keras_mml")
class QuantRMSNorm(RMSNorm):
    """
    Implements Root Mean Square Normalization (RMSNorm) with 8-bit quantization.

    The implementation of RMSNorm follows |RMSNorm Paper|_.

    See :py:class:`~keras_mml.layers.normalizations.RMSNorm` for the full documentation.

    .. |RMSNorm Paper| replace:: *Root Mean Square Layer Normalization*
    .. _RMSNorm Paper: https://arxiv.org/pdf/1910.07467v1
    """

    def call(self, inputs: Float[np.ndarray, "batch_size *dims"]) -> Float[np.ndarray, "batch_size *dims"]:
        x = super().call(inputs)

        scale = 127.0 / ops.clip(ops.max(ops.abs(x), axis=-1, keepdims=True), EPSILON, HUGE)
        y = ops.clip(ops.round(x * scale), -128, 127) / scale
        return y
