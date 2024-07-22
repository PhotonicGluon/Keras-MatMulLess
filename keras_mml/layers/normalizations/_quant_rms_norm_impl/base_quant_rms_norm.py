"""
Base class for the quantized Root Mean Square Normalization (RMSNorm) implementation.
"""

import keras
import numpy as np
from jaxtyping import Float
from keras import ops

from keras_mml.layers.normalizations.rms_norm import RMSNorm

EPSILON = 1e-5
HUGE = 1e9


@keras.saving.register_keras_serializable(package="keras_mml")
class BaseQuantRMSNorm(RMSNorm):
    """
    Base class for the Root Mean Square Normalization (RMSNorm) with 8-bit quantization.
    """

    def call(self, inputs: Float[np.ndarray, "batch_size *dims"]) -> Float[np.ndarray, "batch_size *dims"]:
        # Process the normalization step first
        x_norm = super().call(inputs)

        # Generate the quantized activation values
        scale = 127.0 / ops.clip(ops.max(ops.abs(x_norm), axis=-1, keepdims=True), EPSILON, HUGE)
        x_quant = ops.clip(ops.round(x_norm * scale), -128, 127) / scale

        # Use Straight-Through Estimator (STE) trick by stopping gradient propagation
        return x_norm + ops.stop_gradient(x_quant - x_norm)
