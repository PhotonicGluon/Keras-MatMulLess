"""
Base class for the quantized Root Mean Square Normalization (RMSNorm) implementation.
"""

import keras

from keras_mml.layers.normalizations.rms_norm import RMSNorm

EPSILON = 1e-5
HUGE = 1e9


@keras.saving.register_keras_serializable(package="keras_mml")
class BaseQuantRMSNorm(RMSNorm):
    """
    Base class for the Root Mean Square Normalization (RMSNorm) with 8-bit quantization.
    """
