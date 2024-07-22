"""
Jax implementation for the quantized RMSNorm layer.
"""

import keras

from keras_mml.layers.normalizations._quant_rms_norm_impl.base_quant_rms_norm import BaseQuantRMSNorm


@keras.saving.register_keras_serializable(package="keras_mml")
class JaxQuantRMSNorm(BaseQuantRMSNorm):
    """
    Jax implementation of Root Mean Square Normalization (RMSNorm) with 8-bit quantization.
    """
