"""
Quantized Root Mean Square Normalization (RMSNorm) implementation.
"""

import keras

from keras_mml.layers.normalizations._quant_rms_norm_impl import BackendQuantRMSNorm


@keras.saving.register_keras_serializable(package="keras_mml")
class QuantRMSNorm(BackendQuantRMSNorm):
    """
    Implements Root Mean Square Normalization (RMSNorm) with 8-bit quantization.

    The implementation of RMSNorm follows |RMSNorm Paper|_.

    See :py:class:`~keras_mml.layers.normalizations.RMSNorm` for the full documentation.

    .. |RMSNorm Paper| replace:: *Root Mean Square Layer Normalization*
    .. _RMSNorm Paper: https://arxiv.org/pdf/1910.07467v1
    """
