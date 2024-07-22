"""
Backend-*dependent* selection of the implementation for the quantized RMSNorm layer.
"""

import keras

BACKEND = keras.config.backend()

if BACKEND == "tensorflow":
    from keras_mml.layers.normalizations._quant_rms_norm_impl.tensorflow import TensorflowQuantRMSNorm

    BackendClass = TensorflowQuantRMSNorm
elif BACKEND == "torch":
    from keras_mml.layers.normalizations._quant_rms_norm_impl.torch import TorchQuantRMSNorm

    BackendClass = TorchQuantRMSNorm
elif BACKEND == "jax":
    from keras_mml.layers.normalizations._quant_rms_norm_impl.jax import JaxQuantRMSNorm

    BackendClass = JaxQuantRMSNorm
else:  # pragma: no cover
    raise ValueError(f"Invalid backend: {BACKEND}")


class BackendQuantRMSNorm(BackendClass):
    """
    Class that encapsulates the quantized RMSNorm.
    """

    backend = BACKEND
