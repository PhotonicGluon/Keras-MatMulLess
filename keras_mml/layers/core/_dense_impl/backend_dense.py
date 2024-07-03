"""
Backend-*dependent* selection of the implementation of the core algorithm in the matmul-less Dense
layer.
"""

import keras

BACKEND = keras.config.backend()

if BACKEND == "tensorflow":
    from keras_mml.layers.core._dense_impl.tensorflow_dense import TensorflowDenseMML

    BackendClass = TensorflowDenseMML
elif BACKEND == "torch":
    from keras_mml.layers.core._dense_impl.torch_dense import TorchDenseMML

    BackendClass = TorchDenseMML
elif BACKEND == "jax":
    from keras_mml.layers.core._dense_impl.jax_dense import JaxDenseMML

    BackendClass = JaxDenseMML
else:  # pragma: no cover
    raise ValueError(f"Invalid backend: {BACKEND}")


class BackendDenseMML(keras.Layer, BackendClass):
    """
    Class that encapsulates the core algorithm for matmul-less dense.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            *args: Arguments for :py:class:`keras.Layer`.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.
        """

        keras.Layer.__init__(self, *args, **kwargs)
        BackendClass.__init__(self)
