"""
Backend-*dependent* selection of the implementation of the core algorithm in the matmul-less Dense
layer.
"""

import keras

BACKEND = keras.config.backend()

if BACKEND == "tensorflow":
    from keras_mml.layers.core._dense_impl.tensorflow_dense import TensorflowDenseMML

    backend_class = TensorflowDenseMML
elif BACKEND == "torch":
    # from keras_mml.layers.core._dense_impl.torch_dense import TorchDenseMML

    # backend_class = TorchDenseMML
    from keras_mml.layers.core._dense_impl.fallback_dense import FallbackDenseMML

    backend_class = FallbackDenseMML
elif BACKEND == "jax":
    from keras_mml.layers.core._dense_impl.jax_dense import JaxDenseMML

    backend_class = JaxDenseMML
else:
    from keras_mml.layers.core._dense_impl.fallback_dense import FallbackDenseMML

    backend_class = FallbackDenseMML


class BackendDenseMML(keras.Layer, backend_class):
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
        backend_class.__init__(self)
