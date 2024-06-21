"""
Backend-*dependent* selection of the implementation of the core algorithm in the matmul-less Dense
layer.
"""

import keras

BACKEND = keras.config.backend()

if BACKEND == "tensorflow":
    # TODO: Change according to backend
    from keras_mml.layers._dense_impl.fallback_dense import FallbackDenseMML

    backend_class = FallbackDenseMML
elif BACKEND == "torch":
    from keras_mml.layers._dense_impl.torch_dense import TorchDenseMML

    backend_class = TorchDenseMML
elif BACKEND == "jax":
    # TODO: Change according to backend
    from keras_mml.layers._dense_impl.fallback_dense import FallbackDenseMML

    backend_class = FallbackDenseMML
else:
    raise ImportError(f"Invalid backend '{BACKEND}'")


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
