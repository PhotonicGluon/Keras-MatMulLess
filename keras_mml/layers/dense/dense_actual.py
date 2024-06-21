"""
Handles the selection of the actual, backend-*dependent* ``DenseMML`` class.
"""

import keras

BACKEND = keras.config.backend()

if BACKEND == "tensorflow":
    # TODO: Change according to backend
    from keras_mml.layers.dense.fallback_dense import FallbackDenseMML

    the_class = FallbackDenseMML
elif BACKEND == "torch":
    from keras_mml.layers.dense.torch_dense import TorchDenseMML

    the_class = TorchDenseMML
elif BACKEND == "jax":
    # TODO: Change according to backend
    from keras_mml.layers.dense.fallback_dense import FallbackDenseMML

    the_class = FallbackDenseMML
else:
    raise ImportError(f"Invalid backend '{BACKEND}'")


@keras.saving.register_keras_serializable(package="keras_mml")
class DenseMML(the_class):
    # TODO: Set up auto-documentation
    pass
