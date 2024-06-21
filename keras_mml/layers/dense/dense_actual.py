"""
Handles the selection of the actual, backend-*dependent* ``DenseMML` class.
"""

import keras

from keras_mml.layers.dense.base_dense import BaseDenseMML

BACKEND = keras.config.backend()

# TODO: Do backend selection
if BACKEND == "tensorflow":
    the_class = BaseDenseMML  # TODO: Change according to backend
elif BACKEND == "torch":
    from keras_mml.layers.dense.torch_dense import TorchDenseMML

    the_class = TorchDenseMML
elif BACKEND == "jax":
    the_class = BaseDenseMML  # TODO: Change according to backend
else:
    raise ImportError(f"Invalid backend '{BACKEND}'")


@keras.saving.register_keras_serializable(package="keras_mml")
class DenseMML(the_class):
    # TODO: Set up auto-documentation
    pass
