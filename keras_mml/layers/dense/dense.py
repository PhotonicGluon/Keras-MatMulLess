"""
Handles the selection of the actual, backend-*dependent* ``DenseMML` class.
"""

import keras

from keras_mml.layers.dense.base_dense import BaseDenseMML
from keras_mml.layers.dense.torch_dense import TorchDenseMML

BACKEND = keras.config.backend()

# TODO: Do backend selection
if BACKEND == "tensorflow":
    DenseMML = BaseDenseMML  # TODO: Change according to backend
elif BACKEND == "torch":
    DenseMML = TorchDenseMML
elif BACKEND == "jax":
    DenseMML = BaseDenseMML  # TODO: Change according to backend
else:
    raise ImportError(f"Invalid backend '{BACKEND}'")
