"""
Handles the selection of the actual, backend-*dependent* ``DenseMML` class.
"""

import keras

from keras_mml.layers.dense.base import BaseDenseMML

BACKEND = keras.config.backend()

# TODO: Do backend selection
if BACKEND == "tensorflow":
    pass
elif BACKEND == "torch":
    pass
elif BACKEND == "jax":
    pass
else:
    raise ImportError(f"Invalid backend '{BACKEND}'")

DenseMML = BaseDenseMML  # TODO: Change according to backend
