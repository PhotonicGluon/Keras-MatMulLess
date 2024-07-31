import os

import numpy as np
from einops import asnumpy, rearrange
from keras import layers, models

from keras_mml.layers import QuantRMSNorm


# Calls
def test_call():
    x = np.array([1, 2, 3, 4], dtype="float32")
    y = QuantRMSNorm(has_learnable_weights=False)(x)
    y_pred = asnumpy(y)
    y_true = np.array([0.36802354, 0.7360471, 1.09257, 1.4605935])
    assert np.allclose(y_pred, y_true)


# Training
def test_training():
    # Dataset is just a sequence of known numbers
    x = np.array([1, 2, 3, 4, 5], dtype="float32")
    x = rearrange(x, "(b f) -> b f", f=1)
    y = np.copy(x)

    # Create the simple model
    model = models.Sequential()
    model.add(layers.Input((1,)))
    model.add(layers.Dense(3))
    model.add(QuantRMSNorm())
    model.add(layers.Dense(1))

    # Fit the model
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, verbose=0)
