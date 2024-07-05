import os
import tempfile

import numpy as np
import pytest
from einops import asnumpy
from keras import backend, layers, models, ops

from keras_mml.layers.misc.patches import Patches


# Calls
def test_call():
    x = np.random.random((2, 16, 16, 3))
    layer = Patches(4)
    y = layer(x)
    assert ops.shape(y) == (2, 16, 48)


def test_call_weird_dims():
    x = np.random.random((2, 15, 15, 3))
    layer = Patches(4)
    y = layer(x)
    assert ops.shape(y) == (2, 9, 48)


def test_invalid_patch_size():
    with pytest.raises(ValueError):
        Patches(0)

    with pytest.raises(ValueError):
        Patches(-1)


# Saving/Loading
def test_save_load():
    mock_data = np.random.random((2, 16, 16, 3))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Check saving
        model_path = os.path.join(tmpdir, "test_save_patches.keras")
        model1 = models.Sequential(layers=[layers.Input(shape=(16, 16, 3)), Patches(4)])
        model1_output = asnumpy(model1(mock_data))

        model1.save(model_path)
        assert os.path.isfile(model_path)

        # Check loading
        backend.clear_session()
        model2 = models.load_model(model_path)
        model2_output = asnumpy(model2(mock_data))

        assert np.allclose(model1_output, model2_output)


# Training
def test_training():
    x = np.random.random((2, 16, 16, 3))
    y = np.random.random((2, 16, 10))

    # Create the simple model
    model = models.Sequential()
    model.add(layers.Input((16, 16, 3)))
    model.add(Patches(4))
    model.add(layers.Dense(10))

    # Fit the model
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, verbose=0)
