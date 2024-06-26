import os
import tempfile

import numpy as np
import pytest
from einops import asnumpy, rearrange
from keras import backend, layers, models, ops

from keras_mml.layers import GLUMML


def test_fixed_intermediate_size():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer = GLUMML(8, intermediate_size=16)
    y = layer(x)
    assert ops.shape(y) == (2, 8)
    assert layer.intermediate_size == 16


def test_dynamic_intermediate_size():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer = GLUMML(8)
    y = layer(x)
    assert ops.shape(y) == (2, 8)
    assert layer.intermediate_size == 256


def test_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_data = np.array([[1.0, 2.0, 3.0]])

        # Check saving
        model_path = os.path.join(tmpdir, "test_save_glu_mml.keras")
        model1 = models.Sequential(layers=[layers.Input(shape=(3,)), GLUMML(16), layers.Dense(5)])
        model1_output = asnumpy(model1(mock_data))

        model1.save(model_path)
        assert os.path.isfile(model_path)

        # Check loading
        backend.clear_session()
        model2 = models.load_model(model_path)
        model2_output = asnumpy(model2(mock_data))

        assert np.allclose(model1_output, model2_output)


def test_invalid_activation():
    with pytest.raises(ValueError):
        GLUMML(8, activation="fake")


def test_invalid_units():
    with pytest.raises(ValueError):
        GLUMML(0)

    with pytest.raises(ValueError):
        GLUMML(-1)


def test_training():
    # Dataset is just a sequence of known numbers
    x = np.array([1, 2, 3, 4, 5])
    x = rearrange(x, "(b f) -> b f", f=1)
    y = np.copy(x)

    # Create the simple model
    model = models.Sequential()
    model.add(layers.Input((1,)))
    model.add(GLUMML(3))
    model.add(layers.Dense(1))

    # Fit the model
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, verbose=0)
