import os
import tempfile

import numpy as np
import pytest
from einops import asnumpy, rearrange
from keras import backend, layers, models, ops

from keras_mml.layers.recurrent import LRUMML


# Calls
@pytest.fixture
def call_data():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = rearrange(x, "b (h w) -> b h w", w=1)
    return x


def test_call_partial_mml(call_data):
    layer = LRUMML(4, 4)
    y = layer(call_data)
    assert ops.shape(y) == (2, 4)


def test_call_full_mml(call_data):
    layer = LRUMML(4, 4, fully_mml=True)
    y = layer(call_data)
    assert ops.shape(y) == (2, 4)


def test_call_masked_flat(call_data):
    layer = LRUMML(4, 4)
    y = layer(call_data, mask=np.array([[True, True, False]]))
    assert ops.shape(y) == (2, 4)


def test_call_masked_nested(call_data):
    layer = LRUMML(4, 4)
    y = layer(call_data, mask=[np.array([[True, True, False]])])
    assert ops.shape(y) == (2, 4)


def test_invalid_units():
    with pytest.raises(ValueError):
        LRUMML(0, 2)

    with pytest.raises(ValueError):
        LRUMML(-1, 2)


def test_invalid_state_dim():
    with pytest.raises(ValueError):
        LRUMML(2, 0)

    with pytest.raises(ValueError):
        LRUMML(2, -1)


# Saving/loading
def test_save_load():
    mock_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mock_data = rearrange(mock_data, "b (h w) -> b h w", w=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Check saving
        model_path = os.path.join(tmpdir, "test_save_lru_mml.keras")
        model1 = models.Sequential(layers=[layers.Input(shape=(3, 1)), LRUMML(4, 4), layers.Dense(2)])
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
    # Dataset is just a sequence of known numbers
    x = np.array([1, 2, 3, 4, 5])
    x = rearrange(x, "(b t f) -> b t f", t=1, f=1)
    y = rearrange(x, "b t f -> b (t f)")

    # Create the simple model
    model = models.Sequential()
    model.add(layers.Input((1, 1)))
    model.add(LRUMML(3, 3))
    model.add(layers.Dense(1))

    # Fit the model
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, verbose=0)
