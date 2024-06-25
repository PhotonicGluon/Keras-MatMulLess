import os
import tempfile

import numpy as np
import pytest
from keras import backend, layers, models, ops

from keras_mml.layers import SwiGLUMML
from keras_mml.utils.array import as_numpy


def test_fixed_intermediate_size():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer = SwiGLUMML(8, intermediate_size=16)
    y = layer(x)
    assert ops.shape(y) == (2, 8)
    assert layer.intermediate_size == 16


def test_dynamic_intermediate_size():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer = SwiGLUMML(8)
    y = layer(x)
    assert ops.shape(y) == (2, 8)
    assert layer.intermediate_size == 256


def test_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_data = np.array([[1.0, 2.0, 3.0]])

        # Check saving
        model_path = os.path.join(tmpdir, "test_save_swiglu_mml.keras")
        model1 = models.Sequential(layers=[layers.Input(shape=(3,)), SwiGLUMML(16), layers.Dense(5)])
        model1_output = as_numpy(model1(mock_data))

        model1.save(model_path)
        assert os.path.isfile(model_path)

        # Check loading
        backend.clear_session()
        model2 = models.load_model(model_path)
        model2_output = as_numpy(model2(mock_data))

        assert np.allclose(model1_output, model2_output)


def test_invalid_units():
    with pytest.raises(ValueError):
        SwiGLUMML(0)

    with pytest.raises(ValueError):
        SwiGLUMML(-1)
