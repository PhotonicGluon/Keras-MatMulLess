import os
import tempfile

import numpy as np
import pytest
from einops import asnumpy
from keras import backend, layers, models, ops

from keras_mml.layers import DenseMML


def test_call():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer = DenseMML(4)
    y = layer(x)
    assert ops.shape(y) == (2, 4)

    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]])
    layer = DenseMML(4)
    y = layer(x)
    assert ops.shape(y) == (3, 2, 4)


def test_save_load():
    # Normal, 2D array
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_data = np.array([[1.0, 2.0, 3.0]])

        # Check saving
        model_path = os.path.join(tmpdir, "test_save_dense_mml.keras")
        model1 = models.Sequential(layers=[layers.Input(shape=(3,)), DenseMML(8), DenseMML(16), layers.Dense(5)])
        model1_output = asnumpy(model1(mock_data))

        model1.save(model_path)
        assert os.path.isfile(model_path)

        # Check loading
        backend.clear_session()
        model2 = models.load_model(model_path)
        model2_output = asnumpy(model2(mock_data))

        assert np.allclose(model1_output, model2_output)

    # 3D array
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]])

        # Check saving
        model_path = os.path.join(tmpdir, "test_save_dense_mml.keras")
        model1 = models.Sequential(layers=[layers.Input(shape=(2, 2)), DenseMML(8), layers.Dense(5)])
        model1_output = asnumpy(model1(mock_data))

        model1.save(model_path)
        assert os.path.isfile(model_path)

        # Check loading
        backend.clear_session()
        model2 = models.load_model(model_path)
        model2_output = asnumpy(model2(mock_data))

        assert np.allclose(model1_output, model2_output)

    # 2D array, but no bias
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_data = np.array([[1.0, 2.0, 3.0]])

        # Check saving
        model_path = os.path.join(tmpdir, "test_save_dense_mml.keras")
        model1 = models.Sequential(
            layers=[layers.Input(shape=(3,)), DenseMML(8, use_bias=False), DenseMML(16), layers.Dense(5)]
        )
        model1_output = asnumpy(model1(mock_data))

        model1.save(model_path)
        assert os.path.isfile(model_path)

        # Check loading
        backend.clear_session()
        model2 = models.load_model(model_path)
        model2_output = asnumpy(model2(mock_data))

        assert np.allclose(model1_output, model2_output)


def test_invalid_units():
    with pytest.raises(ValueError):
        DenseMML(0)

    with pytest.raises(ValueError):
        DenseMML(-1)
