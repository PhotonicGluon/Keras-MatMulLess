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


def test_hidden_calls():
    layer = DenseMML(4)

    # _activations_quantization()
    x = np.array([-1.23, 0.12, 4.56, 12.35, 34.324])
    predicted = layer._activations_quantization(ops.array(x))
    predicted = asnumpy(predicted)
    correct = np.array([-1.3513, 0.0000, 4.5946, 12.4323, 34.3240])
    assert np.allclose(predicted, correct, rtol=1e-3)

    # _compute_kernel_scale()
    x = np.array([[-1.23, 0.12, 4.56, 2.3, -5.34]])
    predicted = layer._compute_kernel_scale(ops.array(x))
    predicted = asnumpy(predicted)
    correct = 0.36900369
    assert np.isclose(predicted, correct)

    # _kernel_quantization_for_training()
    x = np.array([[-1.23, 0.12, 4.56, 2.3, -5.34]])
    predicted = layer._kernel_quantization_for_training(ops.array(x))
    predicted = asnumpy(predicted)
    correct = np.array([[-0.0, 0.0, 2.71, 2.71, -2.71]])
    assert np.allclose(predicted, correct)
