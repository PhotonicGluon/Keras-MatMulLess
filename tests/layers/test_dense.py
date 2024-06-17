import os
import tempfile

import numpy as np
from keras import backend, layers, models, ops

from keras_mml.layers.dense import DenseMML
from keras_mml.utils.array import as_numpy


def test_call():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer = DenseMML(4)
    y = layer(x)
    assert ops.shape(y) == (2, 4)


def test_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_data = np.array([[1.0, 2.0, 3.0]])

        # Check saving
        model_path = os.path.join(tmpdir, "test_save_dense_mml.keras")
        model1 = models.Sequential(layers=[layers.Input(shape=(3,)), DenseMML(8), DenseMML(16), layers.Dense(5)])
        model1_output = as_numpy(model1(mock_data))

        model1.save(model_path)
        assert os.path.isfile(model_path)

        # Check loading
        backend.clear_session()
        model2 = models.load_model(model_path)
        model2_output = as_numpy(model2(mock_data))

        assert np.allclose(model1_output, model2_output)
