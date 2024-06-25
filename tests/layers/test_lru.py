import os
import tempfile

import numpy as np
import pytest
from keras import backend, layers, models, ops

from keras_mml.layers._recurrent import LRUMML
from keras_mml.utils.array import as_numpy


def test_call():
    # Partial MML
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape(2, 3, 1)
    layer = LRUMML(4, 4)
    y = layer(x)
    assert ops.shape(y) == (2, 4)

    # Full MML
    layer = LRUMML(4, 4, fully_mml=True)
    y = layer(x)
    assert ops.shape(y) == (2, 4)

    # Masked, flat
    layer = LRUMML(4, 4)
    y = layer(x, mask=np.array([[True, True, False]]))
    assert ops.shape(y) == (2, 4)

    # Masked, nested
    layer = LRUMML(4, 4)
    y = layer(x, mask=[np.array([[True, True, False]])])
    assert ops.shape(y) == (2, 4)


def test_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape(2, 3, 1)

        # Check saving
        model_path = os.path.join(tmpdir, "test_save_gru_mml.keras")
        model1 = models.Sequential(layers=[layers.Input(shape=(3, 1)), LRUMML(4, 4), layers.Dense(2)])
        model1_output = as_numpy(model1(mock_data))

        model1.save(model_path)
        assert os.path.isfile(model_path)

        # Check loading
        backend.clear_session()
        model2 = models.load_model(model_path)
        model2_output = as_numpy(model2(mock_data))

        assert np.allclose(model1_output, model2_output)


def test_invalid_arguments():
    # Units invalid
    with pytest.raises(ValueError):
        LRUMML(0, 2)

    with pytest.raises(ValueError):
        LRUMML(-1, 2)

    # State dim invalid
    with pytest.raises(ValueError):
        LRUMML(2, 0)

    with pytest.raises(ValueError):
        LRUMML(2, -1)
