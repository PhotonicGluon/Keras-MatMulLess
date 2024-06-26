import os
import tempfile

import numpy as np
import pytest
from einops import asnumpy
from keras import backend, layers, models, ops

from keras_mml.layers.recurrent import GRUMML


def test_call():
    # Partial MML
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape(2, 3, 1)
    layer = GRUMML(4)
    y = layer(x)
    assert ops.shape(y) == (2, 4)

    # Full MML
    layer = GRUMML(4, fully_mml=True)
    y = layer(x)
    assert ops.shape(y) == (2, 4)

    # Masked, flat
    layer = GRUMML(4)
    y = layer(x, mask=np.array([[True, True, False]]))
    assert ops.shape(y) == (2, 4)

    # Masked, nested
    layer = GRUMML(4)
    y = layer(x, mask=[np.array([[True, True, False]])])
    assert ops.shape(y) == (2, 4)


def test_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).reshape(2, 3, 1)

        # Check saving
        model_path = os.path.join(tmpdir, "test_save_gru_mml.keras")
        model1 = models.Sequential(layers=[layers.Input(shape=(3, 1)), GRUMML(4), layers.Dense(2)])
        model1_output = asnumpy(model1(mock_data))

        model1.save(model_path)
        assert os.path.isfile(model_path)

        # Check loading
        backend.clear_session()
        model2 = models.load_model(model_path)
        model2_output = asnumpy(model2(mock_data))

        assert np.allclose(model1_output, model2_output)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        GRUMML(0)

    with pytest.raises(ValueError):
        GRUMML(-1)
