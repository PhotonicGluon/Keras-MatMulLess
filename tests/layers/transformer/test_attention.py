import os
import tempfile

import numpy as np
import pytest
from einops import asnumpy, rearrange
from keras import backend, layers, models, ops

from keras_mml.layers import AttentionMML


# Calls
def test_call():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = rearrange(x, "b (h w) -> b h w", w=1)

    layer = AttentionMML(num_heads=2, out_dim=4)
    y = layer(x)
    assert ops.shape(y) == (2, 3, 4)


def test_invalid_num_heads():
    with pytest.raises(ValueError):
        AttentionMML(num_heads=0, out_dim=2)

    with pytest.raises(ValueError):
        AttentionMML(num_heads=-1, out_dim=2)


def test_invalid_output_dim():
    with pytest.raises(ValueError):
        AttentionMML(num_heads=2, out_dim=0)

    with pytest.raises(ValueError):
        AttentionMML(num_heads=2, out_dim=-1)


# Saving/loading
def test_save_load():
    mock_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mock_data = rearrange(mock_data, "b (h w) -> b h w", w=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Check saving
        model_path = os.path.join(tmpdir, "test_save_att_mml.keras")
        model1 = models.Sequential(layers=[layers.Input(shape=(2, 1)), AttentionMML(num_heads=2, out_dim=4)])
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
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = np.copy(x)
    x = rearrange(x, "b (h w) -> b h w", w=1)

    # Create the simple model
    model = models.Sequential()
    model.add(layers.Input((3, 1)))
    model.add(AttentionMML(num_heads=2, out_dim=4))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(3))

    # Fit the model
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, verbose=0)
