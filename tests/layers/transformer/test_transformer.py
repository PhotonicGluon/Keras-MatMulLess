import os
import tempfile

import numpy as np
import pytest
from einops import asnumpy, rearrange
from keras import backend, layers, models, ops

from keras_mml.layers import TransformerBlockMML


def test_call():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = rearrange(x, "b (w e) -> b w e", e=2)

    layer = TransformerBlockMML(embedding_dim=2, ffn_dim=4, num_heads=2)
    y = layer(x)
    assert ops.shape(y) == (3, 1, 2)


def test_save_load():
    mock_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mock_data = rearrange(mock_data, "b (w e) -> b w e", e=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Check saving
        model_path = os.path.join(tmpdir, "test_save_att_mml.keras")
        model1 = models.Sequential(
            layers=[layers.Input(shape=(1, 2)), TransformerBlockMML(embedding_dim=2, ffn_dim=4, num_heads=2)]
        )
        model1_output = asnumpy(model1(mock_data))

        model1.save(model_path)
        assert os.path.isfile(model_path)

        # Check loading
        backend.clear_session()
        model2 = models.load_model(model_path)
        model2_output = asnumpy(model2(mock_data))

        assert np.allclose(model1_output, model2_output)


def test_invalid_arguments():
    # Embedding dimension invalid
    with pytest.raises(ValueError):
        TransformerBlockMML(embedding_dim=0, ffn_dim=2, num_heads=2)

    with pytest.raises(ValueError):
        TransformerBlockMML(embedding_dim=-1, ffn_dim=2, num_heads=2)

    # FFN dimension invalid
    with pytest.raises(ValueError):
        TransformerBlockMML(embedding_dim=2, ffn_dim=0, num_heads=2)

    with pytest.raises(ValueError):
        TransformerBlockMML(embedding_dim=2, ffn_dim=-1, num_heads=2)

    # Number of heads invalid
    with pytest.raises(ValueError):
        TransformerBlockMML(embedding_dim=2, ffn_dim=2, num_heads=0)

    with pytest.raises(ValueError):
        TransformerBlockMML(embedding_dim=2, ffn_dim=2, num_heads=-1)

    # Embedding dimension not divisible by number of heads
    with pytest.raises(ValueError):
        TransformerBlockMML(embedding_dim=3, ffn_dim=2, num_heads=2)


def test_training():
    # Dataset is just a sequence of known numbers
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = rearrange(x, "b (w e) -> b w e", e=2)
    y = np.copy(x)

    # Create the simple model
    model = models.Sequential()
    model.add(layers.Input((1, 2)))
    model.add(TransformerBlockMML(embedding_dim=2, ffn_dim=4, num_heads=2))
    model.add(layers.Dense(1))

    # Fit the model
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, verbose=0)
