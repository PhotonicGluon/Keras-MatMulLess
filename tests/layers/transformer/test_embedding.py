import os
import tempfile

import numpy as np
import pytest
from einops import asnumpy, rearrange
from keras import backend, layers, models, ops

from keras_mml.layers import TokenEmbedding


def test_call():
    x = np.array([[1, 2, 3], [0, 1, 2]])

    # Without positions
    layer = TokenEmbedding(max_len=5, vocab_size=4, embedding_dim=6)
    y = layer(x)
    assert ops.shape(y) == (2, 3, 6)

    # With positions
    layer = TokenEmbedding(max_len=5, vocab_size=4, embedding_dim=6, with_positions=True)
    y = layer(x)
    assert ops.shape(y) == (2, 3, 6)


def test_save_load():
    mock_data = np.array([[1, 2, 3], [0, 1, 2]])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Check saving
        model_path = os.path.join(tmpdir, "test_save_emb_mml.keras")
        model1 = models.Sequential(
            layers=[
                layers.Input(shape=(3,)),
                TokenEmbedding(max_len=5, vocab_size=4, embedding_dim=6, with_positions=True),
            ]
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
    # Max length invalid
    with pytest.raises(ValueError):
        TokenEmbedding(max_len=0, vocab_size=2, embedding_dim=2)

    with pytest.raises(ValueError):
        TokenEmbedding(max_len=-1, vocab_size=2, embedding_dim=2)

    # Vocab size invalid
    with pytest.raises(ValueError):
        TokenEmbedding(max_len=2, vocab_size=0, embedding_dim=2)

    with pytest.raises(ValueError):
        TokenEmbedding(max_len=2, vocab_size=-1, embedding_dim=2)

    # Embedding dimension invalid
    with pytest.raises(ValueError):
        TokenEmbedding(max_len=2, vocab_size=2, embedding_dim=0)

    with pytest.raises(ValueError):
        TokenEmbedding(max_len=2, vocab_size=2, embedding_dim=-1)


def test_training():
    # Dataset is just a sequence of known numbers
    x = np.array([[1, 2, 3], [0, 1, 2]])
    y = np.array([[[1, 1], [2, 2], [3, 3]], [[0, 0], [1, 1], [2, 2]]])

    # Create the simple model
    model = models.Sequential()
    model.add(layers.Input((3,)))
    model.add(TokenEmbedding(max_len=5, vocab_size=4, embedding_dim=2))

    # Fit the model
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, verbose=0)
