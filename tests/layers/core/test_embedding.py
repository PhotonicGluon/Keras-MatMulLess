import os
import tempfile

import numpy as np
import pytest
from einops import asnumpy
from keras import backend, layers, models, ops

from keras_mml.layers import TokenEmbedding
from keras_mml.layers.core.embedding import PatchEmbedding


# ----- TokenEmbedding() -----
# Calls
@pytest.fixture
def token_embedding_call_data():
    return np.array([[1, 2, 3], [0, 1, 2]])


def test_token_embedding_call_without_positions(token_embedding_call_data):
    layer = TokenEmbedding(max_len=5, vocab_size=4, embedding_dim=6)
    y = layer(token_embedding_call_data)
    assert ops.shape(y) == (2, 3, 6)


def test_token_embedding_call_with_positions(token_embedding_call_data):
    layer = TokenEmbedding(max_len=5, vocab_size=4, embedding_dim=6, with_positions=True)
    y = layer(token_embedding_call_data)
    assert ops.shape(y) == (2, 3, 6)


def test_token_embedding_invalid_max_len():
    with pytest.raises(ValueError):
        TokenEmbedding(max_len=0, vocab_size=2, embedding_dim=2)

    with pytest.raises(ValueError):
        TokenEmbedding(max_len=-1, vocab_size=2, embedding_dim=2)


def test_token_embedding_invalid_vocab_size():
    with pytest.raises(ValueError):
        TokenEmbedding(max_len=2, vocab_size=0, embedding_dim=2)

    with pytest.raises(ValueError):
        TokenEmbedding(max_len=2, vocab_size=-1, embedding_dim=2)


def test_token_embedding_invalid_embedding_dim():
    with pytest.raises(ValueError):
        TokenEmbedding(max_len=2, vocab_size=2, embedding_dim=0)

    with pytest.raises(ValueError):
        TokenEmbedding(max_len=2, vocab_size=2, embedding_dim=-1)


# Saving/loadings
def test_token_embedding_save_load():
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


# Training
def test_token_embedding_training():
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


# ----- PatchEmbedding() -----
# Calls
@pytest.fixture
def patch_embedding_call_data():
    return np.random.random((2, 16, 48))


def test_patch_embedding_call_without_positions(patch_embedding_call_data):
    layer = PatchEmbedding(num_patches=16, embedding_dim=4)
    y = layer(patch_embedding_call_data)
    assert ops.shape(y) == (2, 16, 4)


def test_patch_embedding_call_with_positions(patch_embedding_call_data):
    layer = PatchEmbedding(num_patches=16, embedding_dim=4, with_positions=True)
    y = layer(patch_embedding_call_data)
    assert ops.shape(y) == (2, 16, 4)


def test_patch_embedding_invalid_num_patches():
    with pytest.raises(ValueError):
        PatchEmbedding(num_patches=0, embedding_dim=2)

    with pytest.raises(ValueError):
        PatchEmbedding(num_patches=-1, embedding_dim=2)


def test_patch_embedding_invalid_embedding_dim():
    with pytest.raises(ValueError):
        PatchEmbedding(num_patches=2, embedding_dim=0)

    with pytest.raises(ValueError):
        PatchEmbedding(num_patches=2, embedding_dim=-1)


# Saving/loadings
def test_patch_embedding_save_load():
    mock_data = np.random.random((2, 16, 48))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Check saving
        model_path = os.path.join(tmpdir, "test_save_emb_mml.keras")
        model1 = models.Sequential(
            layers=[layers.Input(shape=(16, 48)), PatchEmbedding(num_patches=16, embedding_dim=4)]
        )
        model1_output = asnumpy(model1(mock_data))

        model1.save(model_path)
        assert os.path.isfile(model_path)

        # Check loading
        backend.clear_session()
        model2 = models.load_model(model_path)
        model2_output = asnumpy(model2(mock_data))

        assert np.allclose(model1_output, model2_output)


# Training
def test_patch_embedding_training():
    x = np.random.random((2, 16, 48))
    y = np.random.random((2, 16, 4))

    # Create the simple model
    model = models.Sequential()
    model.add(layers.Input((16, 48)))
    model.add(PatchEmbedding(num_patches=16, embedding_dim=4))

    # Fit the model
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, verbose=0)
