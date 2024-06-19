import numpy as np
import pytest

from keras_mml.utils import as_numpy


def test_tf():
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        pytest.skip("Tensorflow not installed")

    orig = np.array([1, 2, 3])
    converted = tf.convert_to_tensor(orig)
    reverted = as_numpy(converted)

    assert np.array_equal(orig, reverted)


def test_pytorch():
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("PyTorch not installed")

    orig = np.array([1, 2, 3])
    converted = torch.tensor(orig)
    reverted = as_numpy(converted)

    assert np.array_equal(orig, reverted)


def test_jax():
    try:
        import jax.numpy as jnp
    except ModuleNotFoundError:
        pytest.skip("Jax not installed")

    orig = np.array([1, 2, 3])
    converted = jnp.array(orig)
    reverted = as_numpy(converted)

    assert np.array_equal(orig, reverted)
