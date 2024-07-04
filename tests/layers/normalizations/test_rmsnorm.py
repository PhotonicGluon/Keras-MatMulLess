import numpy as np
from einops import asnumpy, rearrange
from keras import layers, models, ops

from keras_mml.layers import RMSNorm


# Calls
def test_no_learnable_weights_1():
    x = np.array([1, 2, 3])
    y = RMSNorm(has_learnable_weights=False)(x)
    y_pred = asnumpy(y)
    y_true = np.array([0.46291004, 0.92582010, 1.38873015])
    assert np.allclose(y_pred, y_true)


def test_no_learnable_weights_2():
    x = np.array([4, -5, 6, -7])
    y = RMSNorm(has_learnable_weights=False)(x)
    y_pred = asnumpy(y)
    y_true = [0.71269665, -0.89087081, 1.06904497, -1.24721913]
    assert np.allclose(y_pred, y_true)


def test_with_learnable_weights_1d():
    x = np.array([1, 2, 3])
    y = RMSNorm(use_bias=False)(x)
    assert ops.shape(y) == ops.shape(x)

    y = RMSNorm(use_bias=True)(x)
    assert ops.shape(y) == ops.shape(x)


def test_with_learnable_weights_2d():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = RMSNorm(use_bias=False)(x)
    assert ops.shape(y) == ops.shape(x)

    y = RMSNorm(use_bias=True)(x)
    assert ops.shape(y) == ops.shape(x)


def test_with_learnable_weights_3d():
    x = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    y = RMSNorm(use_bias=False)(x)
    assert ops.shape(y) == ops.shape(x)

    y = RMSNorm(use_bias=True)(x)
    assert ops.shape(y) == ops.shape(x)


# Training
def test_training():
    # Dataset is just a sequence of known numbers
    x = np.array([1, 2, 3, 4, 5])
    x = rearrange(x, "(b f) -> b f", f=1)
    y = np.copy(x)

    # Create the simple model
    model = models.Sequential()
    model.add(layers.Input((1,)))
    model.add(layers.Dense(3))
    model.add(RMSNorm())
    model.add(layers.Dense(1))

    # Fit the model
    model.compile(loss="mse", optimizer="adam")
    model.fit(x, y, verbose=0)
