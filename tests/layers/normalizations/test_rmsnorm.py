import numpy as np
from keras import ops

from keras_mml.layers import RMSNorm
from keras_mml.utils import as_numpy


def test_no_learnable_weights():
    x = np.array([1, 2, 3])
    y = RMSNorm(has_learnable_weights=False)(x)
    y_pred = as_numpy(y)
    y_true = np.array([0.46291004, 0.92582010, 1.38873015])
    assert np.allclose(y_pred, y_true)

    x = np.array([4, -5, 6, -7])
    y = RMSNorm(has_learnable_weights=False)(x)
    y_pred = as_numpy(y)
    y_true = [0.71269665, -0.89087081, 1.06904497, -1.24721913]
    assert np.allclose(y_pred, y_true)


def test_with_learnable_weights():
    # 1D array
    x = np.array([1, 2, 3])
    y = RMSNorm(use_bias=False)(x)
    assert ops.shape(y) == ops.shape(x)

    y = RMSNorm(use_bias=True)(x)
    assert ops.shape(y) == ops.shape(x)

    # 2D array
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = RMSNorm(use_bias=False)(x)
    assert ops.shape(y) == ops.shape(x)

    y = RMSNorm(use_bias=True)(x)
    assert ops.shape(y) == ops.shape(x)

    # 3D array
    x = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
    y = RMSNorm(use_bias=False)(x)
    assert ops.shape(y) == ops.shape(x)

    y = RMSNorm(use_bias=True)(x)
    assert ops.shape(y) == ops.shape(x)
