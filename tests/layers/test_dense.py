import numpy as np
from keras import ops

from keras_mml.layers.dense import DenseMML


def test_dense():
    x = np.array([[1., 2., 3.], [4., 5., 6.]])
    layer = DenseMML(4)
    y = layer(x)
    assert ops.shape(y) == (2, 4)
