import numpy as np

from keras_mml.layers.dense import DenseMML


def test_dense():
    x = np.array([[1., 2., 3.], [4., 5., 6.]])
    layer = DenseMML(4)
    print(layer(x))
    # print(layer._weights_quantization(x))
    assert 0
