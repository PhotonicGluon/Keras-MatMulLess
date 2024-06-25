import numpy as np
from keras import ops

from keras_mml.layers import BilinearMML, GeGLUMML, ReGLUMML, SeGLUMML, SwiGLUMML


def test_bilinear():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer = BilinearMML(8, intermediate_size=16)
    y = layer(x)
    assert ops.shape(y) == (2, 8)
    assert layer.intermediate_size == 16


def test_geglu():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer = GeGLUMML(8, intermediate_size=16)
    y = layer(x)
    assert ops.shape(y) == (2, 8)
    assert layer.intermediate_size == 16


def test_reglu():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer = ReGLUMML(8, intermediate_size=16)
    y = layer(x)
    assert ops.shape(y) == (2, 8)
    assert layer.intermediate_size == 16


def test_seglu():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer = SeGLUMML(8, intermediate_size=16)
    y = layer(x)
    assert ops.shape(y) == (2, 8)
    assert layer.intermediate_size == 16


def test_swiglu():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer = SwiGLUMML(8, intermediate_size=16)
    y = layer(x)
    assert ops.shape(y) == (2, 8)
    assert layer.intermediate_size == 16
