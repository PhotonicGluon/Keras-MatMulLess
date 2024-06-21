import numpy as np
import pytest

from keras_mml.utils.array import as_numpy


@pytest.fixture
def weights():
    return np.array([[1.2, 2.3, 3.4], [4.5, 5.6, 6.7]], dtype="float32")


def frontend_answer(weight_array, DenseMML):
    weight_shape = weight_array.shape

    frontend_layer = DenseMML(weight_shape[1])
    frontend_layer.build((None, weight_shape[0]))
    frontend_layer.w.assign(weight_array)
    frontend_weights_quantized, frontend_weights_scale = frontend_layer._quantized_weights_for_saving
    frontend_answer = frontend_weights_quantized / frontend_weights_scale
    frontend_answer = as_numpy(frontend_answer)

    return frontend_answer
