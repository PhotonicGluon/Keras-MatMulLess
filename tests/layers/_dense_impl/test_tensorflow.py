import os

import numpy as np
import pytest

try:
    import tensorflow as tf

    os.environ["KERAS_BACKEND"] = "tensorflow"
except ModuleNotFoundError:
    pytest.skip("Tensorflow not installed; skipping", allow_module_level=True)

import keras

if keras.config.backend() != "tensorflow":
    pytest.skip("Somehow Tensorflow was not selected as backend; skipping", allow_module_level=True)

from keras_mml.layers._dense_impl.tensorflow_dense import TensorflowDenseMML
from keras_mml.layers.dense import DenseMML
from keras_mml.utils.array import as_numpy
from tests.layers._dense_impl.setup import frontend_answer, weights


def test_check_implementation_consistency(weights):
    assert TensorflowDenseMML in DenseMML.__mro__

    backend_layer = TensorflowDenseMML()
    backend_answer = backend_layer._weights_quantization(tf.convert_to_tensor(weights))
    backend_answer = as_numpy(backend_answer)

    assert np.allclose(backend_answer, frontend_answer(weights, DenseMML))
