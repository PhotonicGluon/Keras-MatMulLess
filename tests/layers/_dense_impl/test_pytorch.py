import os

import numpy as np
import pytest

try:
    import torch

    os.environ["KERAS_BACKEND"] = "torch"
except ModuleNotFoundError:
    pytest.skip("PyTorch not installed; skipping", allow_module_level=True)

import keras

if keras.config.backend() != "torch":
    pytest.skip("Somehow PyTorch was not selected as backend; skipping", allow_module_level=True)

from keras_mml.layers._dense_impl.torch_dense import TorchDenseMML
from keras_mml.layers.dense import DenseMML
from keras_mml.utils.array import as_numpy
from tests.layers._dense_impl.setup import frontend_answer, weights


def test_check_implementation_consistency(weights):
    assert TorchDenseMML in DenseMML.__mro__

    backend_layer = TorchDenseMML()
    backend_answer = backend_layer._weights_quantization(torch.tensor(weights))
    backend_answer = as_numpy(backend_answer)

    assert np.allclose(backend_answer, frontend_answer(weights, DenseMML))
