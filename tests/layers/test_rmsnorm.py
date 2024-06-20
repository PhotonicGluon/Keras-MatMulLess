import numpy as np
import pytest

from keras_mml.layers.rms_norm import RMSNorm
from keras_mml.utils import as_numpy


def test_rms_norm():
    x = np.array([1, 2, 3])
    y = RMSNorm(2)(x)
    y_pred = as_numpy(y)
    y_true = np.array([0.18898223, 0.37796447, 0.5669467])
    assert np.allclose(y_pred, y_true)

    x = np.array([4, 5, 6, 7])
    y = RMSNorm(5)(x)
    y_pred = as_numpy(y)
    y_true = [0.15936382, 0.19920477, 0.23904571, 0.27888668]
    assert np.allclose(y_pred, y_true)

    with pytest.raises(ValueError):
        RMSNorm(0)

    with pytest.raises(ValueError):
        RMSNorm(-1)
