import numpy as np
import pytest


def test_simple_rms_norm():
    from keras_mml.layers.misc.rms_norm import RMSNorm

    x = np.array([1, 2, 3])
    y = RMSNorm(2)(x)
    y_pred = y.numpy()
    y_true = [0.18898223, 0.37796447, 0.5669467]

    for i in range(len(y_pred)):
        assert y_pred[i] == pytest.approx(y_true[i], 1e-5)

    x = np.array([4, 5, 6, 7])
    y = RMSNorm(5)(x)
    y_pred = y.numpy()
    print(y_pred)
    y_true = [0.15936382, 0.19920477, 0.23904571, 0.27888668]

    for i in range(len(y_pred)):
        assert y_pred[i] == pytest.approx(y_true[i], 1e-5)
