import numpy as np
import pytest

from keras_mml.utils.numba import max_for_last_axis, max_in_interval


# max_in_interval()
def test_max_in_interval():
    # Regular calls
    assert max_in_interval(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 2, 6) == 6
    assert max_in_interval(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 0, 8) == 8
    assert max_in_interval(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 4, 5) == 5

    # Abnormal calls
    with pytest.raises(ValueError):
        max_in_interval(np.array([]), 0, 1)

    with pytest.raises(ValueError):
        max_in_interval(np.array([0, 1, 2, 3]), 1, 1)


# max_for_last_axis()
def max_for_last_axis_orig(a):
    return np.max(a, axis=-1)


def test_max_for_last_axis_1D():
    arrays = [
        np.array([1, 2, 3]),
        np.array([3, 2, 1]),
        np.array([1, 1, 1]),
        np.array([1.2, 2.3, 0.1]),
        np.array([5.5, 5.6, 7.7, 6.5, 12.4, 4.35, 6.45]),
    ]
    for array in arrays:
        assert np.allclose(max_for_last_axis(array), max_for_last_axis_orig(array))


def test_max_for_last_axis_2D():
    arrays = [
        np.array([[1, 2, 3], [3, 2, 1]]),
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        np.array([[1.2, 2.3, 0.1], [1.2, 2.3, 2.1], [1.2, 2.3, 5.1], [1.2, 3.3, -7.1]]),
        np.array([[5.5, 5.6, 7.7, 6.5, 12.4, 4.35, 6.45]]),
    ]
    for array in arrays:
        assert np.allclose(max_for_last_axis(array), max_for_last_axis_orig(array))


def test_max_for_last_axis_3D():
    arrays = [
        np.array([[[1], [2], [3]], [[3], [2], [1]]]),
        np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]),
        np.array([[[1.2, 2.3, 0.1], [1.2, 2.3, 2.1]], [[1.2, 2.3, 5.1], [1.2, 3.3, -7.1]]]),
        np.array([[[5.5, 5.6, 7.7, 6.5, 12.4, 4.35, 6.45]]]),
    ]
    for array in arrays:
        assert np.allclose(max_for_last_axis(array), max_for_last_axis_orig(array))
