import numpy as np
import pytest

from keras_mml.utils.array.encoding import decode_ternary_array, encode_ternary_array


def test_encode_ternary_array():
    x = np.array([[1, 1, 1], [-1, 0, 1], [0, 1, 0], [1, 1, -1]])
    shape, encoded = encode_ternary_array(x)
    assert shape == (4, 3)
    assert encoded == b"\x83\x920"

    x = np.array([[1, 1, 1, -1], [0, 1, 0, 1], [0, 1, 1, -1]])
    shape, encoded = encode_ternary_array(x)
    assert shape == (3, 4)
    assert encoded == b"\x83\x920"


def test_decode_ternary_array():
    # Normal inputs
    x = decode_ternary_array((4, 3), b"\x83\x920")
    assert np.array_equal(x, np.array([[1, 1, 1], [-1, 0, 1], [0, 1, 0], [1, 1, -1]]))

    x = decode_ternary_array((3, 4), b"\x83\x920")
    assert np.array_equal(x, np.array([[1, 1, 1, -1], [0, 1, 0, 1], [0, 1, 1, -1]]))

    # Abnormal inputs
    with pytest.raises(ValueError):
        decode_ternary_array((3, 4), b"")


def test_mix_of_both_encode_and_decode():
    x = np.array([[0, 0, 0, 0]])
    assert np.array_equal(x, decode_ternary_array(*encode_ternary_array(x)))

    x = np.array([[1, 1, 1, 1]])
    assert np.array_equal(x, decode_ternary_array(*encode_ternary_array(x)))

    x = np.array([[-1, -1, -1, -1]])
    assert np.array_equal(x, decode_ternary_array(*encode_ternary_array(x)))

    x = np.array([[0, 1, 1, 1, 1]])
    assert np.array_equal(x, decode_ternary_array(*encode_ternary_array(x)))

    x = np.array([[0, -1, -1, -1, -1]])
    assert np.array_equal(x, decode_ternary_array(*encode_ternary_array(x)))
