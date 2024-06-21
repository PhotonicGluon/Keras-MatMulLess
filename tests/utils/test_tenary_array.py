import numpy as np
import pytest

from keras_mml.utils.array import decode_ternary_array, encode_ternary_array


def test_encode_ternary_array():
    # Normal inputs
    x = np.array([[1, 1, 1], [-1, 0, 1], [0, 1, 0], [1, 1, -1]])
    shape, encoded = encode_ternary_array(x)
    assert shape == (4, 3)
    assert encoded == b"\xab\x49\x58"

    x = np.array([[1, 1, 1, -1], [0, 1, 0, 1], [0, 1, 1, -1]])
    shape, encoded = encode_ternary_array(x)
    assert shape == (3, 4)
    assert encoded == b"\xab\x49\x58"

    # Abnormal inputs
    with pytest.raises(ValueError):
        encode_ternary_array(np.array([1, 1, 1]))

    with pytest.raises(ValueError):
        encode_ternary_array(np.array([[[1, 0, 0]]]))


def test_decode_ternary_array():
    # Normal inputs
    x = decode_ternary_array((4, 3), b"\xab\x49\x58")
    assert np.array_equal(x, np.array([[1, 1, 1], [-1, 0, 1], [0, 1, 0], [1, 1, -1]]))

    x = decode_ternary_array((3, 4), b"\xab\x49\x58")
    assert np.array_equal(x, np.array([[1, 1, 1, -1], [0, 1, 0, 1], [0, 1, 1, -1]]))

    # Abnormal inputs
    with pytest.raises(ValueError):
        decode_ternary_array((1,), b"\xab\x49\x58")

    with pytest.raises(ValueError):
        decode_ternary_array((3, 4), b"")

    # Other tests
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
