import numpy as np

from keras_mml.utils.array.ternary_multiplication import ternary_multiplication


def test_2d_multiplication():
    # Case 1
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([[-1.0, 0, 1.0], [0, 1.0, 0], [1.0, 0, -1.0]]).T
    scale = 2.0

    predicted = ternary_multiplication(x, w, scale)
    correct = np.array([1.0, 1.0, -1.0])

    assert np.allclose(predicted, correct)

    # Case 2
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.array([[-1.0, 0, 1.0, 0, 1.0], [0, 1.0, 0, -1.0, 0], [1.0, 0, -1.0, -1.0, 1.0]]).T
    scale = 3.0

    predicted = ternary_multiplication(x, w, scale)
    correct = np.array([7.0 / 3, -2.0 / 3, -1.0 / 3])

    assert np.allclose(predicted, correct)


def test_3d_multiplication():
    # Case 1
    x = np.array([[1.0, 2.0, 3.0], [1.0, 3.0, 6.0]])
    w = np.array([np.array([[-1.0, 0, 1.0], [0, 1.0, 0], [1.0, 0, -1.0]]).T])
    scale = 2.0

    predicted = ternary_multiplication(x, w, scale)
    correct = np.array([[[1.0, 1.0, -1.0], [2.5, 1.5, -2.5]]])

    assert np.allclose(predicted, correct)

    # Case 2
    x = np.array([[1.0, 2.0, 3.0], [1.0, 3.0, 6.0]])
    w = np.array(
        [
            np.array([[-1.0, 0, 1.0], [0, 1.0, 0], [1.0, 0, -1.0]]).T,
            np.array([[1.0, 0, -1.0], [0, -1.0, 0], [-1.0, 0, 1.0]]).T,
        ]
    )
    scale = 2.0

    predicted = ternary_multiplication(x, w, scale)
    correct = np.array([[[1.0, 1.0, -1.0], [2.5, 1.5, -2.5]], [[-1.0, -1.0, 1.0], [-2.5, -1.5, 2.5]]])

    assert np.allclose(predicted, correct)

    # Case 3
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    w = np.array([np.array([[-1.0, 0, 1.0, 0, 1.0], [0, 1.0, 0, -1.0, 0], [1.0, 0, -1.0, -1.0, 1.0]]).T])
    scale = 3.0

    predicted = ternary_multiplication(x, w, scale)
    correct = np.array([[[7.0 / 3, -2.0 / 3, -1.0 / 3]]])
    assert np.allclose(predicted, correct)

    # Case 4
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    w = np.array(
        [
            np.array([[-1.0, 0, 1.0, 0, 1.0], [0, 1.0, 0, -1.0, 0], [1.0, 0, -1.0, -1.0, 1.0]]).T,
            np.array([[-1.0, 0, 0, 0, 0], [0, -1.0, 0, 0, 0], [0, 0, -1.0, 0, 0]]).T,
        ]
    )
    scale = 3.0

    predicted = ternary_multiplication(x, w, scale)
    correct = np.array([[[7.0 / 3, -2.0 / 3, -1.0 / 3]], [[-1.0 / 3, -2.0 / 3, -1.0]]])
    assert np.allclose(predicted, correct)

    # Case 5
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, 0.0, 1.0, -2.0, 2.0]])
    w = np.array(
        [
            np.array([[-1.0, 0, 1.0, 0, 1.0], [0, 1.0, 0, -1.0, 0], [1.0, 0, -1.0, -1.0, 1.0]]).T,
            np.array([[-1.0, 0, 0, 0, 0], [0, -1.0, 0, 0, 0], [0, 0, -1.0, 0, 0]]).T,
        ]
    )
    scale = 3.0

    predicted = ternary_multiplication(x, w, scale)
    correct = np.array(
        [
            [[7.0 / 3, -2.0 / 3, -1.0 / 3], [4.0 / 3, 2.0 / 3, 2.0 / 3]],
            [[-1.0 / 3, -2.0 / 3, -1.0], [1.0 / 3, 0, -1.0 / 3]],
        ]
    )
    assert np.allclose(predicted, correct)
