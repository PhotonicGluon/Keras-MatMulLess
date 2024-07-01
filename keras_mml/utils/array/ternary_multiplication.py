"""
Implementation of ternary multiplication.
"""

from keras_mml.utils.array._ternary_multiplication_impl import backend_function


def ternary_multiplication(x_quantized, w_quantized, w_scale: float):
    """
    Applies the ternary multiplication algorithm.

    Args:
        x_quantized: Quantized activation values.
        w_quantized: Quantized kernel matrix without scaling applied.
        w_scale: Scale factor for the kernel matrix.

    Returns:
        Multiplied matrix.
    """

    return backend_function(x_quantized, w_quantized, w_scale)
