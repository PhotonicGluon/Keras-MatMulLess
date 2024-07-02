"""
Implementation of ternary multiplication.
"""

from keras import ops

# from keras_mml.utils.array._ternary_multiplication_impl import backend_function


def ternary_multiplication(x_quantized, w_quantized, w_scale: float):
    """
    Applies the ternary multiplication algorithm.

    .. IMPORTANT::
        The current implementation of the ternary multiplication still falls back to ``ops.matmul``
        in Keras, since the backend-dependent implementations seem to be slower than this baseline
        function. This would be changed in a future version.

    Args:
        x_quantized: Quantized activation values.
        w_quantized: Quantized kernel matrix without scaling applied.
        w_scale: Scale factor for the kernel matrix.

    Returns:
        Multiplied matrix.
    """

    # TODO: Optimize backend-dependent ternary multiplication
    # return backend_function(x_quantized, w_quantized, w_scale)

    return ops.matmul(x_quantized, w_quantized) / w_scale
