"""
Tensorflow-optimized implementation of ternary multiplication.
"""

import tensorflow as tf


def tf_ternary_multiplication(x_quantized: tf.Tensor, w_quantized: tf.Tensor, w_scale: float) -> tf.Tensor:
    """
    Applies the ternary multiplication algorithm.

    Args:
        x_quantized: Quantized activation values.
        w_quantized: Quantized kernel matrix without scaling applied.
        w_scale: Scale factor for the kernel matrix.

    Returns:
        Multiplied matrix.
    """

    # TODO: Optimize
    return tf.matmul(x_quantized, w_quantized / w_scale)  # The `matmul` should just involve addition and subtraction
