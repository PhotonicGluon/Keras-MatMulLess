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


# @tf.function(jit_compile=True)
# def _2d_ternary_multiplication(x: tf.Tensor, w: tf.Tensor, scale: float) -> tf.Tensor:
#     """
#     Performs ternary multiplication on a known 1D vector ``x`` and a 2D matrix ``w``.

#     Args:
#         x: 1D vector.
#         w: 2D quantized matrix.
#         scale: Scaling factor.

#     Returns:
#         A vector, which is the result of the ternary multiplication of ``x`` and ``w``.
#     """

#     # TODO: Optimize
#     assert w.ndim == 2
#     assert x.ndim == 1
#     return tf.tensordot(x, w / scale, 1)  # This should just involve addition and subtraction


# @tf.function(jit_compile=True)
# def tf_ternary_multiplication(x_quantized: tf.Tensor, w_quantized: tf.Tensor, w_scale: float) -> tf.Tensor:
#     """
#     Applies the ternary multiplication algorithm.

#     Args:
#         x_quantized: Quantized activation values.
#         w_quantized: Quantized kernel matrix without scaling applied.
#         w_scale: Scale factor for the kernel matrix.

#     Returns:
#         Multiplied matrix.
#     """

#     # If the weights matrix is 2D then we just apply the standard algorithm
#     if w_quantized.ndim == 2:
#         return _2d_ternary_multiplication(x_quantized, w_quantized, w_scale)

#     # Otherwise, we need to treat what we have as a stack of matrices.
#     # First we get the number of stacked matrices and vectors that we need to process
#     w_shape = w_quantized.shape
#     x_shape = x_quantized.shape

#     num_stacked_matrices = 1
#     num_stacked_vectors = 1
#     for i in range(w_quantized.ndim - 2):  # The last 2 indices are the matrices
#         num_stacked_matrices *= w_shape[i]
#         num_stacked_vectors *= x_shape[i]

#     # Identify the shape of the matrices and vectors that will actually be multiplied
#     matrix_shape = (w_shape[-2], w_shape[-1])
#     matrix_stride = w_shape[-2] * w_shape[-1]
#     vector_stride = x_shape[-1]

#     # Flatten the input arrays for easier processing
#     w_flat = tf.reshape(w_quantized, (-1,))
#     x_flat = tf.reshape(x_quantized, (-1,))

#     # Determine output size and shape
#     output_shape = list(w_shape[:-2]) + [x_shape[-2], w_shape[-1]]
#     output_size = 1
#     for dim in output_shape:
#         output_size *= dim
#     output_flat = tf.zeros((output_size,), dtype=x_quantized.dtype)
#     output_stride = w_shape[-1]

#     k = 0  # Output vector pointer
#     for i in range(num_stacked_matrices):
#         matrix_elements = tf.reshape(w_flat[i * matrix_stride : (i + 1) * matrix_stride], matrix_shape)

#         for j in range(num_stacked_vectors):
#             vector_elements = x_flat[j * vector_stride : (j + 1) * vector_stride]
#             product = _2d_ternary_multiplication(vector_elements, matrix_elements, w_scale)
#             output_flat = tf.tensor_scatter_nd_update(
#                 output_flat, tf.reshape(tf.range(k * output_stride, (k + 1) * output_stride), (-1, 1)), product
#             )
#             k += 1

#     return tf.reshape(output_flat, output_shape)
