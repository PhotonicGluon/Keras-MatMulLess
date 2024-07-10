"""
PyTorch-optimized implementation of ternary multiplication.
"""

import torch


def torch_ternary_multiplication(x_quantized: torch.Tensor, w_quantized: torch.Tensor, w_scale: float) -> torch.Tensor:
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
    return torch.matmul(x_quantized, w_quantized / w_scale)  # The `matmul` should just involve addition and subtraction


# @torch_compile(mode="reduce-overhead")
# def _2d_ternary_multiplication(x: torch.Tensor, w: torch.Tensor, scale: float) -> torch.Tensor:
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
#     return torch.tensordot(x, w / scale, 1)  # This should just involve addition and subtraction


# @torch_compile(mode="reduce-overhead")
# def torch_ternary_multiplication(x_quantized: torch.Tensor, w_quantized: torch.Tensor, w_scale: float) -> torch.Tensor:
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
#     w_flat = w_quantized.flatten()
#     x_flat = x_quantized.flatten()

#     # Determine output size and shape
#     output_shape = list(w_shape[:-2]) + [x_shape[-2], w_shape[-1]]
#     output_size = 1
#     for dim in output_shape:
#         output_size *= dim
#     output_flat = torch.zeros((output_size,))
#     output_stride = w_shape[-1]

#     k = 0  # Output vector pointer
#     for i in range(num_stacked_matrices):
#         matrix_elements = w_flat[i * matrix_stride : (i + 1) * matrix_stride].reshape(matrix_shape)

#         for j in range(num_stacked_vectors):
#             vector_elements = x_flat[j * vector_stride : (j + 1) * vector_stride]
#             product = _2d_ternary_multiplication(vector_elements, matrix_elements, w_scale)
#             output_flat[k * output_stride : (k + 1) * output_stride] = product
#             k += 1

#     return output_flat.reshape(output_shape)
