"""
PyTorch-optimized implementation of ternary multiplication.
"""

import torch

from ternary_multiplication.torch.mul.mul_2d import ternary_mul_2d
from ternary_multiplication.torch.mul.mul_nd import ternary_mul_nd


def ternary_multiplication(x_quantized: torch.Tensor, w_quantized: torch.Tensor, w_scale: float) -> torch.Tensor:
    """
    Applies the ternary multiplication algorithm.

    Args:
        x_quantized: Quantized activation values.
        w_quantized: Quantized kernel matrix without scaling applied.
        w_scale: Scale factor for the kernel matrix.

    Returns:
        Multiplied matrix.
    """

    if x_quantized.ndim == w_quantized.ndim - 1:
        if w_quantized.ndim == 2:
            return ternary_mul_2d(x_quantized, w_quantized, w_scale)
        else:
            return ternary_mul_nd(x_quantized, w_quantized, w_scale)

    return torch.matmul(x_quantized, w_quantized / w_scale)  # The `matmul` should just involve addition and subtraction
