"""
PyTorch-optimized version of generalized matrix-matrix ternary multiplication.
"""

import torch


def ternary_mul_nd(x_quantized: torch.Tensor, w_quantized: torch.Tensor, w_scale: float) -> torch.Tensor:
    """
    Applies the ternary multiplication algorithm.

    Args:
        x_quantized: Quantized activation values.
        w_quantized: Quantized kernel matrix without scaling applied.
        w_scale: Scale factor for the kernel matrix.
    Returns:
        Multiplied matrix.
    """

    return torch.matmul(x_quantized, w_quantized / w_scale)  # The `matmul` should just involve addition and subtraction
