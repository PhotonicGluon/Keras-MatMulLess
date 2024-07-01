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
