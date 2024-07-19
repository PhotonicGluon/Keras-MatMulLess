"""
PyTorch implementation of the core algorithm in the matmul-less Dense layer.
"""

from typing import Tuple

import torch

from keras_mml.layers.core._dense_impl.base_dense import EPSILON, BaseDenseMML
from keras_mml.utils.misc.coverage import torch_compile


@torch_compile(mode="reduce-overhead")
def _activations_quantization(x: torch.Tensor) -> torch.Tensor:
    """
    Quantizes the activations to 8-bit precision using absmax quantization.

    Args:
        x: Array of quantization values.

    Returns:
        The quantized activation values.
    """

    scale = 127.0 / torch.unsqueeze(torch.max(torch.abs(x), dim=-1).values.clamp_(EPSILON), -1)
    y = torch.clip(torch.round(x * scale), -128, 127) / scale
    return y


@torch_compile(mode="reduce-overhead")
def _compute_kernel_scale(w: torch.Tensor) -> float:
    """
    Computes the scale factor of the kernel matrix.

    Args:
        w: Kernel matrix.

    Returns:
        Scale factor.
    """

    return 1.0 / torch.mean(torch.abs(w)).clamp_(EPSILON)


@torch_compile(mode="reduce-overhead")
def _quantize_kernel(w: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Quantizes the kernel values to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

    Args:
        w: Kernel matrix.
        scale: Scaling factor.

    Returns:
        The quantized kernel without scaling applied.
    """

    return torch.clip(torch.round(w * scale), -1, 1)


@torch_compile(mode="reduce-overhead")
def _get_x_quantized(x_norm: torch.Tensor) -> torch.Tensor:
    """
    Gets the quantized activations, with support for the backward direction by using STE gradient
    bypass.

    We use a Straight-Through Estimator (STE) trick by stopping gradient propagation.

    Args:
        x_norm: Normalized activation values.

    Returns:
        Quantized activation values.
    """

    return x_norm + (_activations_quantization(x_norm) - x_norm).detach()


@torch_compile(mode="reduce-overhead")
def _get_w_quantized(w: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Gets the quantized kernel matrix, with support for the backward direction by using STE gradient
    bypass.

    We use a Straight-Through Estimator (STE) trick by stopping gradient propagation.

    Args:
        w: Kernel matrix.
        scale: Scaling factor.

    Returns:
        Quantized kernel matrix without scaling applied.
    """

    return w + (_quantize_kernel(w, scale) - w).detach()


class TorchDenseMML(BaseDenseMML):
    """
    Implementation of the Dense layer using the PyTorch backend.
    """

    @staticmethod
    def _compute_kernel_scale(w: torch.Tensor) -> float:
        return _compute_kernel_scale(w)

    @staticmethod
    def _quantize_kernel(w: torch.Tensor, scale: float) -> torch.Tensor:
        return _quantize_kernel(w, scale)

    def _get_quantized_arrays(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._kernel_scale:
            return _get_x_quantized(x_norm), self._kernel.value, self._kernel_scale

        scale = _compute_kernel_scale(self._kernel.value)
        return _get_x_quantized(x_norm), _get_w_quantized(self._kernel.value, scale), scale
