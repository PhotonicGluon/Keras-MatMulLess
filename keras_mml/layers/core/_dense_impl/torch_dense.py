"""
PyTorch implementation of the core algorithm in the matmul-less Dense layer.
"""

from typing import Tuple

import torch

from keras_mml.layers.core._dense_impl.base_dense import EPSILON, BaseDenseMML


@torch.compile(mode="reduce-overhead")
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


@torch.compile(mode="reduce-overhead")
def _compute_kernel_scale(w: torch.Tensor) -> float:
    """
    Computes the scale factor of the kernel matrix.

    Args:
        w: Kernel matrix.

    Returns:
        Scale factor.
    """

    return 1.0 / torch.mean(torch.abs(w)).clamp_(EPSILON)


@torch.compile(mode="reduce-overhead")
def _kernel_quantization_for_training(w: torch.Tensor) -> torch.Tensor:
    """
    Quantizes the kernel values to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

    Args:
        w: Kernel matrix.

    Returns:
        The quantized kernel with the scaling applied.
    """

    scale = _compute_kernel_scale(w)
    u = torch.clip(torch.round(w * scale), -1, 1)
    return u / scale


@torch.compile(mode="reduce-overhead")
def _kernel_quantization_for_saving(w: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Quantizes the kernel values to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

    Args:
        w: Kernel matrix.

    Returns:
        Both the quantized kernel and the scale will be returned, with the scale **not** applied to
        the quantized kernel.
    """

    scale = _compute_kernel_scale(w)
    u = torch.clip(torch.round(w * scale), -1, 1)
    return u, scale


@torch.compile(mode="reduce-overhead")
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


@torch.compile(mode="reduce-overhead")
def _get_w_quantized(w: torch.Tensor) -> torch.Tensor:
    """
    Gets the quantized kernel matrix, with support for the backward direction by using STE gradient
    bypass.

    We use a Straight-Through Estimator (STE) trick by stopping gradient propagation.

    Args:
        w: Kernel matrix.

    Returns:
        Quantized kernel matrix.
    """

    return w + w + (_kernel_quantization_for_training(w) - w).detach()


@torch.compile(mode="reduce-overhead")
def _get_quantized_arrays_for_training(x_norm: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets the quantized activation and kernel values for training the model.

    Args:
        x_norm: Normalized activation values.
        w: Kernel matrix.

    Returns:
        A tuple. The first value is the quantized activation values. The second is the quantized
        kernel values.
    """

    x_quantized = _get_x_quantized(x_norm)
    w_quantized = _get_w_quantized(w)
    return x_quantized, w_quantized


@torch.compile(mode="reduce-overhead")
def _get_quantized_arrays_for_inference(
    x_norm: torch.Tensor, w: torch.Tensor, w_scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets the quantized activation and kernel values for inference.

    Args:
        x_norm: Normalized activation values.
        w: Kernel matrix.
        w_scale: Scaling factor for the kernel matrix.

    Returns:
        A tuple. The first value is the quantized activation values. The second is the quantized
        kernel values.
    """

    x_quantized = _get_x_quantized(x_norm)
    w_quantized = w / w_scale
    return x_quantized, w_quantized


class TorchDenseMML(BaseDenseMML):
    @staticmethod
    def _activations_quantization(x: torch.Tensor):
        return _activations_quantization(x)

    @staticmethod
    def _compute_kernel_scale(w: torch.Tensor) -> float:
        return _compute_kernel_scale(w)

    def _kernel_quantization_for_training(self, w: torch.Tensor) -> torch.Tensor:
        return _kernel_quantization_for_training(w)

    def _kernel_quantization_for_saving(self, w: torch.Tensor) -> Tuple[torch.Tensor, float]:
        return _kernel_quantization_for_saving(w)

    def _get_quantized_arrays(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._kernel_scale:
            return _get_quantized_arrays_for_inference(x_norm, self._kernel, self._kernel_scale)
        else:
            return _get_quantized_arrays_for_training(x_norm, self._kernel)
