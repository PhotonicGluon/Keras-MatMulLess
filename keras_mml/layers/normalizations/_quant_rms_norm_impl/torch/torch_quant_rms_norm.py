"""
PyTorch implementation for the quantized RMSNorm layer.
"""

import keras
import torch
from jaxtyping import Float

from keras_mml.layers.normalizations._quant_rms_norm_impl.base_quant_rms_norm import EPSILON, BaseQuantRMSNorm
from keras_mml.layers.normalizations._quant_rms_norm_impl.torch.impl_2d import quant_rms_norm_2d
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


@keras.saving.register_keras_serializable(package="keras_mml")
class TorchQuantRMSNorm(BaseQuantRMSNorm):
    """
    PyTorch implementation of Root Mean Square Normalization (RMSNorm) with 8-bit quantization.
    """

    def call(self, inputs: Float[torch.Tensor, "batch_size *dims"]) -> Float[torch.Tensor, "batch_size *dims"]:
        if inputs.ndim == 2:  # More efficient
            return quant_rms_norm_2d(inputs, self._gain, self._bias, epsilon=EPSILON)

        x_norm = super().call(inputs)
        return _get_x_quantized(x_norm)
