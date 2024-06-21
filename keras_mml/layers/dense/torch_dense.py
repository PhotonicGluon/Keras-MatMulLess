"""
Implements a matmul-less Dense layer, with PyTorch optimizations.
"""

from typing import Tuple, Union

import torch
from overrides import override

from keras_mml.layers.dense.base_dense import EPSILON, BaseDenseMML


class TorchDenseMML(BaseDenseMML):
    """
    Dense layer without matrix multiplication, implemented in PyTorch.

    Implementation largely follows https://github.com/microsoft/unilm/tree/master/bitnet.
    """

    @staticmethod
    def _activations_quantization(x: torch.Tensor) -> torch.Tensor:
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=EPSILON)
        y = (x * scale).round().clamp_(-128, 127) / scale
        return y

    @staticmethod
    def _weights_quantization(w: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        scale = 1.0 / w.abs().mean().clamp_(min=EPSILON)
        u = (w * scale).round().clamp_(-1, 1) / scale
        return u

    @override(check_signature=False)
    def _get_quantized_arrays(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the quantized activations and weights
        x_quantized = (
            x_norm + (self._activations_quantization(x_norm) - x_norm).detach()
        )  # STE trick using torch detach

        if self._weight_scale:
            # Weights should have been pre-quantized
            w_quantized = self.w / self._weight_scale
        else:
            w: torch.Tensor = self.w.value
            w_quantized = w + (self._weights_quantization(w) - w).detach()

        return x_quantized, w_quantized
