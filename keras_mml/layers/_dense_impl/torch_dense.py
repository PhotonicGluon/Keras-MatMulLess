"""
PyTorch implementation of the core algorithm in the matmul-less Dense layer.
"""

from typing import Tuple

import torch
from overrides import override

from keras_mml.layers._dense_impl.base_dense import BaseDenseMML


class TorchDenseMML(BaseDenseMML):
    """
    Dense layer without matrix multiplication, implemented in PyTorch.

    Implementation largely follows https://github.com/microsoft/unilm/tree/master/bitnet.
    """

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
