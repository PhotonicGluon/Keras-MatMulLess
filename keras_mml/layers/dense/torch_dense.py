"""
Implements a matmul-less Dense layer, with PyTorch optimizations.
"""

import torch
import torch.nn.functional as F

from keras_mml.layers.dense.base_dense import BaseDenseMML


class TorchDenseMML(BaseDenseMML):
    # TODO: ADD DOCS
    def _call(self, x_norm: torch.Tensor) -> torch.Tensor:
        # Get the quantized activations and weights
        x_quantized, x_scale = self._activations_quantization(x_norm)
        x_quantized = x_norm + (x_quantized - x_norm).detach()  # STE trick using torch detach

        if self._weight_scale:
            # Weights should have been pre-quantized
            w_quantized, w_scale = self.w, self._weight_scale
        else:
            w_quantized, w_scale = self._weights_quantization(self.w)
            w_quantized = self.w + (w_quantized - self.w).detach()

        scaling = w_scale * x_scale

        # Perform kernel operation
        # TODO: Make this more efficient when we are doing inference only
        y = F.linear(x_quantized, w_quantized) / scaling  # The `matmul` should just involve addition and subtraction
        return y
