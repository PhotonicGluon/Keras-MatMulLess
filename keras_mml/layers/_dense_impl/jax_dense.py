"""
Jax implementation of the core algorithm in the matmul-less Dense layer.
"""

from typing import Tuple

import jax
from overrides import override

from keras_mml.layers._dense_impl.base_dense import BaseDenseMML


class JaxDenseMML(BaseDenseMML):
    """
    Dense layer without matrix multiplication, implemented in Jax.
    """

    @override(check_signature=False)
    def _get_quantized_arrays(self, x_norm: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # Get the quantized activations and weights
        x_quantized = x_norm + jax.lax.stop_gradient(
            self._activations_quantization(x_norm) - x_norm
        )  # STE trick by stopping gradient propagation

        if self._weight_scale:
            # Weights should have been pre-quantized
            w_quantized = self.w / self._weight_scale
        else:
            w: jax.Array = self.w.value
            w_quantized = w + jax.lax.stop_gradient(self._weights_quantization(w) - w)

        return x_quantized, w_quantized
