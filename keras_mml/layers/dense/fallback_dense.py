"""
Implements a matmul-less Dense layer, for when the backend is not usable.

TODO: REMOVE ONCE ALL BACKENDS ARE SETTLED
"""

from typing import Any, Tuple

from keras import ops
from overrides import override

from keras_mml.layers.dense.base_dense import EPSILON, HUGE, BaseDenseMML


class FallbackDenseMML(BaseDenseMML):
    @staticmethod
    def _activations_quantization(x):
        """
        Quantizes the activations to 8-bit precision using absmax quantization.

        Args:
            x: Array of quantization values.

        Returns:
            The quantized activation values.
        """

        scale = 127.0 / ops.clip(ops.max(ops.abs(x), axis=-1, keepdims=True), EPSILON, HUGE)
        y = ops.clip(ops.round(x * scale), -128, 127) / scale
        return y

    @staticmethod
    def _weights_quantization(w) -> Any:
        """
        Quantizes the weights to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

        Args:
            w: Array of weights.
            with_scale: Whether the scale value should be returned along with the quantized values.

        Returns:
            The quantized weights.
        """

        scale = 1.0 / ops.clip(ops.mean(ops.abs(w)), EPSILON, HUGE)
        u = ops.clip(ops.round(w * scale), -1, 1) / scale
        return u

    @override
    def _get_quantized_arrays(self, x_norm) -> Tuple[Any, Any]:
        # Get the quantized activations and weights
        x_quantized = self._activations_quantization(x_norm)

        if self._weight_scale:
            # Weights should have been pre-quantized
            w_quantized = self.w / self._weight_scale
        else:
            w_quantized = self._weights_quantization(self.w)

        return x_quantized, w_quantized
