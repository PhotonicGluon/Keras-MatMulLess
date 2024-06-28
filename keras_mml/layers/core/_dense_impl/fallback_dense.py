"""
Fallback implementation of the core algorithm in the matmul-less Dense layer.
"""

from typing import Any, Tuple

from keras import ops

from keras_mml.layers.core._dense_impl.base_dense import EPSILON, HUGE, BaseDenseMML


class FallbackDenseMML(BaseDenseMML):
    @staticmethod
    def _activations_quantization(x):
        scale = 127.0 / ops.expand_dims(ops.clip(ops.max(ops.abs(x), axis=-1), EPSILON, HUGE), -1)
        y = ops.clip(ops.round(x * scale), -128, 127) / scale
        return y

    @staticmethod
    def _compute_kernel_scale(w) -> float:
        return 1.0 / ops.clip(ops.mean(ops.abs(w)), EPSILON, HUGE)

    def _kernel_quantization_for_training(self, w) -> Any:
        scale = self._compute_kernel_scale(w)
        u = ops.clip(ops.round(w * scale), -1, 1)
        return u / scale

    def _kernel_quantization_for_saving(self, w) -> Tuple[Any, float]:
        scale = self._compute_kernel_scale(w)
        u = ops.clip(ops.round(w * scale), -1, 1)
        return u, scale

    def _get_quantized_arrays(self, x_norm) -> Tuple[Any, Any]:
        # Get the quantized activations and kernel
        # (We use a Straight-Through Estimator (STE) trick by stopping gradient propagation)
        x_quantized = x_norm + ops.stop_gradient(self._activations_quantization(x_norm) - x_norm)

        if self._kernel_scale:
            # Kernel values should have been pre-quantized
            w_quantized = self._kernel / self._kernel_scale
        else:
            w = self._kernel
            w_quantized = w + ops.stop_gradient(self._kernel_quantization_for_training(w) - w)

        return x_quantized, w_quantized
