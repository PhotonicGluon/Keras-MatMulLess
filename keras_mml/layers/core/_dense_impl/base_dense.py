"""
Base class for all matmul-less dense layers.
"""

from typing import Any, Tuple

import keras
from keras import ops

EPSILON = 1e-5
HUGE = 1e9


class BaseDenseMML:
    """
    Base dense layer that exposes methods that needs to be overridden.
    """

    def __init__(self):
        """
        Initialization method.
        """

        self._kernel: keras.Variable = None  #: Variable storing the kernel matrix.
        self._kernel_scale = None  #: Used for when the layer is loaded from file.

    @staticmethod
    def _compute_kernel_scale(w) -> float:
        """
        Computes the scale factor of the kernel matrix.

        Args:
            w: Kernel matrix.

        Returns:
            Scale factor.
        """

        raise NotImplementedError  # pragma: no cover

    @staticmethod
    def _quantize_kernel(w, scale: float):
        """
        Quantizes the kernel values to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

        Args:
            w: Kernel matrix.
            scale: Scaling factor.

        Returns:
            The quantized kernel without scaling applied.
        """

        raise NotImplementedError  # pragma: no cover

    def _get_quantized_arrays(self, x_norm) -> Tuple[Any, Any, float]:
        """
        Gets the quantized activation and weight values.

        Args:
            x_norm: Normalized activation values.

        Returns:
            A tuple. The first value is the quantized activation values. The second is the quantized
            weight values without scaling applied. The last value is the scaling factor.
        """

        raise NotImplementedError  # pragma: no cover

    @staticmethod
    def _ternary_multiplication(x_quantized, w_quantized, w_scale):
        """
        Applies the ternary multiplication algorithm.

        Args:
            x_quantized: Quantized activation values.
            w_quantized: Quantized kernel matrix without scaling applied.
            w_scale: Scale factor for the kernel matrix.
        """

        # TODO: Optimize
        return ops.matmul(
            x_quantized, w_quantized / w_scale
        )  # The `matmul` should just involve addition and subtraction
