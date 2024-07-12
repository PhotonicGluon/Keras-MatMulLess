"""
Base class for all matmul-less dense layers.
"""

from typing import Any, Tuple

import keras
import numpy as np
from jaxtyping import Float

from keras_mml.utils.array.ternary_multiplication import ternary_multiplication

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
    def _compute_kernel_scale(w: Float[np.ndarray, "*dims"]) -> float:
        """
        Computes the scale factor of the kernel matrix.

        Args:
            w: Kernel matrix.

        Returns:
            Scale factor.
        """

        raise NotImplementedError  # pragma: no cover

    @staticmethod
    def _quantize_kernel(w: Float[np.ndarray, "*dims"], scale: float) -> Float[np.ndarray, "*dims"]:
        """
        Quantizes the kernel values to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

        Args:
            w: Kernel matrix.
            scale: Scaling factor.

        Returns:
            The quantized kernel without scaling applied.
        """

        raise NotImplementedError  # pragma: no cover

    def _get_quantized_arrays(
        self, x_norm: Float[np.ndarray, "*dims"]
    ) -> Tuple[Float[np.ndarray, "*dims_1"], Float[np.ndarray, "*dims_2"], float]:
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
    def _ternary_multiplication(
        x_quantized: Float[np.ndarray, "*dims_1"], w_quantized: Float[np.ndarray, "*dims_2"], w_scale: float
    ) -> Any:
        """
        Applies the ternary multiplication algorithm.

        Args:
            x_quantized: Quantized activation values.
            w_quantized: Quantized kernel matrix without scaling applied.
            w_scale: Scale factor for the kernel matrix.

        Returns:
            Multiplied matrix.
        """

        return ternary_multiplication(x_quantized, w_quantized, w_scale)
