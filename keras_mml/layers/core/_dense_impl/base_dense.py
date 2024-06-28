"""
Base class for all matmul-less dense layers.
"""

from typing import Any, Tuple

import keras

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

    # Helper methods
    @staticmethod
    def _activations_quantization(x):
        """
        Quantizes the activations to 8-bit precision using absmax quantization.

        Args:
            x: Array of quantization values.

        Returns:
            The quantized activation values.
        """

        raise NotImplementedError

    @staticmethod
    def _compute_kernel_scale(w) -> float:
        """
        Computes the scale factor of the kernel matrix.

        Args:
            w: Kernel matrix.

        Returns:
            Scale factor.
        """

        raise NotImplementedError

    def _kernel_quantization_for_training(self, w) -> Any:
        """
        Quantizes the kernel values to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

        Args:
            w: Kernel matrix.

        Returns:
            The quantized kernel with the scaling applied.
        """

        raise NotImplementedError

    def _kernel_quantization_for_saving(self, w) -> Tuple[Any, float]:
        """
        Quantizes the kernel values to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

        Args:
            w: Kernel matrix.

        Returns:
            Both the quantized kernel and the scale will be returned, with the scale **not**
            applied to the quantized kernel.
        """

        raise NotImplementedError

    def _get_quantized_arrays(self, x_norm) -> Tuple[Any, Any]:
        """
        Gets the quantized activation and weight values.

        Args:
            x_norm: Normalized activation values.

        Returns:
            A tuple. The first value is the quantized activation values. The second is the quantized
            weight values.
        """

        raise NotImplementedError
