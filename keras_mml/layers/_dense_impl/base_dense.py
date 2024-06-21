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

        #: Variable storing the weights matrix.
        self.w: keras.Variable = None

        self._weight_scale = None  #: Used for when the layer is loaded from file

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
