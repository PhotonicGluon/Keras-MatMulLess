"""
Implements a matmul-less Gated Linear Unit (GLU) layer.
"""

from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np
from jaxtyping import Float
from keras import activations, ops

from keras_mml.layers.core import DenseMML

#: Set of activations that can be used with :py:class:`~GLUMML`.
PERMITTED_ACTIVATIONS = {"sigmoid", "linear", "relu", "gelu", "silu", "selu"}


@keras.saving.register_keras_serializable(package="keras_mml")
class GLUMML(keras.Layer):
    """
    General Gated Linear Unit (GLU) without matrix multiplications.

    This is a modified implementation of ``HGRNBitMLP`` from the `GitHub repository
    <https://github.com/ridgerchu/matmulfreellm>`_ of |MatMulFreeLLM|_ where, instead of just
    permitting the Swish activation, we permit other activations via the
    :py:attr:`~GLUMML.activation` attribute.

    See section 3.3.2 of the aforementioned paper for the notation used in the implementation of the
    code.

    Attributes:
        units: Dimensionality of the output space.
        hidden_ratio: Ratio adjusting the intermediate size.
        intermediate_size: Intermediate size. See the :py:func:`~GLUMML.__init__` method on how the
            intermediate size is determined.
        activation: GLU activation function.

    .. |MatMulFreeLLM| replace:: *Scalable MatMul-free Language Modeling*
    .. _MatMulFreeLLM: https://arxiv.org/pdf/2406.02528v5
    """

    def __init__(
        self,
        units: int,
        hidden_ratio: int = 4,
        intermediate_size: Optional[int] = None,
        activation: str = "sigmoid",
        **kwargs,
    ):
        """
        Initializes a new instance of the layer.

        Args:
            units: Dimensionality of the output space.
            hidden_ratio: Ratio adjusting the intermediate size. Ignored if an intermediate size is
                specified.
            intermediate_size: Intermediate size. If None, will choose a multiple of 256 closest to
                :math:`\\frac23 lr` where :math:`l` is the hidden shape given by the input into the
                layer and :math:`r` is the :py:attr:`~GLUMML.hidden_ratio`.
            activation: GLU activation function.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the units provided is not a positive integer.
            ValueError: If the activation function specified is not in the
                :py:const:`~PERMITTED_ACTIVATIONS`.
        """

        if units <= 0:
            raise ValueError(f"Received an invalid value for units, expected a positive integer, got {units}")

        if activation not in PERMITTED_ACTIVATIONS:
            raise ValueError(
                f"GLU activation '{activation}' not allowed; permitted activations are {PERMITTED_ACTIVATIONS}"
            )

        super().__init__(**kwargs)

        # Main attributes
        self.units = units
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.activation = activations.get(activation)

        # Hidden weights/layers
        self._gate_dense = None
        self._down_dense = None

    # Public methods
    def build(self, input_shape: Tuple[int, ...]):
        """
        Create layer weights.

        Args:
            input_shape: Shape of the input.
        """

        hidden_size = input_shape[-1]

        if self.intermediate_size is None:
            # The `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
            intermediate_size = int(hidden_size * self.hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
            self.intermediate_size = intermediate_size

        self._gate_dense = DenseMML(self.intermediate_size * 2)  # We will use this layer for both $W_g$ and $W_u$
        self._gate_dense.build(input_shape)

        self._down_dense = DenseMML(self.units)  # We will use this layer for $W_d$
        self._down_dense.build((None, self.intermediate_size))

        self.built = True

    def call(
        self, inputs: Float[np.ndarray, "batch_size *dims last_dim"]
    ) -> Float[np.ndarray, "batch_size *dims units"]:
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer.

        Returns:
            Transformed inputs.
        """

        g_and_u = self._gate_dense(inputs)
        g, u = ops.split(g_and_u, 2, axis=-1)
        p = ops.multiply(self.activation(g), u)
        d = self._down_dense(p)
        return d

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Computes the output shape of the layer.

        Args:
            input_shape: Shape of the input into the layer.

        Returns:
            Shape of the output.
        """

        input_shape = list(input_shape)
        input_shape[-1] = self.units
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration for the layer.

        Returns:
            Layer configuration.
        """

        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "hidden_ratio": self.hidden_ratio,
                "intermediate_size": self.intermediate_size,
                "activation": activations.serialize(self.activation),
            }
        )
        return config
