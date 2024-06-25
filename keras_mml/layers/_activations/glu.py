"""
Implements a matmul-less Gated Linear Unit (GLU) layer.
"""

from typing import Any, Dict, Optional, Tuple

import keras
from keras import activations, ops

from keras_mml.layers.core import DenseMML

#: Set of activations that can be used with :py:class:`~GLUMML`.
PERMITTED_ACTIVATIONS = {"sigmoid", "linear", "relu", "gelu", "silu", "selu"}


@keras.saving.register_keras_serializable(package="keras_mml")
class GLUMML(keras.Layer):
    """
    Gated Linear Unit (GLU) without matrix multiplication.

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
            hidden_ratio: Ratio adjusting the intermediate size.
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

        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(
                f"Received an invalid value for argument `units`, expected a positive integer, got {units}"
            )

        if activation not in PERMITTED_ACTIVATIONS:
            raise ValueError(
                f"GLU activation '{activation}' not allowed; permitted activations are {PERMITTED_ACTIVATIONS}"
            )

        self.input_spec = keras.layers.InputSpec(ndim=2)

        self.units = units
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.activation = activations.get(activation)

        self.down_dense = DenseMML(units)  # We will use this layer for $W_d$

    # Public methods
    def build(self, input_shape: Tuple[int, ...]):
        """
        Create layer weights.

        Args:
            input_shape: Shape of the input.

        Raises:
            ValueError: If the input shape does not have a rank of 2 (i.e., something like
                ``(batch_size, d0)``).
        """

        self.hidden_size = input_shape[-1]

        if self.intermediate_size is None:
            # The `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
            intermediate_size = int(self.hidden_size * self.hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
            self.intermediate_size = intermediate_size

        self.gate_dense = DenseMML(self.intermediate_size * 2)  # We will use this layer for both $W_g$ and $W_u$
        self.gate_dense.build(input_shape)

    def call(self, inputs):
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer.

        Returns:
            Transformed inputs.
        """

        g_and_u = self.gate_dense(inputs)
        g, u = ops.split(g_and_u, 2, axis=-1)
        p = ops.multiply(self.activation(g), u)
        d = self.down_dense(p)
        return d

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
