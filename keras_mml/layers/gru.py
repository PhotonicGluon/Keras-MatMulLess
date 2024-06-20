"""
Implements a matmul-less Gated Recurrent Unit (GRU) layer.
"""

from typing import Any, Dict, Optional, Tuple

import keras
from keras import activations, ops

from keras_mml.layers.dense import DenseMML


@keras.saving.register_keras_serializable(package="keras_mml")
class GRUCellMML(keras.Layer):
    """
    TODO: ADD DOCS
    """

    def __init__(
        self,
        units: int,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        **kwargs,
    ):  # TODO: Add more
        """
        TODO: ADD DOCS
        """

        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, " f"expected a positive integer, got {units}."
            )

        self.input_spec = keras.layers.InputSpec(ndim=2)

        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)

    def build(self, input_shape: Tuple[int, ...]):
        """
        TODO: ADD DOCS
        """

        self.kernel = DenseMML(self.units * 3, name="kernel")  # Will be used for $W_r$, $W_z$, and $W$
        self.kernel.build(input_shape)

        self.recurrent_kernel = DenseMML(
            self.units * 3, name="recurrent_kernel"
        )  # Will be used for $U_r$, $U_z$, and $U$
        self.recurrent_kernel.build((None, self.units))

        self.built = True

    def call(self, inputs, states, training=False):
        """
        TODO: ADD DOCS
        """

        h_tm1 = states[0] if keras.tree.is_nested(states) else states  # Previous state

        # Inputs projected by all gate matrices at once
        matrix_x = self.kernel(inputs)
        x_r, x_z, x_h = ops.split(matrix_x, 3, axis=-1)

        # The hidden state is projected by all gate matrices at once
        matrix_inner = self.recurrent_kernel(h_tm1)
        recurrent_r = matrix_inner[:, : self.units]
        recurrent_z = matrix_inner[:, self.units : self.units * 2]
        recurrent_h = matrix_inner[:, self.units * 2 :]

        # Apply recurrent activations
        r = self.recurrent_activation(x_r + recurrent_r)
        z = self.recurrent_activation(x_z + recurrent_z)

        recurrent_h = r * recurrent_h

        # Form h_temp, which is $\tilde{h}_j^{\langle t \rangle}$
        h_temp = self.activation(x_h + recurrent_h)

        # Previous and candidate states are mixed by the update gate
        h = z * h_tm1 + (1 - z) * h_temp
        new_state = [h] if keras.tree.is_nested(states) else h
        return h, new_state


@keras.saving.register_keras_serializable(package="keras_mml")
class GRUMML(keras.layers.RNN):
    """
    TODO: ADD DOCS
    """

    def __init__(
        self, units: int, activation: str = "tanh", recurrent_activation: str = "sigmoid", **kwargs
    ):  # TODO: ADD MORE
        """
        TODO: ADD DOCS
        """

        cell = GRUCellMML(units, activation=activation, recurrent_activation=recurrent_activation)
        super().__init__(cell, **kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=3)

    def call(self, sequences, initial_state: Optional[Any] = None, mask: Optional[Any] = None, training: bool = False):
        """
        TODO: ADD DOCS
        """
        return super().call(sequences, initial_state=initial_state, mask=mask, training=training)

    def get_config(self) -> Dict[str, Any]:
        """
        Gets the configuration for the layer.

        Returns:
            Layer configuration.
        """

        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "recurrent_activation": activations.serialize(self.recurrent_activation),
        }
        base_config = super().get_config()
        del base_config["cell"]

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GRUMML":
        """
        Creates the layer from the given configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Created :py:class:`~GRUMML` instance.
        """

        return cls(**config)
