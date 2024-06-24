"""
Implements a matmul-less Gated Recurrent Unit (GRU) layer.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
from keras import activations, ops

from keras_mml.layers.dense import DenseMML
from keras_mml.layers.rms_norm import RMSNorm


@keras.saving.register_keras_serializable(package="keras_mml")
class GRUCellMML(keras.Layer):
    """
    Based on HGRU/HGRN...?

    See https://arxiv.org/pdf/2303.06349?

    TODO: ADD DOCS
    """

    def __init__(
        self,
        units: int,
        activation: str = "silu",
        recurrent_activation: str = "sigmoid",
        seed: Optional[int] = None,
        **kwargs,
    ):  # TODO: Add more
        """
        TODO: ADD DOCS
        """

        if units <= 0:
            raise ValueError(
                f"Received an invalid value for argument `units`, expected a positive integer, got {units}."
            )

        super().__init__(**kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=2)

        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)

        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed=seed)

        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape: Tuple[int, ...]):
        """
        TODO: ADD DOCS
        """

        super().build(input_shape)

        self.f_gate = DenseMML(self.units, activation=self.recurrent_activation, use_bias=True, name="forget_gate")
        self.f_gate.build(input_shape)

        self.c_gate = DenseMML(self.units, activation=self.activation, use_bias=True, name="candidate_state_gate")
        self.c_gate.build(input_shape)

        self.g_gate = keras.layers.Dense(
            self.units, activation=self.recurrent_activation, use_bias=True, name="data_gate"
        )
        self.g_gate.build(input_shape)

        self.o_gate = keras.layers.Dense(self.units, use_bias=True, name="output_gate")
        self.o_gate.build((None, self.units))

        self.built = True

    def call(self, inputs, states, training=False):
        """
        TODO: ADD DOCS
        """

        # Get the previous state
        h_tm1 = states[0] if keras.tree.is_nested(states) else states

        # Get main gate outputs
        f = self.f_gate(inputs)
        c = self.c_gate(inputs)
        g = self.g_gate(inputs)

        # Compute new state
        h = f * h_tm1 + (1 - f) * c
        new_state = [h] if keras.tree.is_nested(states) else h

        # Get output
        o_prime = g * h
        output = self.o_gate(o_prime)

        return output, new_state

    def get_config(self):
        """
        Gets the configuration for the layer.

        Returns:
            Layer configuration.
        """

        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": activations.serialize(self.activation),
                "recurrent_activation": activations.serialize(self.recurrent_activation),
                "seed": self.seed,
            }
        )
        return config

    def get_initial_state(self, batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Gets the initial states.

        Args:
            batch_size: Batch size for the cell. Defaults to None.

        Returns:
            Initial states.
        """

        return [ops.zeros((batch_size, self.state_size), dtype=self.compute_dtype)]


@keras.saving.register_keras_serializable(package="keras_mml")
class GRUMML(keras.layers.RNN):
    """
    TODO: ADD DOCS
    """

    def __init__(
        self, units: int, activation: str = "silu", recurrent_activation: str = "sigmoid", **kwargs
    ):  # TODO: ADD MORE
        """
        TODO: ADD DOCS
        """

        cell = GRUCellMML(units, activation=activation, recurrent_activation=recurrent_activation, name="grumml_cell")
        super().__init__(cell, **kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=3)

    # Properties
    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    # Public methods
    def call(self, sequences, initial_state: Optional[Any] = None, mask: Optional[Any] = None, training: bool = False):
        """
        TODO: ADD DOCS
        """
        return super().call(sequences, initial_state=initial_state, mask=mask, training=training)

    def inner_loop(self, sequences, initial_state, mask, training: bool = False):
        """
        TODO: ADD DOCS
        """

        if keras.tree.is_nested(initial_state):
            initial_state = initial_state[0]
        if keras.tree.is_nested(mask):
            mask = mask[0]
        return super().inner_loop(sequences, initial_state, mask=mask, training=training)

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
