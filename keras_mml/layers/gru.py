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
class GRUCellMML(keras.Layer):  # TODO: Rename?
    """
    Based on HGRU/HGRN...?

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

        self.kernel_dense = keras.layers.Dense(
            self.units * 3, name="kernel", use_bias=False
        )  # Will be used for $W_f$, $W_c$, and $W_g$
        self.kernel_dense.build(input_shape)

        # self.g_norm = RMSNorm(input_shape[-1], name="output_gate_norm")
        # self.g_norm.build(input_shape)

        # self.out_dense = keras.layers.Dense(self.units, name="out_kernel")  # Will be used for $W_o$
        # self.out_dense.build((None, self.units))

        self.built = True

    def call(self, inputs, states, training=False):
        """
        TODO: ADD DOCS
        """

        h_tm1 = states[0] if keras.tree.is_nested(states) else states  # Previous state

        # Inputs projected by all gate matrices at once
        matrix_x = self.kernel_dense(inputs)
        x_f, x_i, x_c = ops.split(matrix_x, 3, axis=-1)

        # Apply activations
        f = self.recurrent_activation(x_f)
        i = self.recurrent_activation(x_i)
        c = self.activation(x_c)

        # Previous and candidate states are mixed by the update gate
        h = f * h_tm1 + i * c
        new_state = [h] if keras.tree.is_nested(states) else h

        return h, new_state

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
