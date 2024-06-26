"""
Implements an almost matmul-less Gated Recurrent Unit (GRU) layer.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
from einops import rearrange
from keras import activations, ops

from keras_mml.layers.core import DenseMML


@keras.saving.register_keras_serializable(package="keras_mml")
class GRUCellMML(keras.Layer):
    """
    Cell class for the :py:class:`~GRUMML` layer.

    This class processes one step within the whole time sequence input, whereas :py:class:`~GRUMML`
    processes the whole sequence.

    Attributes:
        units: Dimensionality of the output space.
        fully_mml: Whether to use matmul-free operations for all the layers.
        activation: Activation function to use.
        recurrent_activation: Activation function to use for the recurrent step.
    """

    def __init__(
        self,
        units: int,
        fully_mml: bool = False,
        num_heads: int = 1,
        activation: str = "silu",
        recurrent_activation: str = "sigmoid",
        **kwargs,
    ):  # TODO: Add more arguments
        """
        Initializes a new instance of the layer.

        Args:
            units: Dimensionality of the output space.
            fully_mml: Whether to use matmul-free operations for all the layers.
            num_heads: Number of heads to use when performing the recurrent step.
            activation: Activation function to use.
            recurrent_activation: Activation function to use for the recurrent step.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the units provided is not a positive integer.
            ValueError: If the number of heads to use is not a positive integer.
            ValueError: If the number of heads does not divide the units provided.
        """

        if units <= 0:
            raise ValueError(
                f"Received an invalid value for output dimension, expected a positive integer, got {units}."
            )

        if num_heads <= 0:
            raise ValueError(
                f"Received an invalid value for the number of heads, expected a positive integer, got {num_heads}."
            )
        elif units % num_heads != 0:
            raise ValueError(
                "Output dimension must be divisible by number of heads. "
                f"Got output dimension of {units} but wanted to use {num_heads} heads."
            )

        super().__init__(**kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=2)

        self.units = units
        self.fully_mml = fully_mml
        self.num_heads = num_heads

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)

        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape: Tuple[int, int]):
        """
        Create layer weights.

        Args:
            input_shape: Shape of the input.
        """

        self.head_dim = max(1, self.units // self.num_heads)  # Ensure that head dimension is at least 1

        # Decide what layer class to use for the output-adjacent layers
        if self.fully_mml:
            output_layer_class = DenseMML
        else:
            output_layer_class = keras.layers.Dense

        # Define gates
        self.f_gate = DenseMML(self.units, activation=self.recurrent_activation, use_bias=True, name="forget_gate")
        self.f_gate.build(input_shape)

        self.c_gate = DenseMML(self.units, activation=self.activation, use_bias=True, name="candidate_state_gate")
        self.c_gate.build(input_shape)

        self.g_gate = output_layer_class(
            self.units, activation=self.recurrent_activation, use_bias=True, name="data_gate"
        )
        self.g_gate.build(input_shape)

        self.o_gate = output_layer_class(self.units, use_bias=True, name="output_gate")
        self.o_gate.build((None, self.units))

        self.built = True

    def call(self, inputs, states, training=False):
        """
        Calling method of the cell.

        Args:
            inputs: Inputs into the layer. Has shape ``(batch, features)``.
            states: State(s) from the previous timestep. Has shape ``(batch, units)``.
            training: Whether the layer should behave in training mode or in inference mode.

        Returns:
            Transformed inputs.
        """

        # Get the previous state
        h_tm1 = states[0] if keras.tree.is_nested(states) else states

        # Get main gate outputs
        f = self.f_gate(inputs)
        c = self.c_gate(inputs)

        # Split for multiple heads
        f, c = map(
            lambda x: rearrange(x, "batch (heads features) -> batch heads features", heads=self.num_heads), (f, c)
        )

        # Compute new state
        h = f * h_tm1 + (1 - f) * c
        new_state = [h] if keras.tree.is_nested(states) else h

        # Get output
        g = self.g_gate(inputs)
        o_prime = g * rearrange(h, "batch heads features -> batch (heads features)")
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
                "fully_mml": self.fully_mml,
                "num_heads": self.num_heads,
                "activation": activations.serialize(self.activation),
                "recurrent_activation": activations.serialize(self.recurrent_activation),
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

        return [ops.zeros((batch_size, self.num_heads, self.head_dim), dtype=self.compute_dtype)]


@keras.saving.register_keras_serializable(package="keras_mml")
class GRUMML(keras.layers.RNN):
    """
    Gated Recurrent Unit (GRU) layer, mostly without matrix multiplications.
    
    The implementation of this layer mostly follows the :math:`\\mathrm{MLGRU}` implementation in
    |MatMulFreeLLM|_ (see section 3.3.1). We differ from the implementation of
    :math:`\\mathrm{MLGRU}` by allowing :math:`\\mathbf{g}_t` and :math:`\\mathbf{o}_t` to be
    regular matrix multiplications, rather than just matmul-free ternary weights. The option to make
    everything ternary weights is controlled by the :py:attr:`~fully_mml` attribute.
    
    Specifically, we perform the following recurrence steps.
    
    .. math::
        \\begin{align*}
            \\mathbf{f}_t &= \\sigma(\\mathbf{x}_t\\mathbf{W}_f + \\mathbf{b}_f)\\\\
            \\mathbf{c}_t &= \\tau(\\mathbf{x}_t\\mathbf{W}_c + \\mathbf{b}_c)\\\\
            \\mathbf{h}_t &= \\mathbf{f}_t\\odot\\mathbf{h}_{t-1} 
                                + (1-\\mathbf{f}_t)\\odot\\mathbf{c}_t \\\\
            \\mathbf{g}_t &= \\sigma(\\mathbf{x}_t\\mathbf{W}_g + \\mathbf{b}_g)\\\\
            \\mathbf{o}_t' &= \\mathbf{g}_t\\odot\\mathbf{h}_t\\\\
            \\mathbf{o}_t &= \\mathbf{o}_t'\\mathbf{W}_o + \\mathbf{b}_o\\\\
        \\end{align*}
        
    where
    
    - :math:`\\mathbf{W}_f` and :math:`\\mathbf{W}_c` are ternary weights (and so do not use matrix\
        multiplications during their operation);
    - :math:`\\mathbf{W}_g` and :math:`\\mathbf{W}_o` are (possible) ternary weights, or just\
        regular weight matrices;
    - :math:`\\sigma` is the :py:attr:`~.recurrent_activation` (e.g., Sigmoid activation); and
    - :math:`\\tau` is the :py:attr:`~.activation` (e.g., Silu activation).
    
    Attributes:
        units: Dimensionality of the output space.
        fully_mml: Whether to use matmul-free operations for all the layers.
        activation: Activation function to use.
        recurrent_activation: Activation function to use for the recurrent step.
    
    .. |MatMulFreeLLM| replace:: *Scalable MatMul-free Language Modeling*
    .. _MatMulFreeLLM: https://arxiv.org/pdf/2406.02528v5
    """

    def __init__(
        self,
        units: int,
        fully_mml: bool = False,
        num_heads: int = 1,
        activation: str = "silu",
        recurrent_activation: str = "sigmoid",
        **kwargs,
    ):  # TODO: Add more arguments
        """
        Initializes a new instance of the layer.

        Args:
            units: Dimensionality of the output space.
            fully_mml: Whether to use matmul-free operations for all the layers.
            num_heads: Number of heads to use for the recurrent step. See |HRGN2|_, section 3.2 for
                details on the multi-headed variant.
            activation: Activation function to use.
            recurrent_activation: Activation function to use for the recurrent step.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the units provided is not a positive integer.

        .. |HGRN2| replace::*HGRN2: Gated Linear RNNs with State Expansion*
        .. _HGRN2: https://arxiv.org/pdf/2404.07904v1
        """

        cell = GRUCellMML(
            name="grumml_cell",
            units=units,
            fully_mml=fully_mml,
            num_heads=num_heads,
            activation=activation,
            recurrent_activation=recurrent_activation,
        )
        super().__init__(cell, **kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=3)

    # Properties
    @property
    def units(self):
        """
        :meta private:
        """
        return self.cell.units

    @property
    def fully_mml(self):
        """
        :meta private:
        """
        return self.cell.fully_mml

    @property
    def num_heads(self):
        """
        :meta private:
        """
        return self.cell.num_heads

    @property
    def activation(self):
        """
        :meta private:
        """
        return self.cell.activation

    @property
    def recurrent_activation(self):
        """
        :meta private:
        """
        return self.cell.recurrent_activation

    # Public methods
    def call(self, sequences, initial_state: Optional[List] = None, mask: Optional[Any] = None, training: bool = False):
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer, with shape ``(batch, timesteps, features)``.
            initial_state: List of initial state tensors to be passed to the first call of the cell.
                If not provided, will cause creation of zero-filled initial state tensors.
            mask: Binary tensor of shape ``(samples, timesteps)`` indicating whether a given
                timestep should be masked. An individual True entry indicates that the corresponding
                timestep should be utilized, while a False entry indicates that the corresponding
                timestep should be ignored.
            training: Indicates whether the layer should behave in training mode or in inference
                mode. This argument is passed to the cell when calling it.

        Returns:
            Transformed inputs.
        """

        output = super().call(sequences, initial_state=initial_state, mask=mask, training=training)

        if keras.config.backend() == "jax" and mask is not None:
            # FIXME:
            #   I have no idea why, but when using the Jax backend along with masking, a second copy of the outputs is
            #   returned along with the first. The following code just takes the first output and ignores the rest.

            output = output[0]

        return output

    def inner_loop(self, sequences, initial_state, mask, training: bool = False):
        """
        Handles the execution of the recurrent loop of the recurrent neural network.

        Args:
            sequences: Inputs into the layer, with shape ``(batch, timesteps, features)``.
            initial_state: List of initial state tensors to be passed to the first call of the cell.
                If not provided, will cause creation of zero-filled initial state tensors.
            mask: Binary tensor of shape ``(samples, timesteps)`` indicating whether a given
                timestep should be masked. An individual True entry indicates that the corresponding
                timestep should be utilized, while a False entry indicates that the corresponding
                timestep should be ignored.
            training: Indicates whether the layer should behave in training mode or in inference
                mode. This argument is passed to the cell when calling it.

        Returns:
            Transformed inputs.
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
            "fully_mml": self.fully_mml,
            "num_heads": self.num_heads,
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
            Created instance.
        """

        return cls(**config)
