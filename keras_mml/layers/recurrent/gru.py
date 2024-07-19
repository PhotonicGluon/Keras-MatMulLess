"""
Implements an almost matmul-less Gated Recurrent Unit (GRU) layer.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
from einops import rearrange
from jaxtyping import Float
from keras import activations, constraints, initializers, ops, regularizers

from keras_mml.layers.core import DenseMML
from keras_mml.layers.recurrent.rnn import RNN


@keras.saving.register_keras_serializable(package="keras_mml")
class GRUCellMML(keras.Layer):
    """
    Cell class for the :py:class:`~GRUMML` layer.

    This class processes one step within the whole time sequence input, whereas :py:class:`~GRUMML`
    processes the whole sequence.

    Attributes:
        units: Dimensionality of the output space.
        fully_mml: Whether to use matmul-free operations for all the layers.
        num_heads: Number of heads to use when performing the recurrent step.
        activation: Activation function to use.
        recurrent_activation: Activation function to use for the recurrent step.
        use_bias: Whether to use a bias vector for the layer.
        weights_initializer: Initializer for the gates' matrices. Used for the linear transformation
            of the inputs.
        bias_initializer: Initializer for the bias vector.
        weights_regularizer: Regularizer function applied to the gates' matrices.
        bias_regularizer: Regularizer function applied to the bias vector.
        weights_constraint: Constraint function applied to the gates' matrices.
        bias_constraint: Constraint function applied to the bias vector.
        state_size: Size of the recurrent state.
        output_size: Size of the output vector.
    """

    def __init__(
        self,
        units: int,
        fully_mml: bool = False,
        num_heads: int = 1,
        activation: str = "silu",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        weights_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        weights_regularizer: Optional[str] = None,
        bias_regularizer: Optional[str] = None,
        weights_constraint: Optional[str] = None,
        bias_constraint: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes a new instance of the layer.

        Args:
            units: Dimensionality of the output space.
            fully_mml: Whether to use matmul-free operations for all the layers.
            num_heads: Number of heads to use when performing the recurrent step.
            activation: Activation function to use.
            recurrent_activation: Activation function to use for the recurrent step.
            use_bias: Whether to use a bias vector for the layer.
            weights_initializer: Initializer for the gates' matrices. Used for the linear
                transformation of the inputs.
            bias_initializer: Initializer for the bias vector.
            weights_regularizer: Regularizer function applied to the gates' matrices.
            bias_regularizer: Regularizer function applied to the bias vector.
            weights_constraint: Constraint function applied to the gates' matrices.
            bias_constraint: Constraint function applied to the bias vector.
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

        if units % num_heads != 0:
            raise ValueError(
                "Output dimension must be divisible by number of heads. "
                f"Got output dimension of {units} but wanted to use {num_heads} heads."
            )

        super().__init__(**kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=2)

        self.state_size = units
        self.output_size = units

        # Main attributes
        self.units = units
        self.fully_mml = fully_mml
        self.num_heads = num_heads
        self.use_bias = use_bias

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)

        self.weights_initializer = initializers.get(weights_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.weights_regularizer = regularizers.get(weights_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.weights_constraint = constraints.get(weights_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self._head_dim = None

        # Hidden weights/layers
        self._kernel = None
        self._g_gate = None
        self._o_gate = None

    def build(self, input_shape: Tuple[int, int]):
        """
        Create layer weights.

        Args:
            input_shape: Shape of the input.
        """

        self._head_dim = max(1, self.units // self.num_heads)

        if self.fully_mml:
            # Incorporate the forget gate (f), the candidate state gate (c), and the data gate (g) in one kernel matrix
            self._kernel = DenseMML(
                self.units * 3,
                use_bias=self.use_bias,
                kernel_initializer=self.weights_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.weights_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.weights_constraint,
                bias_constraint=self.bias_constraint,
                name="kernel",
            )
            self._kernel.build(input_shape)

            self._g_gate = None

            self._o_gate = DenseMML(
                self.units,
                use_bias=self.use_bias,
                kernel_initializer=self.weights_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.weights_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.weights_constraint,
                bias_constraint=self.bias_constraint,
                name="output_gate",
            )
            self._o_gate.build((None, self.units))
        else:
            # Incorporate the only the forget gate (f) and the candidate state gate (c) in the kernel matrix. The data
            # gate needs to be separated
            self._kernel = DenseMML(
                self.units * 2,
                use_bias=self.use_bias,
                kernel_initializer=self.weights_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.weights_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.weights_constraint,
                bias_constraint=self.bias_constraint,
                name="kernel",
            )
            self._kernel.build(input_shape)

            self._g_gate = keras.layers.Dense(
                self.units,
                activation=self.recurrent_activation,
                use_bias=self.use_bias,
                kernel_initializer=self.weights_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.weights_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.weights_constraint,
                bias_constraint=self.bias_constraint,
                name="data_gate",
            )
            self._g_gate.build(input_shape)

            self._o_gate = keras.layers.Dense(
                self.units,
                use_bias=self.use_bias,
                kernel_initializer=self.weights_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.weights_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.weights_constraint,
                bias_constraint=self.bias_constraint,
                name="output_gate",
            )
            self._o_gate.build((None, self.units))

        self.built = True

    def call(
        self, inputs: Float[np.ndarray, "batch_size features"], states: Float[np.ndarray, "*state_dims"], training=False
    ) -> Float[np.ndarray, "batch_size units"]:
        """
        Calling method of the cell.

        Args:
            inputs: Inputs into the layer.
            states: State(s) from the previous timestep.
            training: Whether the layer should behave in training mode or in inference mode.

        Returns:
            Transformed inputs.
        """

        # Get the previous state
        h_tm1 = states[0] if keras.tree.is_nested(states) else states

        # Pass inputs through the kernel matrix
        kernel_out = self._kernel(inputs)

        if self.fully_mml:
            # Split into f, c, and g
            f, c, g = ops.split(kernel_out, 3, axis=-1)

            # Remember to apply activation for g!
            g = self.recurrent_activation(g)
        else:
            # Split into f and c
            f, c = ops.split(kernel_out, 2, axis=-1)

            # Process g separately
            g = self._g_gate(inputs)  # This already applies the activation, so no need to do it again

        # Apply activations for f and c
        f = self.recurrent_activation(f)
        c = self.activation(c)

        # Split for multiple heads
        f, c = map(
            lambda x: rearrange(x, "batch (heads features) -> batch heads features", heads=self.num_heads), (f, c)
        )

        # Compute new state
        h = f * h_tm1 + (1 - f) * c
        new_state = [h] if keras.tree.is_nested(states) else h

        # Get output
        o_prime = g * rearrange(h, "batch heads features -> batch (heads features)")
        output = self._o_gate(o_prime)

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
                "use_bias": self.use_bias,
                "weights_initializer": initializers.serialize(self.weights_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "weights_regularizer": regularizers.serialize(self.weights_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "weights_constraint": constraints.serialize(self.weights_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config

    def get_initial_state(self, batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Gets the initial states.

        Args:
            batch_size: Batch size for the cell.

        Returns:
            Initial states.
        """

        return [ops.zeros((batch_size, self.num_heads, self._head_dim), dtype=self.compute_dtype)]


@keras.saving.register_keras_serializable(package="keras_mml")
class GRUMML(RNN):
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
        num_heads: Number of heads to use when performing the recurrent step.
        activation: Activation function to use.
        recurrent_activation: Activation function to use for the recurrent step.
        use_bias: Whether to use a bias vector for the layer.
        weights_initializer: Initializer for the gates' matrices. Used for the linear transformation
            of the inputs.
        bias_initializer: Initializer for the bias vector.
        weights_regularizer: Regularizer function applied to the gates' matrices.
        bias_regularizer: Regularizer function applied to the bias vector.
        weights_constraint: Constraint function applied to the gates' matrices.
        bias_constraint: Constraint function applied to the bias vector.

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
        use_bias: bool = True,
        weights_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        weights_regularizer: Optional[str] = None,
        bias_regularizer: Optional[str] = None,
        weights_constraint: Optional[str] = None,
        bias_constraint: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes a new instance of the layer.

        Args:
            units: Dimensionality of the output space.
            fully_mml: Whether to use matmul-free operations for all the layers.
            num_heads: Number of heads to use for the recurrent step. See |HGRN2|_, section 3.2, for
                details on the multi-headed variant.
            activation: Activation function to use.
            recurrent_activation: Activation function to use for the recurrent step.
            use_bias: Whether to use a bias vector for the layer.
            weights_initializer: Initializer for the gates' matrices. Used for the linear
                transformation of the inputs.
            bias_initializer: Initializer for the bias vector.
            weights_regularizer: Regularizer function applied to the gates' matrices.
            bias_regularizer: Regularizer function applied to the bias vector.
            weights_constraint: Constraint function applied to the gates' matrices.
            bias_constraint: Constraint function applied to the bias vector.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the units provided is not a positive integer.
            ValueError: If the number of heads to use is not a positive integer.
            ValueError: If the number of heads does not divide the units provided.

        .. |HGRN2| replace:: *HGRN2: Gated Linear RNNs with State Expansion*
        .. _HGRN2: https://arxiv.org/pdf/2404.07904v1
        """

        cell = GRUCellMML(
            name="grumml_cell",
            units=units,
            fully_mml=fully_mml,
            num_heads=num_heads,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
            weights_regularizer=weights_regularizer,
            bias_regularizer=bias_regularizer,
            weights_constraint=weights_constraint,
            bias_constraint=bias_constraint,
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

    @property
    def use_bias(self):
        """
        :meta private:
        """
        return self.cell.use_bias

    @property
    def weights_initializer(self):
        """
        :meta private:
        """
        return self.cell.weights_initializer

    @property
    def bias_initializer(self):
        """
        :meta private:
        """
        return self.cell.bias_initializer

    @property
    def weights_regularizer(self):
        """
        :meta private:
        """
        return self.cell.weights_regularizer

    @property
    def bias_regularizer(self):
        """
        :meta private:
        """
        return self.cell.bias_regularizer

    @property
    def weights_constraint(self):
        """
        :meta private:
        """
        return self.cell.weights_constraint

    @property
    def bias_constraint(self):
        """
        :meta private:
        """
        return self.cell.bias_constraint

    # Public methods
    def call(
        self,
        sequences: Float[np.ndarray, "batch_size timesteps features"],
        initial_state: Optional[List] = None,
        mask: Optional[Any] = None,
        training: bool = False,
    ) -> Float[np.ndarray, "batch_size timesteps"]:
        """
        Calling method of the layer.

        Args:
            sequences: Inputs into the layer.
            initial_state: List of initial state tensors to be passed to the first call of the cell.
                If not provided, will cause creation of zero-filled initial state tensors.
            mask: Binary tensor indicating whether a given timestep should be masked. An individual
                True entry indicates that the corresponding timestep should be utilized, while a
                False entry indicates that the corresponding timestep should be ignored.
            training: Indicates whether the layer should behave in training mode or in inference
                mode. This argument is passed to the cell when calling it.

        Returns:
            Transformed inputs.
        """

        output = super().call(sequences, initial_state=initial_state, mask=mask, training=training)

        if keras.config.backend() == "jax" and mask is not None:
            # I have no idea why, but when using the Jax backend along with masking, a second copy of the outputs is
            # returned along with the first. The following code just takes the first output and ignores the rest.

            output = output[0]

        return output

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
            "use_bias": self.use_bias,
            "weights_initializer": initializers.serialize(self.weights_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "weights_regularizer": regularizers.serialize(self.weights_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "weights_constraint": constraints.serialize(self.weights_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
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
