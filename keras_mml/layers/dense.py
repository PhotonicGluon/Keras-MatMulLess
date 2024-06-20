"""
Implements a matmul-less Dense layer.
"""

from typing import Any, Dict, Optional, Tuple

import keras
from keras import activations, initializers, ops

from keras_mml.layers.rms_norm import RMSNorm

EPSILON = 1e-5
HUGE = 1e9
WEIGHTS_NAME = "weights"


@keras.saving.register_keras_serializable(package="keras_mml")
class DenseMML(keras.Layer):
    """
    Dense layer without matrix multiplication.

    The core of the layer is the ``BitLinear`` layer described in https://arxiv.org/pdf/2310.11453
    and https://arxiv.org/pdf/2402.17764. It uses bit quantization to reduce matrix multiplication
    operations to simple addition and subtraction.

    This implementation differs from ``BitLinear`` by allowing an activation function to be
    specified. More precisely, :py:class:`~DenseMML` implements the operation
    :math:`\\mathbf{y} = \\sigma\\left(\\mathbf{x}\\mathbf{W}^\\intercal\\right)` where
    :math:`\\mathbf{x}` is the quantized input vector, :math:`\\mathbf{W}` is the quantized weights
    matrix, and :math:`\\sigma` is the element-wise activation function.

    This implementation only allows the layer to have a rank of 2. That is, the input to this layer
    must be of the form ``(batch_size, d0)``. An input shape that does not conform to this will
    raise a :py:exc:`ValueError`.

    .. WARNING::
       Once a model that uses this layer is loaded from a file, it **cannot** be retrained.

    Attributes:
        units: Dimensionality of the output space.
        activation: Activation function.
        weights_initializer: Initializer for the weights matrix.
    """

    def __init__(
        self,
        units: int,
        activation: Optional[str] = None,
        weights_initializer: str = "glorot_uniform",
        **kwargs,
    ):
        """
        Initializes a new :py:class:`~DenseMML` layer.

        Args:
            units: Dimensionality of the output space.
            activation: Activation function to use. If you don't specify anything, no activation is
                applied (i.e. "linear" activation: :math:`\\sigma(\\mathbf{x}) = \\mathbf{x}`).
            weights_initializer: Initializer for the weights matrix.

        Raises:
            ValueError: If the units provided is not a positive integer.
        """

        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(
                f"Received an invalid value for argument `units`, expected a positive integer, got {units}"
            )

        self.input_spec = keras.layers.InputSpec(ndim=2)

        self.units = units
        self.activation = activations.get(activation)
        self.weights_initializer = initializers.get(weights_initializer)

        self._weight_scale = None  # Used for when the layer is loaded from file

    # Helper methods
    @staticmethod
    def _activations_quantization(x) -> Tuple[Any, Any]:
        """
        Quantizes the activations to 8-bit precision using absmax quantization.

        This is equation (4) in the aforementioned paper.

        Args:
            x: Array of quantization values.

        Returns:
            A tuple. The first value is the quantized activation values. The second is the scaling
            factor needed for equation (11) in the paper.
        """

        scale = 127.0 / ops.clip(ops.max(ops.abs(x), axis=-1, keepdims=True), EPSILON, HUGE)
        y = ops.clip(ops.round(x * scale), -128, 127)
        return y, scale

    @staticmethod
    def _weights_quantization(w) -> Tuple[Any, Any]:
        """
        Quantizes the weights to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

        This is equation (1) in the aforementioned paper.

        Args:
            w: Array of weights.

        Returns:
            A tuple. The first is the quantized weights. The second is the scaling factor needed.
        """

        scale = 1.0 / ops.clip(ops.mean(ops.abs(w)), EPSILON, HUGE)
        u = ops.clip(ops.round(w * scale), -1, 1)
        return u, scale

    # Public methods
    def build(self, input_shape: Tuple[int, int]):
        """
        Create layer weights.

        Args:
            input_shape: Shape of the input.

        Raises:
            ValueError: If the input shape does not have a rank of 2 (i.e., something like
                ``(batch_size, d0)``).
        """

        self.w = self.add_weight(
            name=WEIGHTS_NAME,
            shape=(input_shape[-1], self.units),
            initializer=self.weights_initializer,
            trainable=True,
        )

    def call(self, inputs):
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer.

        Returns:
            Transformed inputs.
        """

        # First normalize the inputs
        input_dim = ops.shape(inputs)[1]
        x_norm = RMSNorm(input_dim)(inputs)

        # Then get the quantized activations and weights
        x_quantized, x_scale = self._activations_quantization(x_norm)

        if self._weight_scale:
            # Weights should have been pre-quantized
            w_quantized, w_scale = self.w, self._weight_scale
        else:
            w_quantized, w_scale = self._weights_quantization(self.w)

        scaling = w_scale * x_scale

        # Perform kernel operation
        # TODO: Make this more efficient when we are doing inference only
        x = ops.matmul(x_quantized, w_quantized) / scaling  # The `matmul` should just involve addition and subtraction

        # Then apply activation
        if self.activation is not None:
            x = self.activation(x)
        return x

    def compute_output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Computes the output shape given a tensor of a given shape.

        Args:
            input_shape: Input shape into the layer.

        Returns:
            Output shape after passing through the layer.
        """

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

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
                "activation": activations.serialize(self.activation),
                "weights_initializer": initializers.serialize(self.weights_initializer),
            }
        )
        return config

    def save_own_variables(self, store: Dict):
        """
        Saves the state of the layer.

        Args:
            store: Dictionary where the state of the model will be saved.
        """

        # Pre-quantize the weights
        w_quantized, w_scale = self._weights_quantization(self.w)

        # TODO: Develop more efficient weight saving method than this simple method
        store["weights"] = w_quantized
        store["scale"] = w_scale

    def load_own_variables(self, store: Dict):
        """
        Loads the state of the layer.

        Args:
            store: Dictionary from which the state of the model will be loaded.

        Raises:
            ValueError: If the layer is missing variables when loading from a file.
        """

        # TODO: Develop more efficient weight loading method than this simple method
        try:
            self.w = store["weights"][()]
            self._weight_scale = store["scale"][()]
        except ValueError:  # pragma: no cover
            raise ValueError("DenseMML layer missing values when loading from file")
