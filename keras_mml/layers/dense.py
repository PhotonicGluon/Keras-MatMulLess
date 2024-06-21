"""
Implements a matmul-less Dense layer.
"""

from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np
from keras import activations, initializers, ops

from keras_mml.layers.rms_norm import RMSNorm
from keras_mml.utils.array import as_numpy, decode_ternary_array, encode_ternary_array

EPSILON = 1e-5
HUGE = 1e9


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
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

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

    # Properties
    @property
    def _quantized_weights_for_saving(self) -> Tuple[Any, float]:
        """
        Returns a tuple. The first value is the quantized weights, and the second is the scale.
        """

        scale = 1.0 / ops.clip(ops.mean(ops.abs(self.w)), EPSILON, HUGE)
        u = ops.clip(ops.round(self.w * scale), -1, 1)

        return u, scale

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

        scale = 127.0 / ops.clip(ops.max(ops.abs(x), axis=-1, keepdims=True), EPSILON, HUGE)
        y = ops.clip(ops.round(x * scale), -128, 127) / scale
        return y

    @staticmethod
    def _weights_quantization(w) -> Any:
        """
        Quantizes the weights to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

        Args:
            w: Array of weights.

        Returns:
            The quantized weights.
        """

        scale = 1.0 / ops.clip(ops.mean(ops.abs(w)), EPSILON, HUGE)
        u = ops.clip(ops.round(w * scale), -1, 1) / scale
        return u

    def _get_quantized_arrays(self, x_norm) -> Tuple[Any, Any]:
        """
        Gets the quantized activation and weight values.

        Args:
            x_norm: Normalized activation values.

        Returns:
            A tuple. The first value is the quantized activation values. The second is the quantized
            weight values.
        """

        # Get the quantized activations and weights
        # (We use a Straight-Through Estimator (STE) trick by stopping gradient propagation)
        x_quantized = x_norm + ops.stop_gradient(self._activations_quantization(x_norm) - x_norm)

        if self._weight_scale:
            # Weights should have been pre-quantized
            w_quantized = self.w / self._weight_scale
        else:
            w = self.w
            w_quantized = w + ops.stop_gradient(self._weights_quantization(w) - w)

        return x_quantized, w_quantized

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
            name="weights",
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

        # Get the quantized arrays
        x_quantized, w_quantized = self._get_quantized_arrays(x_norm)

        # Perform kernel operation
        # TODO: Make this more efficient when we are doing inference only
        x = ops.matmul(x_quantized, w_quantized)  # The `matmul` should just involve addition and subtraction

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
        w_quantized, w_scale = self._quantized_weights_for_saving

        # Encode the ternary weights efficiently
        shape, encoded = encode_ternary_array(as_numpy(w_quantized))

        # Then store the variables
        store["encoded"] = np.frombuffer(encoded, dtype="uint8")
        store["shape"] = shape
        store["scale"] = w_scale

    def load_own_variables(self, store: Dict):
        """
        Loads the state of the layer.

        Args:
            store: Dictionary from which the state of the model will be loaded.

        Raises:
            ValueError: If the layer is missing variables when loading from a file.
        """

        # Get the variables from the store first
        try:
            encoded = store["encoded"][()].tobytes()
            shape = store["shape"][()]
            scale = store["scale"][()]
        except ValueError:  # pragma: no cover
            raise ValueError("DenseMML layer missing values when loading from file")

        # Then recover the weights
        self.w.assign(decode_ternary_array(shape, encoded))
        self._weight_scale = scale
