"""
Implements a matmul-less Dense layer.
"""

from typing import Any, Dict, Optional, Tuple, Union

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
    Dense layer without matrix multiplications.

    The core of the layer is the ``BitLinear`` layer described in |1.58 Bit LLMs|_. It uses ternary
    quantization to reduce matrix multiplication operations to simple addition and subtraction.

    This implementation differs from ``BitLinear`` by allowing an activation function to be
    specified. More precisely, :py:class:`~DenseMML` implements the operation

    .. math::
        \\mathbf{y} = \\sigma\\left(\\mathbf{x}\\mathbf{W}^\\intercal + \\mathbf{b}\\right)

    where :math:`\\mathbf{x}` is the quantized input vector, :math:`\\mathbf{W}` is the quantized
    weights matrix, :math:`\\mathbf{b}` is the bias vector, and :math:`\\sigma` is the element-wise
    activation function.

    This implementation only allows the layer to have a rank of 2. That is, the input to this layer
    must be of the form ``(batch_size, d0)``.

    .. WARNING::
       Once a model that uses this layer is loaded from a file, it **cannot** be retrained.

    Attributes:
        units: Dimensionality of the output space.
        use_bias: Whether the layer uses a bias vector.
        kernel_initializer: Initializer for the weights matrix.
        bias_initializer: Initializer for the bias vector.

    .. |1.58 Bit LLMs| replace:: *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*
    .. _1.58 Bit LLMs: https://arxiv.org/pdf/2402.17764
    """

    def __init__(
        self,
        units: int,
        activation: Optional[str] = None,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **kwargs,
    ):
        """
        Initializes a new :py:class:`~DenseMML` layer.

        Args:
            units: Dimensionality of the output space.
            activation: Activation function to use. If you don't specify anything, no activation is
                applied (i.e. "linear" activation: :math:`\\sigma(\\mathbf{x}) = \\mathbf{x}`).
            use_bias: Whether the layer uses a bias vector.
            kernel_initializer: Initializer for the weights matrix.
            bias_initializer: Initializer for the bias vector.
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
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self._weight_scale = None  # Used for when the layer is loaded from file

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
    def _weights_quantization(w, for_saving: bool = False) -> Union[Any, Tuple[Any, float]]:
        """
        Quantizes the weights to 1.58 bits (i.e., :math:`\\log_{2}3` bits).

        Args:
            w: Array of weights.
            for_saving: Whether this should output values in preparation for saving.

        Returns:
            The quantized weights with the scaling applied. If the quantization is for saving, then
            both the quantized weights and the scale will be returned, with the scale **not**
            applied to the quantized weights.
        """

        scale = 1.0 / ops.clip(ops.mean(ops.abs(w)), EPSILON, HUGE)
        u = ops.clip(ops.round(w * scale), -1, 1)

        if for_saving:
            return u, scale
        return u / scale

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
            initializer=self.kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=(self.units,), initializer=self.bias_initializer, trainable=True
            )
        else:
            self.bias = None

        self.built = True

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

        # Then apply bias and activation
        if self.bias is not None:
            x = ops.add(x, self.bias)
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

    def save_own_variables(self, store: Dict):
        """
        Saves the state of the layer.

        Args:
            store: Dictionary where the state of the model will be saved.
        """

        # Pre-quantize the weights
        w_quantized, w_scale = self._weights_quantization(self.w, for_saving=True)

        # Encode the ternary weights efficiently
        shape, encoded = encode_ternary_array(as_numpy(w_quantized))

        # Then store the variables
        store["kernel_encoded"] = np.frombuffer(encoded, dtype="uint8")
        store["kernel_shape"] = shape
        store["kernel_scale"] = w_scale
        store["bias"] = self.bias

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
            encoded = store["kernel_encoded"][()].tobytes()
            shape = store["kernel_shape"][()]
            scale = store["kernel_scale"][()]
            bias = store["bias"][()]
        except ValueError:  # pragma: no cover
            raise ValueError("DenseMML layer missing values when loading from file")

        # Then recover the weights
        self.w.assign(decode_ternary_array(shape, encoded))
        self._weight_scale = scale

        self.bias = bias
