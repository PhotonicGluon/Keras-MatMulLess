"""
Implements a matmul-less Dense layer.
"""

from typing import Any, Dict, Optional, Tuple

import keras
from keras import activations, initializers, ops

from keras_mml.layers.rms_norm import RMSNorm

EPSILON = 1e-5
HUGE = 1e9


@keras.saving.register_keras_serializable(package="keras_mml")
class DenseMML(keras.Layer):
    """
    Dense layer without matrix multiplication.

    The core of the layer is the ``BitLinear`` layer described in https://arxiv.org/pdf/2310.11453.
    It uses bit quantization to reduce matrix multiplication operations to simple addition and
    subtraction.

    This implementation differs from ``BitLinear`` by allowing an activation function to be
    specified. More precisely, :py:class:`~DenseMML` implements the operation
    :math:`\\mathbf{y} = \\sigma\\left(\\mathbf{x}\\mathbf{W}^\\intercal\\right)` where
    :math:`\\mathbf{x}` is the quantized input vector, :math:`\\mathbf{W}` is the quantized weights
    matrix, and :math:`\\sigma` is the element-wise activation function.

    This implementation only allows the layer to have a rank of 2. That is, the input to this layer
    must be of the form ``(batch_size, d0)``. An input shape that does not conform to this will
    raise a :py:exc:`ValueError`.

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
        """

        super().__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.weights_initializer = initializers.get(weights_initializer)

    # Helper methods
    @staticmethod
    def _activations_quantization(x):
        """
        Quantizes the activations to 8-bit precision using absmax quantization.

        This is equation (4) in the aforementioned paper.

        Also pre-undoes part of the scaling in (11) by dividing the clipped values by the
        scale.

        Args:
            x: Array of quantization values.

        Returns:
            Quantized activation values.
        """

        absolutes = ops.abs(x)
        maxima = ops.max(absolutes, axis=-1, keepdims=True)
        clipped = ops.clip(maxima, EPSILON, HUGE)  # This is gamma in the paper
        scale = 127.0 / clipped

        rescaled = x * scale
        rounded = ops.round(rescaled)
        clipped = ops.clip(rounded, -128, 127)

        y = clipped / scale  # Perform part of the undoing in eq. (11)
        return y

    @staticmethod
    def _weights_quantization(w):
        """
        Quantizes the weights to 1-bit precision.

        This is equation (1) in the aforementioned paper.

        Also pre-undoes part of the scaling in (11) by multiplying the weights by the scale.

        Args:
            w: Array of weights.

        Returns:
            Quantized weights.
        """

        absolutes = ops.abs(w)
        scale = ops.mean(absolutes)
        alpha = ops.mean(w)

        signs = ops.sign(w - alpha)

        u = signs * scale
        return u

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

        if len(input_shape) != 2:
            raise ValueError(f"DenseMML input shape must have rank 2 (received: {input_shape})")

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

        x_quantized = self._activations_quantization(x_norm)
        w_quantized = self._weights_quantization(self.w)
        x = ops.matmul(x_quantized, w_quantized)  # This, in theory, just involves addition and subtraction

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
