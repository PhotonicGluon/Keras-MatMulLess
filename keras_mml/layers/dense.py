"""
Implements a matmul-less Dense layer.
"""

from typing import Tuple

import keras
from keras import ops

from keras_mml.layers.rms_norm import RMSNorm

EPSILON = 1e-5
HUGE = 1e9


@keras.saving.register_keras_serializable(package="keras_mml")
class DenseMML(keras.Layer):
    """
    Dense layer without matrix multiplication.

    Specifically, this is the ``BitLinear`` layer described in https://arxiv.org/pdf/2310.11453. It
    uses bit quantization to reduce matrix multiplication operations to simple addition and 
    subtraction.

    This implementation only allows the layer to have a rank of 2. That is, the input to this layer 
    must be of the form ``(batch_size, d0)``. An input shape that does not conform to this will 
    raise a :py:exc:`ValueError`.

    Attributes:
        units: Dimensionality of the output space.
    """

    def __init__(self, units:int, **kwargs):
        """
        Initializes a new :py:class:`~DenseMML` layer.

        Args:
            units: Dimensionality of the output space.
        """

        super().__init__(**kwargs)
        self.units = units

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
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
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
        y = ops.matmul(x_quantized, w_quantized)  # This, in theory, just involves addition and subtraction
        return y
