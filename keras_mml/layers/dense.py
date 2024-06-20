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

        self._beta = None  # Used for when the layer is loaded from file

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

        absolutes = ops.abs(x)
        maxima = ops.max(absolutes, axis=-1, keepdims=True)
        clipped = ops.clip(maxima, EPSILON, HUGE)  # This is gamma in the paper
        scale = 127.0 / clipped

        rescaled = x * scale
        rounded = ops.round(rescaled)
        clipped = ops.clip(rounded, -128, 127)

        return clipped, scale

    @staticmethod
    def _weights_quantization(w) -> Tuple[Any, Any]:
        """
        Quantizes the weights to 1-bit precision.

        This is equation (1) in the aforementioned paper.

        Args:
            w: Array of weights.

        Returns:
            A tuple. The first is the quantized weights. The second is the scaling factor needed in
            equation (11) in the paper.
        """

        absolutes = ops.abs(w)
        scale = ops.mean(absolutes)
        alpha = ops.mean(w)

        signs = ops.sign(w - alpha)

        return signs, scale

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

        x_quantized, gamma_qb = self._activations_quantization(x_norm)
        w_quantized, beta = self._weights_quantization(self.w)

        if self._beta is not None:  # Using a saved layer
            # FIXME: This doesn't work for saved layers
            beta = self._beta

        scaling = beta / gamma_qb  # See eq. (11)
        x = ops.matmul(x_quantized, w_quantized) * scaling  # The `matmul` should just involve addition and subtraction

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

        all_vars = self._trainable_variables + self._non_trainable_variables
        for i, v in enumerate(all_vars):
            if v.name == WEIGHTS_NAME:
                # Pre-quantize the weights
                w_quantized, beta = self._weights_quantization(v)

                # For more efficient saving, convert the weights to boolean values, where 1 is True and -1 (and 0) are
                # False
                w_bools = w_quantized > 0
                store[f"{i}"] = w_bools
                store[f"{i}-beta"] = beta
            else:  # pragma: no cover
                store[f"{i}"] = v

    def load_own_variables(self, store: Dict):
        """
        Loads the state of the layer.

        Args:
            store: Dictionary from which the state of the model will be loaded.

        Raises:
            ValueError: If the layer was never built but the weights file lists variables for the
                layer.
            ValueError: If the expected number of variables for the layer does not match the number
                of variables provided by the weights file.
        """

        all_vars = self._trainable_variables + self._non_trainable_variables
        expected_count = len(all_vars)
        for v in all_vars:
            if v.name == WEIGHTS_NAME:
                expected_count += 1

        if len(store.keys()) != expected_count:
            if expected_count == 0 and not self.built:  # pragma: no cover
                raise ValueError(
                    f"Layer '{self.name}' was never built and thus it doesn't have any variables. "
                    f"However the weights file lists {len(store.keys())} "
                    "variables for this layer.\n"
                    "In most cases, this error indicates that either:\n\n"
                    "1. The layer is owned by a parent layer that "
                    "implements a `build()` method, but calling the "
                    "parent's `build()` method did NOT create the state of "
                    f"the child layer '{self.name}'. A `build()` method "
                    "must create ALL state for the layer, including "
                    "the state of any children layers.\n\n"
                    "2. You need to implement "
                    "the `def build_from_config(self, config)` method "
                    f"on layer '{self.name}', to specify how to rebuild "
                    "it during loading. "
                    "In this case, you might also want to implement the "
                    "method that generates the build config at saving time, "
                    "`def get_build_config(self)`. "
                    "The method `build_from_config()` is meant "
                    "to create the state "
                    "of the layer (i.e. its variables) upon deserialization.",
                )
            raise ValueError(  # pragma: no cover
                f"Layer '{self.name}' expected {expected_count} variables, "
                "but received "
                f"{len(store.keys())} variables during loading. "
                f"Expected: {[v.name for v in all_vars]}"
            )

        for i, v in enumerate(all_vars):
            if v.name == WEIGHTS_NAME:
                # Get the quantized weights and the scale
                w_quantized = store[f"{i}"][()]  # These are HDF5 dataset objects
                beta = store[f"{i}-beta"][()]

                # Pre-multiply them to get back the correct weights
                w = w_quantized * beta
                v.assign(w)

                # Update the beta value to signal that we are using a saved weight
                self._beta = beta

            else:  # pragma: no cover
                v.assign(store[f"{i}"])
