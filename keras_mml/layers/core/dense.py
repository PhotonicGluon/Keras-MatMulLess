"""
Implements a matmul-less Dense layer.
"""

from typing import Dict, Optional, Tuple

import keras
import numpy as np
from einops import asnumpy
from jaxtyping import Float
from keras import activations, constraints, initializers, ops, regularizers

from keras_mml.layers.core._dense_impl import BackendDenseMML
from keras_mml.layers.normalizations.rms_norm import RMSNorm
from keras_mml.utils.array.encoding import decode_ternary_array, encode_ternary_array

EPSILON = 1e-5
HUGE = 1e9


@keras.saving.register_keras_serializable(package="keras_mml")
class DenseMML(BackendDenseMML):
    """
    Dense layer without matrix multiplications.

    The core of the layer is the ``BitLinear`` layer described in |1.58 Bit LLMs|_. It uses ternary
    quantization to reduce matrix multiplication operations to simple addition and subtraction.

    This implementation differs from ``BitLinear`` by allowing an activation function to be
    specified. More precisely, :py:class:`~DenseMML` implements the operation

    .. math::
        \\mathbf{y} = \\sigma\\left(\\mathbf{x}\\mathbf{W}^\\intercal + \\mathbf{b}\\right)

    where :math:`\\mathbf{x}` is the quantized input vector, :math:`\\mathbf{W}` is the quantized
    weights matrix (i.e., the *kernel* matrix), :math:`\\mathbf{b}` is the bias vector, and
    :math:`\\sigma` is the element-wise activation function.

    .. IMPORTANT::
       See the :ref:`pitfalls <explanations/dense:pitfalls>` when using this layer.

    .. NOTE::
       If the input to the layer (say :math:`\\mathbf{x}`) has a rank greater than 2, then this
       computes the dot product of :math:`\\mathbf{x}` and :math:`\\mathbf{W}` along the last axis
       of the :math:`\\mathbf{x}` and axis ``0`` of :math:`\\mathbf{W}`.

       For example, suppose :math:`\\mathbf{x}` has shape ``(batch_size, d0, d1)``. Then
       :math:`\\mathbf{W}` is created to have shape ``(d1, units)`` and it operates along axis ``2``
       of :math:`\\mathbf{x}` on every sub-tensor of shape ``(1, 1, d1)`` (there are
       ``batch_size * d0`` such sub-tensors). The output in this case will have shape
       ``(batch_size, d0, units)``.

    .. WARNING::
       Once a model that uses this layer is loaded from a file, it **cannot** be retrained.

    Attributes:
        units: Dimensionality of the output space.
        use_bias: Whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the kernel matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.

    .. |1.58 Bit LLMs| replace:: *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*
    .. _1.58 Bit LLMs: https://arxiv.org/pdf/2402.17764v1
    """

    def __init__(
        self,
        units: int,
        activation: Optional[str] = None,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[str] = None,
        bias_regularizer: Optional[str] = None,
        activity_regularizer: Optional[str] = None,
        kernel_constraint: Optional[str] = None,
        bias_constraint: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes a new :py:class:`~DenseMML` layer.

        Args:
            units: Dimensionality of the output space.
            activation: Activation function to use. If you don't specify anything, no activation is
                applied (i.e. "linear" activation: :math:`\\sigma(\\mathbf{x}) = \\mathbf{x}`).
            use_bias: Whether the layer uses a bias vector.
            kernel_initializer: Initializer for the kernel matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer function applied to the kernel matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to the output of the layer (i.e., its
                activation).
            kernel_constraint: Constraint function applied to the kernel matrix.
            bias_constraint: Constraint function applied to the bias vector.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the units provided is not a positive integer.
        """

        if units <= 0:
            raise ValueError(
                f"Received an invalid value for argument `units`, expected a positive integer, got {units}"
            )

        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        self.input_spec = keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True

        # Main attributes
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self._kernel_scale = None  # Used for when the layer is loaded from file

        # Hidden weights/layers
        self._activation_norm = None
        self._kernel = None
        self._bias = None

    # Public methods
    def build(self, input_shape: Tuple[int, ...]):
        """
        Create layer weights.

        Args:
            input_shape: Shape of the input.
        """

        input_dim = input_shape[-1]

        self._activation_norm = RMSNorm()
        self._activation_norm.build(input_shape)

        self._kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self._bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )

        self.input_spec = keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(
        self, inputs: Float[np.ndarray, "batch_size *dims last_dim"]
    ) -> Float[np.ndarray, "batch_size *dims units"]:
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer.

        Returns:
            Transformed inputs.
        """

        # First normalize the inputs
        x_norm = self._activation_norm(inputs)

        # Get the quantized arrays
        x_quantized, w_quantized, w_scale = self._get_quantized_arrays(x_norm)

        # Perform ternary multiplication
        x = self._ternary_multiplication(x_quantized, w_quantized, w_scale)

        # Then apply bias and activation
        if self._bias is not None:
            x = ops.add(x, self._bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
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

        # Pre-quantize the kernel values
        w_scale = self._compute_kernel_scale(self._kernel)
        w_quantized = self._quantize_kernel(self._kernel, w_scale)

        # Encode the ternary array efficiently
        shape, encoded = encode_ternary_array(asnumpy(w_quantized))

        # Then store the variables
        store["kernel_encoded"] = np.frombuffer(encoded, dtype="uint8")
        store["kernel_shape"] = shape
        store["kernel_scale"] = w_scale

        if self.use_bias:
            store["bias"] = self._bias

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
            w_scale = store["kernel_scale"][()]

            if self.use_bias:
                bias = store["bias"][()]
            else:
                bias = None
        except ValueError as e:  # pragma: no cover
            raise ValueError("DenseMML layer missing values when loading from file") from e

        # Then recover the weights
        self._kernel.assign(decode_ternary_array(shape, encoded))
        self._kernel_scale = w_scale

        self._bias = bias
