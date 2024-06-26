"""
Root Mean Square Normalization (RMSNorm) implementation.
"""

from typing import Optional, Tuple

import keras
from keras import constraints, initializers, ops, regularizers


@keras.saving.register_keras_serializable(package="keras_mml")
class RMSNorm(keras.Layer):
    """
    Implements Root Mean Square Normalization (RMSNorm).

    The implementation of RMSNorm follows |RMSNorm Paper|_.

    .. admonition:: Calling Convention
        :class: tip

        - **Input Shape**: ``(batch_size, d1, ..., dn)``, i.e., allows any shape.
        - **Output Shape**: ``(batch_size, d1, ..., units)``

    Attributes:
        has_learnable_weights: Whether the layer has learnable per-element affine parameters.
        use_bias: Whether the layer uses a bias vector.
        gain_initializer: Initializer for the gain weights.
        bias_initializer: Initializer for the bias vector.
        gain_regularizer: Regularizer for the gain weights.
        bias_regularizer: Regularizer for the bias vector.
        gain_constraint: Constraint for the gain weights.
        bias_constraint: Constraint for the bias vector.
        scale: Scaling factor. Available only after layer is built.

    .. |RMSNorm Paper| replace:: *Root Mean Square Layer Normalization*
    .. _RMSNorm Paper: https://arxiv.org/pdf/1910.07467v1
    """

    def __init__(
        self,
        has_learnable_weights: bool = True,
        use_bias: bool = False,
        gain_initializer: str = "ones",
        bias_initializer: str = "zeros",
        gain_regularizer: Optional[str] = None,
        bias_regularizer: Optional[str] = None,
        gain_constraint: Optional[str] = None,
        bias_constraint: Optional[str] = None,
        **kwargs
    ):
        """
        Initializes a new RMSNorm instance.

        Args:
            has_learnable_weights: When set to True, this layer has learnable per-element affine
                parameters initialized to ones (for weights, a.k.a. for gains) and zeros (for
                biases).
            use_bias: Whether the layer uses a bias vector. Ignored if
                :py:attr:`~has_learnable_weights` is False.
            gain_initializer: Initializer for the gain weights.
            bias_initializer: Initializer for the bias vector.
            gain_regularizer: Regularizer for the gain weights.
            bias_regularizer: Regularizer for the bias vector.
            gain_constraint: Constraint for the gain weights.
            bias_constraint: Constraint for the bias vector.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.
        """

        super().__init__(**kwargs)

        self.has_learnable_weights = has_learnable_weights
        self.use_bias = use_bias

        self.gain_initializer = initializers.get(gain_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.gain_regularizer = regularizers.get(gain_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.gain_constraint = constraints.get(gain_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape: Tuple[int, ...]):
        """
        Create layer weights.

        Args:
            input_shape: Shape of the input.
        """

        dim = input_shape[-1]
        self.scale = dim**0.5

        if self.has_learnable_weights:
            self.gain = self.add_weight(
                input_shape[1:],
                initializer=self.gain_initializer,
                regularizer=self.gain_regularizer,
                constraint=self.gain_constraint,
                name="gain",
            )
            if self.use_bias:
                self.bias = self.add_weight(
                    input_shape[1:],
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name="bias",
                )
            else:
                self.bias = None
        else:
            self.gain = None
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

        output = ops.normalize(inputs, order=2, axis=-1) * self.scale
        if self.gain is not None:
            output *= self.gain
        if self.bias is not None:
            output += self.bias
        return output
