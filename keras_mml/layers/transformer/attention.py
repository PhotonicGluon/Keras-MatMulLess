"""
Implements a matmul-less attention layer.
"""

from typing import Tuple

import keras
import numpy as np
from jaxtyping import Float

from keras_mml.layers.recurrent.gru import GRUMML


@keras.saving.register_keras_serializable(package="keras_mml")
class AttentionMML(keras.Layer):
    """
    Multi-headed attention layer that is mostly without matrix multiplications.

    Unlike the Keras implementation, this is **not** an implementation of multi-headed attention in
    the |AttentionPaper|_ paper. Rather, this layer follows the description of the token-mixer in
    |MatMulFreeLLM|_ (see section 3.3.1), where we use
    :py:class:`~keras_mml.layers.recurrent.GRUMML` as the attention mechanism.

    Attributes:
        num_heads: Number of attention heads.
        out_dim: Output dimension.
        fully_mml: Whether to use full matmul-less layers in the attention mechanism.

    .. |AttentionPaper| replace:: *Attention Is All You Need*
    .. _AttentionPaper: https://arxiv.org/pdf/1706.03762v7
    .. |MatMulFreeLLM| replace:: *Scalable MatMul-free Language Modeling*
    .. _MatMulFreeLLM: https://arxiv.org/pdf/2406.02528v5
    """

    def __init__(self, num_heads: int, out_dim: int, fully_mml: bool = True, **kwargs):
        """
        Initializes a new instance of the layer.

        Args:
            num_heads: Number of attention heads.
            out_dim: Output dimension.
            fully_mml: Whether to use full matmul-less layers in the attention mechanism.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the number of heads is not a positive integer.
            ValueError: If the output dimension is not a positive integer.
        """

        if num_heads <= 0:
            raise ValueError(
                f"Received an invalid value for the number of heads, expected a positive integer, got {num_heads}."
            )

        if out_dim <= 0:
            raise ValueError(
                f"Received an invalid value for the output dimension, expected a positive integer, got {out_dim}."
            )

        super().__init__(**kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=3)

        # Main attributes
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.fully_mml = fully_mml

        # Hidden weights/layers
        self._internal_layer = GRUMML(
            out_dim,
            fully_mml=fully_mml,
            num_heads=num_heads,
            activation="silu",
            recurrent_activation="sigmoid",
            return_sequences=True,
        )

    def build(self, input_shape: Tuple[int, int, int]):
        """
        Build the layer.

        Args:
            input_shape: Shape of the input.
        """

        super().build(input_shape)
        self._internal_layer.build(input_shape)

    def call(
        self, inputs: Float[np.ndarray, "batch_size sequence_length features"]
    ) -> Float[np.ndarray, "batch_size sequence_length out_dim"]:
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer.

        Returns:
            Transformed inputs.
        """

        return self._internal_layer(inputs)

    def compute_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Computes the output shape of the layer.

        Args:
            input_shape: Shape of the input into the layer.

        Returns:
            Shape of the output.
        """

        return self._internal_layer.compute_output_shape(input_shape)
