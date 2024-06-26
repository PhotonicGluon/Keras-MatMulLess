"""
Implements a matmul-less transformer block.
"""

from typing import Tuple

import keras

from keras_mml.layers.activations import SwiGLUMML
from keras_mml.layers.normalizations.rms_norm import RMSNorm
from keras_mml.layers.transformer.attention import AttentionMML


@keras.saving.register_keras_serializable(package="keras_mml")
class TransformerBlockMML(keras.Layer):
    """
    TODO: ADD
    """

    def __init__(
        self,
        embedding_dim: int,
        ffn_dim: int,
        num_heads: int,
        hidden_ratio: int = 4,
        rate: float = 0.1,
        **kwargs,
    ):  # TODO: Add
        """
        TODO: ADD
        """

        if embedding_dim <= 0:
            raise ValueError(
                f"Received an invalid value for embedding dimension, expected a positive integer, got {embedding_dim}"
            )

        if ffn_dim <= 0:
            raise ValueError(
                "Received an invalid value for the feed forward network dimension, "
                f"expected a positive integer, got {ffn_dim}"
            )

        if num_heads <= 0:
            raise ValueError(
                f"Received an invalid value for the number of heads, expected a positive integer, got {num_heads}"
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by the number of heads. "
                f"Got embedding dimension of {embedding_dim} but wanted to use {num_heads} heads."
            )

        super().__init__(**kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=3)

        self.embedding_dim = embedding_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.hidden_ratio = hidden_ratio
        self.rate = rate

        self.attention = AttentionMML(num_heads, embedding_dim, fully_mml=True)
        self.attention_dropout = keras.layers.Dropout(rate)
        self.attention_norm = RMSNorm()

        self.ffn = SwiGLUMML(embedding_dim, hidden_ratio=hidden_ratio, intermediate_size=ffn_dim)
        self.ffn_dropout = keras.layers.Dropout(rate)
        self.ffn_norm = RMSNorm()

    def build(self, input_shape: Tuple[int, int, int]):
        """
        Build the layer.

        Args:
            input_shape: Shape of the input.
        """

        super().build(input_shape)

    def call(self, inputs):
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer.

        Returns:
            Transformed inputs.
        """

        attention_output = self.attention(inputs)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_norm(inputs + attention_output)

        ffn_output = self.ffn(attention_output)
        ffn_output = self.ffn_dropout(ffn_output)
        ffn_output = self.ffn_norm(attention_output + ffn_output)
        return ffn_output
