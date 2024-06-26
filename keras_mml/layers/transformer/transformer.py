"""
Implements a matmul-less transformer block.
"""

from typing import Optional

import keras

from keras_mml.layers.activations import SwiGLUMML
from keras_mml.layers.normalizations.rms_norm import RMSNorm
from keras_mml.layers.recurrent.gru import GRUMML


@keras.saving.register_keras_serializable(package="keras_mml")
class TransformerBlockMML(keras.Layer):
    """
    TODO: ADD
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_ratio: int = 4,
        rate: float = 0.1,
        intermediate_size: Optional[int] = None,
        activation: str = "sigmoid",
        **kwargs,
    ):  # TODO: Add
        """
        TODO: ADD
        """

        super().__init__(**kwargs)

        if embedding_dim <= 0:
            raise ValueError(
                f"Received an invalid value for embedding dimension, expected a positive integer, got {embedding_dim}"
            )

        self.embedding_dim = embedding_dim

        self.attention = GRUMML(embedding_dim, fully_mml=True)
        self.attention_dropout = keras.layers.Dropout(rate)
        self.attention_norm = RMSNorm()

        self.ffn = SwiGLUMML(
            embedding_dim, hidden_ratio=hidden_ratio, intermediate_size=intermediate_size, activation=activation
        )
        self.ffn_dropout = keras.layers.Dropout(rate)
        self.ffn_norm = RMSNorm()

    def call(self, inputs):
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer.

        Returns:
            Transformed inputs.
        """

        attn_output = self.attention(inputs, inputs)
        attn_output = self.attention_dropout(attn_output)
        attn_output = self.attention_norm(inputs + attn_output)

        ffn_output = self.ffn(attn_output)
        ffn_output = self.ffn_dropout(ffn_output)
        return self.ffn_norm(attn_output + ffn_output)
