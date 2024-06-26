"""
Implements embedding layers.
"""

from typing import Tuple

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="keras_mml")
class TokenEmbedding(keras.layers.Layer):
    """
    TODO: ADD
    """

    def __init__(self, max_len: int, vocab_size: int, embedding_dim: int, with_positions: bool = False, **kwargs):
        """
        TODO: ADD
        """

        if max_len <= 0:
            raise ValueError(
                "Received an invalid value for the maximum sequence length. "
                f"Expected a positive integer, but got {max_len} instead."
            )

        if vocab_size <= 0:
            raise ValueError(
                "Received an invalid value for the vocabulary size. "
                f"Expected a positive integer, but got {vocab_size} instead."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "Received an invalid value for the embedding dimension. "
                f"Expected a positive integer, but got {embedding_dim} instead."
            )

        super().__init__(**kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=2)

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.with_positions = with_positions

        self.token_embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)

        if with_positions:
            self.pos_embedding = keras.layers.Embedding(input_dim=max_len, output_dim=embedding_dim)
        else:
            self.pos_embedding = None

    def build(self, input_shape: Tuple[int, int]):
        """
        Build the layer.

        Args:
            input_shape: Shape of the input.
        """

        super().build(input_shape)

        self.token_embedding.build(input_shape)
        if self.pos_embedding is not None:
            self.pos_embedding.build(input_shape)

    def call(self, inputs):
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer.

        Returns:
            Transformed inputs.
        """

        tokens = self.token_embedding(inputs)
        if self.pos_embedding is None:
            return tokens

        max_len = ops.shape(inputs)[-1]
        positions = ops.arange(start=0, stop=max_len, step=1)
        positions = self.pos_embedding(positions)

        return tokens + positions
