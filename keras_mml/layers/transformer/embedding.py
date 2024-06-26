"""
Implements embedding layers.
"""

from typing import Tuple

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="keras_mml")
class TokenEmbedding(keras.layers.Layer):
    """
    Turns positive integers (indices) into vectors of fixed size.

    For example, ``[[1, 2], [3, 4], [5, 6]]``, which could be interpreted as 3 sentences with 2
    words each, could be embedded as ``[[[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]], [[1.1, 1.2, 1.3],
    [1.3, 1.4, 1.5]], [[2.1, 2.2, 2.3], [2.3, 2.4, 2.5]]]``, which has shape ``(3, 2, 3)`` and can
    be interpreted as 3 sentences with 2 words each with an embedding dimension of 3.

    This layer could optionally include position information in the embeddings by enabling the
    :py:attr:`with_positions` attribute.

    .. admonition:: Calling Convention
        :class: tip

        - **Input Shape**: 2D tensor of shape ``(batch_size, sequence_length)``
        - **Output Shape**: ``(batch_size, sequence_length, embedding_dim)``

    Attributes:
        max_len: Maximum length of a sentence.
        vocab_size: Size of the vocabulary. Typically this is one more than the maximum integer
            index.
        embedding_dim: Embedding dimension.
        with_positions: Whether to include position information in the embeddings.
    """

    def __init__(self, max_len: int, vocab_size: int, embedding_dim: int, with_positions: bool = False, **kwargs):
        """
        Initializes a new instance of the layer.

        Args:
            max_len: Maximum length of a sentence.
            vocab_size: Size of the vocabulary. Typically this is one more than the maximum integer
                index.
            embedding_dim: Embedding dimension.
            with_positions: Whether to include position information in the embeddings.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the maximum sentence length is not a positive integer.
            ValueError: If the vocabulary size is not a positive integer.
            ValueError: If the embedding dimension is not a positive integer.
        """

        if max_len <= 0:
            raise ValueError(
                "Received an invalid value for the maximum sentence length. "
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
