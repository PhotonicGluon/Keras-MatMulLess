"""
Implements embedding layers.
"""

from typing import Tuple

import keras
import numpy as np
from jaxtyping import Float
from keras import ops

from keras_mml.layers.core.dense import DenseMML


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

        # Main attributes
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.with_positions = with_positions

        # Hidden weights/layers
        self._token_embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)

        if with_positions:
            self._pos_embedding = keras.layers.Embedding(input_dim=max_len, output_dim=embedding_dim)
        else:
            self._pos_embedding = None

    def build(self, input_shape: Tuple[int, int]):
        """
        Build the layer.

        Args:
            input_shape: Shape of the input.
        """

        self._token_embedding.build(input_shape)
        if self._pos_embedding is not None:
            self._pos_embedding.build(input_shape)

        self.built = True

    def call(
        self, inputs: Float[np.ndarray, "batch_size sequence_len"]
    ) -> Float[np.ndarray, "batch_size sequence_len embedding_dim"]:
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer.

        Returns:
            Transformed inputs.
        """

        tokens = self._token_embedding(inputs)
        if self._pos_embedding is None:
            return tokens

        max_len = ops.shape(inputs)[-1]
        positions = ops.arange(start=0, stop=max_len, step=1)
        positions = self._pos_embedding(positions)

        return tokens + positions


@keras.saving.register_keras_serializable(package="keras_mml")
class PatchEmbedding(keras.layers.Layer):
    """
    Turns image patches into vectors of fixed size.

    The image patches should have come from the :py:class:`~keras_mml.layers.misc.Patches` layer.

    This layer could optionally include position information in the embeddings by enabling the
    :py:attr:`with_positions` attribute.

    Attributes:
        num_patches: Number of patches in each image.
        embedding_dim: Embedding dimension.
        use_mml: Whether to use a matmul-less projection to embed the patches.
        with_positions: Whether to include position information in the embeddings.
    """

    def __init__(
        self, num_patches: int, embedding_dim: int, use_mml: bool = True, with_positions: bool = False, **kwargs
    ):
        """
        Initializes a new instance of the layer.

        Args:
            num_patches: Number of patches in each image.
            embedding_dim: Embedding dimension.
            use_mml: Whether to use a matmul-less projection to embed the patches.
            with_positions: Whether to include position information in the embeddings.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the number of patches is not a positive integer.
            ValueError: If the embedding dimension is not a positive integer.
        """

        if num_patches <= 0:
            raise ValueError(f"Invalid number of patches, expected a positive integer, got {num_patches}")

        if embedding_dim <= 0:
            raise ValueError(f"Invalid embedding dimension, expected a positive integer, got {embedding_dim}")

        super().__init__(**kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=3)

        # Main attributes
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.use_mml = use_mml
        self.with_positions = with_positions

        # Hidden weights/layers
        if self.use_mml:
            self._projection = DenseMML(embedding_dim)
        else:
            self._projection = keras.layers.Dense(embedding_dim)

        if with_positions:
            self._pos_embedding = keras.layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)
        else:
            self._pos_embedding = None

    def build(self, input_shape: Tuple[int, int, int]):
        """
        Build the layer.

        Args:
            input_shape: Shape of the input.
        """

        self._projection.build(input_shape)
        if self._pos_embedding is not None:
            self._pos_embedding.build(input_shape)

        self.built = True

    def call(
        self, inputs: Float[np.ndarray, "batch_size patch_count patch_dim"]
    ) -> Float[np.ndarray, "batch_size patch_count embedding_dim"]:
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer.

        Returns:
            Transformed inputs.
        """

        projected_patches = self._projection(inputs)
        if self._pos_embedding is None:
            return projected_patches

        positions = ops.expand_dims(ops.arange(start=0, stop=self.num_patches, step=1), axis=0)
        positions = self._pos_embedding(positions)

        return projected_patches + positions
