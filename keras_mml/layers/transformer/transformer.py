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
    Transformer block layer that is mostly without matrix multiplications.

    The core flow of the transformer block follows the |AttentionPaper|_ paper, while referencing
    the Keras example |KerasTransformer|_ for its high-level implementation. However, we use the
    custom :py:class:`~keras_mml.layers.transformer.AttentionMML` class for the attention mechanism
    and :py:class:`~keras_mml.layers.activations.SwiGLUMML` for the feed-forward network (FFN) part.

    .. admonition:: Calling Convention
        :class: tip

        - **Input Shape**: 3D tensor of shape ``(batch_size, sequence_length, features)``
        - **Output Shape**: ``(batch_size, sequence_length, embedding_dim)``

    Attributes:
        embedding_dim: Dimension of the embeddings.
        ffn_dim: Dimension of the intermediate (i.e., hidden) layer of the feed-forward network.
        num_heads: Number of heads to use for multi-headed attention.
        fully_mml: Whether to use full matmul-less layers in the attention mechanism.
        rate: Dropout rate to apply for the attention mechanism and the feed-forward network.

    .. |AttentionPaper| replace:: *Attention Is All You Need*
    .. _AttentionPaper: https://arxiv.org/pdf/1706.03762v7
    .. |KerasTransformer| replace:: *Text classification with Transformer*
    .. _KerasTransformer: https://keras.io/examples/nlp/text_classification_with_transformer/
    """

    def __init__(
        self,
        embedding_dim: int,
        ffn_dim: int,
        num_heads: int,
        fully_mml: bool = True,
        rate: float = 0.1,
        **kwargs,
    ):
        """
        Initializes a new instance of the layer.

        Args:
            embedding_dim: Dimension of the embeddings.
            ffn_dim: Dimension of the intermediate (i.e., hidden) layer of the feed-forward network.
            num_heads: Number of heads to use for multi-headed attention.
            fully_mml: Whether to use full matmul-less layers in the attention mechanism.
            rate: Dropout rate to apply for the attention mechanism and the feed-forward network.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the embedding dimension is not a positive integer.
            ValueError: If the dimension of the intermediate layer of the feed-forward network is
                not a positive integer.
            ValueError: If the number of heads is not a positive integer.
            ValueError: If the embedding dimension is not divisible by the number of heads.
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
        self.fully_mml = fully_mml
        self.rate = rate

        self.attention = AttentionMML(num_heads, embedding_dim, fully_mml=fully_mml)
        self.attention_dropout = keras.layers.Dropout(rate)
        self.attention_norm = RMSNorm()

        self.ffn = SwiGLUMML(embedding_dim, intermediate_size=ffn_dim)
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
