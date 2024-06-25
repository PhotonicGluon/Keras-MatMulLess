"""
Implements embedding layers.
"""

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="keras_mml")
class TokenAndPositionEmbedding(keras.layers.Layer):
    """
    TODO: ADD
    """

    def __init__(self, max_len: int, vocab_size: int, embed_dim: int, **kwargs):
        """
        TODO: ADD
        """

        super().__init__(**kwargs)

        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, inputs):
        """
        TODO: ADD
        """

        max_len = ops.shape(inputs)[-1]
        positions = ops.arange(start=0, stop=max_len, step=1)
        positions = self.pos_emb(positions)
        inputs = self.token_emb(inputs)

        return inputs + positions
