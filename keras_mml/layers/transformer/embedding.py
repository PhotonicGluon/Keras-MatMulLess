"""
Implements embedding layers.
"""

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="keras_mml")
class TokenEmbedding(keras.layers.Layer):
    """
    TODO: ADD
    """

    def __init__(self, max_len: int, vocab_size: int, embed_dim: int, with_positions: bool = False, **kwargs):
        """
        TODO: ADD
        """

        super().__init__(**kwargs)

        self.input_spec = keras.layers.InputSpec(ndim=2)

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.with_positions = with_positions

        self.token_embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

        if with_positions:
            self.pos_embedding = keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        else:
            self.pos_embedding = None

    def call(self, inputs):
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer.

        Returns:
            Transformed inputs.
        """

        inputs = self.token_embedding(inputs)
        if self.pos_embedding is None:
            return inputs

        max_len = ops.shape(inputs)[-1]
        positions = ops.arange(start=0, stop=max_len, step=1)
        positions = self.pos_embedding(positions)

        return inputs + positions
