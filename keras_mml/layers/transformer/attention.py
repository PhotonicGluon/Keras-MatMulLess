"""
Implements a matmul-less attention layer.
"""

import keras

from keras_mml.layers.recurrent.gru import GRUMML


@keras.saving.register_keras_serializable(package="keras_mml")
class AttentionMML(keras.Layer):
    """
    TODO: Add
    """

    def __init__(self, num_heads: int, out_dim: int, fully_mml: bool = True, **kwargs):
        """
        TODO: Add
        """

        super().__init__(**kwargs)

        self.internal_layer = GRUMML(out_dim, fully_mml=fully_mml, num_heads=num_heads, return_sequences=True)

    def call(self, inputs):
        """
        Calling method of the layer.

        Args:
            inputs: Inputs into the layer.

        Returns:
            Transformed inputs.
        """

        return self.internal_layer(inputs)
