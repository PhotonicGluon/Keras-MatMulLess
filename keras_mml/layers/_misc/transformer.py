"""
Implements a matmul-less transformer block.
"""

import keras


@keras.saving.register_keras_serializable(package="keras_mml")
class TransformerBlockMML(keras.Layer):
    """
    TODO: ADD
    """
