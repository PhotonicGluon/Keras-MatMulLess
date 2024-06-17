"""
Implements a matmul-less Gated Linear Unit (GLU) layer.
"""

import keras


@keras.saving.register_keras_serializable(package="keras_mml")
class GLUMML(keras.Layer):
    """
    TODO: ADD DOCS
    """
