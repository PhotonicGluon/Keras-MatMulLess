"""
Implements a matmul-less Gated Recurrent Unit (GRU) layer.
"""

import keras


@keras.saving.register_keras_serializable(package="keras_mml")
class GRUMML(keras.Layer):
    """
    TODO: ADD DOCS
    """
