"""
Implements an image patch extraction layer.
"""

import keras


@keras.saving.register_keras_serializable(package="keras_mml")
class Patches(keras.Layer):
    """
    TODO: ADD DOCS
    """
