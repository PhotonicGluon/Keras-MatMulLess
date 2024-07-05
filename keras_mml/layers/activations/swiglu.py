"""
Implements a matmul-less SwiGLU layer.
"""

import keras
from keras import activations

from keras_mml.layers.activations.glu import GLUMML


@keras.saving.register_keras_serializable(package="keras_mml")
class SwiGLUMML(GLUMML):
    """
    Swish with :math:`\\beta = 1` (i.e., Sigmoid Linear Unit (SiLU)) activated Gated Linear Unit
    (GLU) without matrix multiplications.

    See :py:class:`~.GLUMML` for the full documentation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes a new instance of the layer.

        Args:
            *args: Arguments to be passed into :py:class:`~.GLUMML`.
            **kwargs: Keyword arguments to be passed into :py:class:`~.GLUMML`.
        """

        super().__init__(*args, **kwargs)
        self.activation = activations.get("silu")
