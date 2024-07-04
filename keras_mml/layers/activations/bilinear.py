"""
Implements a matmul-less Bilinear layer.
"""

import keras
from keras import activations

from keras_mml.layers.activations.glu import GLUMML


@keras.saving.register_keras_serializable(package="keras_mml")
class BilinearMML(GLUMML):
    """
    Gated Linear Unit (GLU) without matrix multiplications and any activation functions. Also called
    "Bilinear" (see |GLUVariants|_, section 2).

    See :py:class:`~.GLUMML` for the full documentation.

    .. |GLUVariants| replace:: *GLU Variants Improve Transformer*
    .. _GLUVariants: https://arxiv.org/pdf/2002.05202v1
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes a new instance of the layer.

        Args:
            *args: Arguments to be passed into :py:class:`~.GLUMML`.
            **kwargs: Keyword arguments to be passed into :py:class:`~.GLUMML`.
        """

        super().__init__(*args, **kwargs)
        self.activation = activations.get("linear")
