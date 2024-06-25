"""
Root Mean Square Normalization (RMSNorm) implementation.
"""

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="keras_mml")
class RMSNorm(keras.Layer):
    """
    Implements Root Mean Square Normalization in |RMSNorm Paper|_.

    Attributes:
        scale: Scaling factor.

    .. |RMSNorm Paper| replace:: *Root Mean Square Layer Normalization*
    .. _RMSNorm Paper: https://arxiv.org/pdf/1910.07467v1
    """

    def __init__(self, dim: int, **kwargs):
        """
        Initializes a new RMSNorm instance.

        Args:
            dim: Embedding size. Will be the square of the scaling factor (i.e., :py:attr:`~scale`).
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the given embedding size is not a positive integer.
        """

        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"Received an invalid value for argument `dim`, expected a positive integer, got {dim}")
        self.scale = dim**-0.5

    def call(self, x):
        """
        Args:
            x: Input tensor to normalize.

        Returns:
            The output tensor after applying RMSNorm.
        """

        return ops.normalize(x, order=2, axis=-1) * self.scale
