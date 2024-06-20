"""
Root Mean Square Normalization (RMSNorm) implementation.
"""

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="keras_mml")
class RMSNorm(keras.Layer):
    """
    Implements Root Mean Square Normalization introduced in https://arxiv.org/pdf/1910.07467.pdf.

    Attributes:
        scale: Scaling factor.
    """

    def __init__(self, dim: int):
        """
        Initializes a new RMSNorm instance.

        Args:
            dim: Embedding size. Will be the square of the scaling factor (i.e., :py:attr:`~scale`).

        Raises:
            ValueError: If the given embedding size is not a positive integer.
        """

        super().__init__()

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
