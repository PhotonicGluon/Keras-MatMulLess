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
        """

        super().__init__()
        self.scale = dim**-0.5

    def call(self, x):
        """
        Args:
            x: Input tensor to normalize.

        Returns:
            The output tensor after applying RMSNorm.
        """

        return ops.normalize(x, order=2, axis=-1) * self.scale
