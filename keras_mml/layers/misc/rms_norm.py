import keras
from keras import ops


@keras.saving.register_keras_serializable(package="keras_mml")
class RMSNorm(keras.Layer):
    """
    Implements Root Mean Square Normalization introduced in https://arxiv.org/pdf/1910.07467.pdf.

    Args:
        dim: embedding size

    Attributes:
        scale: scaling factor. Is the inverse square root of the embedding size (i.e., `dim`)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**-0.5

    def call(self, x):
        """
        Args:
            x: input tensor to normalize

        Returns:
            The output tensor after applying RMSNorm.
        """

        return ops.normalize(x, order=2, axis=-1) * self.scale
