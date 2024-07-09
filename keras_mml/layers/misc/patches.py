"""
Implements an image patch extraction layer.
"""

import keras
import numpy as np
from jaxtyping import Float
from keras import ops


@keras.saving.register_keras_serializable(package="keras_mml")
class Patches(keras.Layer):
    """
    Layer that creates patches for an image.

    This layer implements the patch extraction algorithm described in |ViT|_. Useful for use in
    Vision Transformers (ViT).

    Adapted from the ``Patches`` class in the Keras code example |ViT-CodeEx|_.

    Attributes:
        patch_size: Size of the patches.

    .. |ViT| replace:: *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
    .. _ViT: https://arxiv.org/pdf/2010.11929v2
    .. |ViT-CodeEx| replace:: *Image classification with Vision Transformer*
    .. _ViT-CodeEx: https://keras.io/examples/vision/image_classification_with_vision_transformer/
    """

    def __init__(self, patch_size: int, **kwargs):
        """
        Initializes a new instance of the layer.

        Args:
            patch_size: Size of the patches.
            **kwargs: Keyword arguments for :py:class:`keras.Layer`.

        Raises:
            ValueError: If the patch size provided is not a positive integer.
        """

        if patch_size <= 0:
            raise ValueError(f"Invalid patch size, expected a positive integer, got {patch_size}")

        super().__init__(**kwargs)
        self.input_spec = keras.layers.InputSpec(ndim=4)

        self.patch_size = patch_size

    def call(
        self, inputs: Float[np.ndarray, "batch_size height width channels"]
    ) -> Float[np.ndarray, "batch_size patch_count patch_dim"]:
        """
        Calling method of the layer.

        .. NOTE::
            ``patch_dim`` is equal to ``channels * (patch_size)**2``.

        Args:
            inputs: Inputs into the layer.

        Returns:
            Transformed inputs.
        """

        # Extract the needed information from the input shape
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        # Get the number of patches width-wise and height-wise
        num_patches_w = width // self.patch_size
        num_patches_h = height // self.patch_size

        # Form the patches
        patches = ops.image.extract_patches(inputs, size=self.patch_size, data_format="channels_last")
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )

        return patches
