"""
Implements a matmul-less attention layer.
"""

import keras


@keras.saving.register_keras_serializable(package="keras_mml")
class AttentionMML(keras.Layer):
    """
    TODO: Add

    References:
    - https://arxiv.org/pdf/2404.07904v1
    - https://github.com/Doraemonzzz/hgru2-pytorch/blob/main/hgru2_pytorch/models/modeling_hgrn2.py
    - https://github.com/ridgerchu/matmulfreellm/blob/master/mmfreelm/layers/hgrn_bit.py#L22
    """
