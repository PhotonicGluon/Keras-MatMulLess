"""
Implements a matmul-less Gated Recurrent Unit (GRU) layer.
"""

import keras


@keras.saving.register_keras_serializable(package="keras_mml")
class GRUMML(keras.Layer):
    """
    TODO: ADD DOCS

    Look at https://github.com/ridgerchu/matmulfreellm/blob/4a27497acfaa95fd7e88dad720ef8782f695b0f0/mmfreelm/layers/hgrn_bit.py#L22, https://arxiv.org/pdf/2406.02528 section 3.3.1, https://arxiv.org/pdf/2311.04823 section 3.2.
    """
