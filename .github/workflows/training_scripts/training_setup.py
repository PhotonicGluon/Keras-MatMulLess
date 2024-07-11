"""
Setup for training scripts.
"""

import keras

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
keras.utils.set_random_seed(812)

if keras.config.backend() == "tensorflow":
    import tensorflow as tf

    # If using TensorFlow, this will make GPU ops as deterministic as possible,
    # but it will affect the overall performance, so be mindful of that.
    tf.config.experimental.enable_op_determinism()
if keras.config.backend() == "torch":
    # Enable eager mode for PyTorch
    import os

    os.environ["PYTEST_USE_EAGER"] = "true"
