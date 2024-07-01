"""
Backend-*dependent* selection of the implementation of ternary multiplication.
"""

import keras

BACKEND = keras.config.backend()

if BACKEND == "tensorflow":
    from keras_mml.utils.array._ternary_multiplication_impl.tensorflow_multiplication import tf_ternary_multiplication

    backend_function = tf_ternary_multiplication
elif BACKEND == "torch":
    from keras_mml.utils.array._ternary_multiplication_impl.torch_multiplication import torch_ternary_multiplication

    backend_function = torch_ternary_multiplication
elif BACKEND == "jax":
    from keras_mml.utils.array._ternary_multiplication_impl.jax_multiplication import jax_ternary_multiplication

    backend_function = jax_ternary_multiplication
else:  # pragma: no cover
    raise ValueError(f"Invalid backend: {BACKEND}")
