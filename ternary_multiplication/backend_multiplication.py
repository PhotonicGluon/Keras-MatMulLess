"""
Backend-*dependent* selection of the implementation of ternary multiplication.
"""

import keras

BACKEND = keras.config.backend()

if BACKEND == "tensorflow":
    from ternary_multiplication.tensorflow_multiplication import tf_ternary_multiplication

    backend_function = tf_ternary_multiplication
elif BACKEND == "torch":
    from ternary_multiplication.torch.mul import ternary_multiplication

    backend_function = ternary_multiplication
elif BACKEND == "jax":
    from ternary_multiplication.jax_multiplication import jax_ternary_multiplication

    backend_function = jax_ternary_multiplication
else:  # pragma: no cover
    raise ValueError(f"Invalid backend: {BACKEND}")
