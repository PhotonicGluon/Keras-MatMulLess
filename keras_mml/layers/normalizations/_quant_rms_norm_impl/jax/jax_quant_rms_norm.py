"""
Jax implementation for the quantized RMSNorm layer.
"""

import jax
import jax.numpy as jnp
import keras
from jaxtyping import Float

from keras_mml.layers.normalizations._quant_rms_norm_impl.base_quant_rms_norm import EPSILON, HUGE, BaseQuantRMSNorm


@jax.jit
def _activations_quantization(x: jax.Array) -> jax.Array:
    """
    Quantizes the activations to 8-bit precision using absmax quantization.

    Args:
        x: Array of quantization values.

    Returns:
        The quantized activation values.
    """

    scale = 127.0 / jnp.expand_dims(jnp.clip(jnp.max(jnp.abs(x), axis=-1), EPSILON, HUGE), -1)
    y = jnp.clip(jnp.round(x * scale), -128, 127) / scale
    return y


@jax.jit
def _get_x_quantized(x_norm: jax.Array) -> jax.Array:
    """
    Gets the quantized activations, with support for the backward direction by using STE gradient
    bypass.

    We use a Straight-Through Estimator (STE) trick by stopping gradient propagation.

    Args:
        x_norm: Normalized activation values.

    Returns:
        Quantized activation values.
    """

    return x_norm + jax.lax.stop_gradient(_activations_quantization(x_norm) - x_norm)


@keras.saving.register_keras_serializable(package="keras_mml")
class JaxQuantRMSNorm(BaseQuantRMSNorm):
    """
    Jax implementation of Root Mean Square Normalization (RMSNorm) with 8-bit quantization.
    """

    def call(self, inputs: Float[jax.Array, "batch_size *dims"]) -> Float[jax.Array, "batch_size *dims"]:
        x_norm = super().call(inputs)
        return _get_x_quantized(x_norm)
