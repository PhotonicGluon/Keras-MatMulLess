"""
Helper Triton functions.
"""

import warnings

import triton
import triton.backends


def get_current_target() -> triton.backends.compiler.GPUTarget:
    """
    Gets the current target for the Triton runtime.

    Returns:
        GPU target instance.
    """

    return triton.runtime.driver.active.get_current_target()


def is_cuda() -> bool:
    """
    Checks if the target for the Triton runtime is CUDA.

    Will also emit a warning if the compute capacity of the CUDA-supported GPU is less than 7.0, which is the lowest
    stable supported by Triton.

    Returns:
        Whether the current backend is CUDA or not.
    """

    current_target = get_current_target()
    if current_target.backend != "cuda":
        return False

    if current_target.arch < 70:  # Compute capacity is below 7.0
        warnings.warn(
            "Compute capacity of CUDA device is below 7.0. The Triton compilation may fail terribly!", stacklevel=1
        )

    return True
