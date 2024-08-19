"""
Checks for Triton.
"""

import warnings


def has_triton_package() -> bool:
    """
    Checks if the Triton package is available.

    Returns:
        Whether the Triton package is found on the system.
    """

    try:
        import triton

        return triton is not None
    except ImportError:
        return False


def can_use_triton() -> bool:
    """
    Checks if the system can use the Triton package for PyTorch speedup.

    Returns:
        Whether Triton can be used.
    """

    from torch._dynamo.device_interface import get_interface_for_device

    device_interface = get_interface_for_device("cuda")

    if device_interface.is_available():
        if device_interface.Worker.get_device_properties().major < 7:  # Compute capacity below 7.0
            warnings.warn(
                "Compute capacity of CUDA device is below 7.0. The Triton compilation may fail terribly!", stacklevel=1
            )
        return has_triton_package()
    return False
