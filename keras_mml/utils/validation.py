"""
Utilities for validating values.
"""

from typing import Tuple


def ensure_is_rank_2(input_shape: Tuple[int, ...]):
    """
    Ensures that the input shape is of rank 2.

    That is, this ensures that the input shape is of the shape :math:`(x, y)` where :math:`x` and
    :math:`y` are integers.

    Args:
        input_shape: Input shape to check.

    Raises:
        ValueError: If the input shape is not of rank 2.
    """

    if len(input_shape) != 2:
        raise ValueError(f"Input shape must have rank 2 (received: {input_shape})")
