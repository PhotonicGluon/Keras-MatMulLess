"""
NumPy functions implemented in Numba.
"""

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def max_in_interval(array: np.ndarray, start: int, end: int) -> int:
    """
    Gets the maximum of the array in the interval :math:`[\\mathrm{start}, \\mathrm{end})`.

    Args:
        array: Array. Assumed to be 1D.
        start: Start of the interval. Included.
        end: End of the interval. Excluded.

    Returns:
        Maximum in the interval.

    Raises:
        ValueError: If the array has no elements.
        ValueError: If the provided start index is bigger than the end.

    Examples:
        >>> max_in_interval(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 2, 6)
        6
    """

    if len(array) == 0:
        raise ValueError("Can't find maximum of zero-size array")

    if start >= end:
        raise ValueError("End must be after start")

    maximum = array[start]
    for i in range(start + 1, end):
        if array[i] > maximum:
            maximum = array[i]
    return maximum


@nb.jit(nopython=True, parallel=True)
def max_for_last_axis(array: np.ndarray) -> np.ndarray:
    """
    Computes :py:func:`np.max(array, axis=-1)`.

    Args:
        array: Array.

    Returns:
        Maxima for the array.
    """

    if array.ndim == 1:
        return np.max(array)

    shape = array.shape
    output = np.zeros(shape[:-1], dtype=array.dtype)

    num_indices = np.prod(np.array(output.shape))

    for i in nb.prange(num_indices):
        maximum = max_in_interval(array.flat, i * shape[-1], (i + 1) * shape[-1])
        output.flat[i] = maximum

    return output
