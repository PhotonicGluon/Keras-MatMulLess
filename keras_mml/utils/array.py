"""
Utilities for handing numpy arrays.
"""

import os
from typing import Any

import numpy as np


def as_numpy(x: Any) -> np.ndarray:
    """
    Converts the given object into a numpy array.

    This function will first attempt to call the ``.numpy()`` method of the object and return it.
    Failing to do so, it will revert to the :external:py:func:`numpy.array()` function.

    Args:
        x: Object to convert into a numpy array.

    Returns:
        Numpy array of the given object.
    """

    try:
        return x.numpy()
    except RuntimeError:
        if os.environ["KERAS_BACKEND"] == "torch":
            return as_numpy(x.detach())
    except AttributeError:
        pass

    return np.array(x)
