"""
Custom decorators for test coverage management.
"""

import os
from typing import Callable, Optional


def torch_compile(model: Optional[Callable] = None, **kwargs) -> Callable:  # pragma: no cover
    """
    Custom decorator similar to :py:func:`torch.compile`. However the compilation of the function
    will not be performed if the ``DISABLE_TORCH_COMPILE`` environment variable is set.

    Arguments:
        model: Module/function to optimize.
        *args: Arguments for :py:func:`torch.compile`.
        **kwargs: Keyword arguments for :py:func:`torch.compile`.

    Returns:
        Decorated model.
    """

    if os.environ.get("DISABLE_TORCH_COMPILE"):
        return lambda x: x  # Identity function

    import torch  # We want to import only if we are not using eager

    return torch.compile(model, **kwargs)
