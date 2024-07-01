"""
Custom decorators for test coverage management.
"""

import sys
from typing import Callable, Optional


def torch_compile(model: Optional[Callable] = None, *args, **kwargs) -> Callable:  # pragma: no cover
    """
    Custom decorator similar to :py:func:`torch.compile`. However the compilation of the function
    will not be performed if running in a PyTest suite.

    Arguments:
        model: Module/function to optimize.
        *args: Arguments for :py:func:`torch.compile`.
        **kwargs: Keyword arguments for :py:func:`torch.compile`.

    Returns:
        Decorated model.
    """

    if "pytest" in sys.modules:
        return lambda x: x  # Identity function

    import torch

    return torch.compile(model, *args, **kwargs)
