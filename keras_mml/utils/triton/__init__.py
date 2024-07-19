"""
Triton utility functions.

Used with the PyTorch backend.
"""

try:
    import triton
except ModuleNotFoundError:
    pass
else:
    from .helpers import get_current_target, is_cuda
