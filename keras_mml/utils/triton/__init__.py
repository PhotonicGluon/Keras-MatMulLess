"""
Triton utility functions.

Used with the PyTorch backend.
"""

from .checks import can_use_triton

# try:
#     if not can_use_triton():
#         raise ModuleNotFoundError
# except ModuleNotFoundError:
#     pass
# else:
#     pass  # Add other imports here
