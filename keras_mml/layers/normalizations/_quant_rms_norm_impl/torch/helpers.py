"""
Helper Triton code.
"""

from typing import List

import triton


def get_autotune_config() -> List[triton.Config]:
    """
    Gets the list of configurations to use when autotuning the compilation.

    Returns:
        List of Triton configurations.
    """

    return [
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ]
