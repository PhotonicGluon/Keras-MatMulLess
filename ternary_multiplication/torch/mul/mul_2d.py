"""
Triton-optimized version of matrix-vector ternary multiplication.
"""

# ruff: noqa: E731, S101, N806, N803, PLR2044

from typing import List

import torch
import triton
import triton.language as tl
from jaxtyping import Float

from ternary_multiplication.torch.helpers import is_cuda


def get_autotune_config_2d() -> List[triton.Config]:
    """
    Gets the autotune configuration for the Triton kernel.

    Raises:
        ValueError: If not on a CUDA-enabled GPU.

    Returns:
        A list of configurations for the autotuner to select.
    """

    if not is_cuda():
        raise ValueError("Can't use Triton if not on CUDA!")

    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 256,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 128,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
            },
            num_stages=5,
            num_warps=2,
        ),
    ]


@triton.autotune(
    configs=get_autotune_config_2d(),
    key=["M", "N"],
)
@triton.jit
def ternary_mul_2d_kernel(
    # Pointers to arrays
    x_ptr,
    w_ptr,
    z_ptr,
    # Scaling factor
    scale,
    # `W` matrix dimensions
    M,
    N,
    # The stride variables represent how much to increase the pointer by when moving by 1 element in a particular
    # dimension. E.g. `stride_wm` is how much to increase `w_ptr` by to get the element one row down (the `W` matrix
    # has `M` rows).
    stride_xm,
    stride_wm,
    stride_wn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Kernel for computing the ternary multiplication ``z = xW``.

    Here ``x`` has shape ``(M,)``, ``W`` has shape ``(M, N)``, and ``z`` has shape ``(N,)``.

    Args:
        x_ptr: Pointer to ``x``.
        w_ptr: Pointer to ``W``.
        z_ptr: Pointer to ``z``.
        scale: Scaling factor.
        M: Value for ``M``.
        N: Value for ``N``.
        stride_xm: How much to increase ``x_ptr`` when moving 1 element along the ``M`` direction.
        stride_wm: How much to increase ``w_ptr`` when moving 1 element along the ``M`` direction.
        stride_wn: How much to increase ``w_ptr`` when moving 1 element along the ``N`` direction.
        BLOCK_SIZE_M: Block size for ``M``. Will be a power of 2.
        BLOCK_SIZE_N: Block size for ``N``. Will be a power of 2.
    """

    # ----------------------------------------------------------
    # Create pointers for the first blocks of `x` and `W`.
    # We will advance this pointer as we move in the `M` direction and accumulate.
    # - `x_ptrs` is a block of `BLOCK_SIZE_M` pointers
    # - `w_ptrs` is a block of pointers with shape `(BLOCK_SIZE_M, BLOCK_SIZE_N)`
    pid_0 = tl.program_id(axis=0)

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = (pid_0 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N  # Guard against wrong offsets

    x_ptrs = x_ptr + offs_m
    w_ptrs = w_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the `z` vector.
    # We accumulate into a block of `BLOCK_SIZE_N` elements of FP32 values for higher accuracy.
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        # Load the next block of `x` and `W`, generate a mask by checking along `M`.
        # If it is out of bounds, set it to 0.
        x = tl.load(x_ptrs, mask=offs_m < M - m * BLOCK_SIZE_M, other=0.0)[:, None]  # Force broadcast to correct shape
        w = tl.load(w_ptrs, mask=offs_m[:, None] < M - m * BLOCK_SIZE_M, other=0.0)

        # Since `w` is ternary, we only really care about the sign of the element in the array, and so
        # we just need to perform two conditional checks
        elements_to_sum = tl.where(w > 0, x, tl.where(w < 0, -x, tl.zeros_like(x)))
        accumulator = accumulator + tl.sum(elements_to_sum, axis=0)  # Sum along the `M` direction

        # Advance the ptrs to the next `M` block.
        x_ptrs += BLOCK_SIZE_M * stride_xm
        w_ptrs += BLOCK_SIZE_M * stride_wm

    z = accumulator / scale  # TODO: Do we want to reduce precision back to FP16?

    # -----------------------------------------------------------
    # Write back the block of the output vector `z` with masks.
    offs_z = pid_0 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    z_ptrs = z_ptr + offs_z
    z_mask = offs_z < N
    tl.store(z_ptrs, z, mask=z_mask)


def ternary_mul_2d(
    x: Float[torch.Tensor, "M"], w: Float[torch.Tensor, "M N"], scale: float
) -> Float[torch.Tensor, "N"]:
    """
    Applies the ternary multiplication algorithm.

    Args:
        x: 1D vector.
        w: 2D ternary matrix.
        scale: Scaling factor.

    Returns:
        Result of the ternary multiplication.
    """

    # Check constraints
    assert len(x) == w.shape[0], "Incompatible dimensions"
    assert x.is_contiguous(), "x must be contiguous"
    assert x.is_cuda and w.is_cuda

    # Get dimensions
    M, N = w.shape

    # Allocate output
    z = torch.empty((N,), device=x.device, dtype=torch.float32)  # TODO: Change precision?

    # 1D launch kernel where each block gets its own program
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    # fmt: off
    ternary_mul_2d_kernel[grid](
        x, w, z,
        scale,
        M, N,
        x.stride(0),
        w.stride(0), w.stride(1)
    )
    # fmt: on

    return z
