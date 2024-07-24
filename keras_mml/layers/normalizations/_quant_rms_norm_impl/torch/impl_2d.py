"""
Triton implementation of the quantized RMSNorm algorithm for 2D inputs.
"""

# ruff: noqa: N803, N806

import math
from typing import Optional

import torch
import triton
import triton.language as tl
from jaxtyping import Float

from keras_mml.layers.normalizations._quant_rms_norm_impl.torch.helpers import get_autotune_config


@triton.autotune(
    configs=get_autotune_config(),
    key=["N", "HAS_GAIN", "HAS_BIAS"],
)
@triton.jit
def quant_rms_norm_fwd_kernel(
    # fmt: off
    # Pointers to arrays
    x_ptr, y_ptr, gain_ptr, bias_ptr, rrms_ptr,
    # Strides
    stride_x_row, stride_y_row,
    # Some constants
    N, EPSILON,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr,
    HAS_GAIN: tl.constexpr,
    HAS_BIAS: tl.constexpr
    # fmt: on
):
    """
    Forward kernel.

    Performs RMSNorm on the input matrix, followed by 8-bit quantization.

    Args:
        x_ptr: Pointer to the input matrix.
        y_ptr: Pointer to the output matrix.
        gain_ptr: Pointer to the gain weight.
        bias_ptr: Pointer to the bias weight.
        rrms_ptr: Pointer to the reciprocal root mean square (RRMS) vector, which will be used for
            the backward pass.
        stride_x_row: How much to increase ``x_ptr`` when moving by 1 row.
        stride_y_row: How much to increase ``y_ptr`` when moving by 1 row.
        N: Number of columns in ``x``.
        EPSILON: Small value to avoid division by zero errors.
        BLOCK_SIZE_N: Block size for ``N``.
        HAS_GAIN: Whether a gain weight was provided.
        HAS_BIAS: Whether a bias weight was provided.
    """

    # Map the PID to the row of X that should be loaded
    pid = tl.program_id(0)
    x_ptr += pid * stride_x_row

    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)  # Load in higher precision

    # Compute reciprocal root mean square (rrms)
    mean_of_squares = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(mean_of_squares + EPSILON)
    tl.store(rrms_ptr + pid, rrms)  # We add PID since that is the row that the RRMS is corresponding to

    # Normalize
    x_hat = x * rrms

    # Apply gain and bias
    y = x_hat

    if HAS_GAIN:
        gain = tl.load(gain_ptr + offsets, mask=mask).to(tl.float32)
        y = y * gain
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offsets, mask=mask).to(tl.float32)
        y = y + bias

    # Apply 8-bit quantization
    scale = 127.0 / tl.maximum(tl.max(tl.abs(y), 0), EPSILON)
    y = tl.extra.cuda.libdevice.round(y * scale)  # TODO: This is CUDA only... can we generalize this?
    y = tl.maximum(tl.minimum(y, 127), -128) / scale  # The nested max and min creates the clamp/clip function

    # Write output
    y_ptr += pid * stride_y_row
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.autotune(
    configs=get_autotune_config(),
    key=["N", "HAS_GAIN", "HAS_BIAS"],
)
@triton.jit
def quant_rms_norm_bwd_kernel(
    # fmt: off
    # Gradient inputs
    grad_output_ptr, dx_ptr, dg_ptr, db_ptr,
    # Original inputs
    x_ptr, gain_ptr, rrms_ptr,
    # Strides
    stride_x_row, stride_grad_output_row, stride_dx_row,
    # Some constants
    M, N, ROWS_PER_PROGRAM,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr,
    HAS_GAIN: tl.constexpr,
    HAS_BIAS: tl.constexpr
    # fmt: on
):
    """
    Backward kernel.

    Performs the backward pass of RMSNorm on the input matrix. 8-bit quantization is ignored for the
    backward pass.

    Args:
        grad_output_ptr: Pointer to the output gradient matrix.
        dx_ptr: Pointer to the output matrix containing gradients for the original input matrix.
        dg_ptr: Pointer to the output matrix containing gradients for the gain weight.
        db_ptr: Pointer to the output matrix containing gradients for the bias weight.
        x_ptr: Pointer to the input matrix.
        gain_ptr: Pointer to the gain weight.
        rrms_ptr: Pointer to the reciprocal root mean square (RRMS) vector.
        stride_x_row: How much to increase ``x_ptr`` when moving by 1 row.
        stride_grad_output_row: How much to increase ``grad_output_ptr`` when moving by 1 row.
        stride_dx_row: How much to increase ``dx_ptr`` when moving by 1 row.
        M: Number of rows in ``x``.
        N: Number of columns in ``x``.
        ROWS_PER_PROGRAM: Number of rows of ``x`` to compute per program.
        BLOCK_SIZE_N: Block size for ``N``.
        HAS_GAIN: Whether a gain weight was provided.
        HAS_BIAS: Whether a bias weight was provided.
    """

    # Map the PID to the elements of `x`, `dx`, `dg`, and `db` that should be computed
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROGRAM

    x_ptr += row_start * stride_x_row
    grad_output_ptr += row_start * stride_grad_output_row
    dx_ptr += row_start * stride_dx_row

    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < N

    # Load gradient array, and prepare gradient and bias output gradient arrays
    if HAS_GAIN:
        gain = tl.load(gain_ptr + offsets, mask=mask).to(tl.float32)
        dg = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # Iterate through the rows
    row_end = min(row_start + ROWS_PER_PROGRAM, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)  # Load in higher precision
        grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rrms = tl.load(rrms_ptr + row)  # Load the reciprocal root mean square (rrms)

        # Compute `x_hat` and the product of the gradient output with the gain
        x_hat = x * rrms

        # Apply contributions to the gain and bias gradients
        gradient_gain_product = grad_output
        if HAS_GAIN:
            gradient_gain_product = grad_output * gain
            dg += grad_output * x_hat
        if HAS_BIAS:
            db += grad_output

        # Compute `dx`
        intermediate_const = tl.sum(x_hat * gradient_gain_product, axis=0) / N
        dx = (gradient_gain_product - x_hat * intermediate_const) * rrms

        # Write `dx`
        tl.store(dx_ptr + offsets, dx, mask=mask)

        # Update pointers to move to next row
        x_ptr += stride_x_row
        grad_output_ptr += stride_grad_output_row
        dx_ptr += stride_dx_row

    # Once we finished computing all the rows for this program, we can write the final `dg` and `db`
    if HAS_GAIN:
        tl.store(dg_ptr + pid * N + offsets, dg, mask=mask)
    if HAS_BIAS:
        tl.store(db_ptr + pid * N + offsets, db, mask=mask)


class QuantRMSNorm2DFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Float[torch.Tensor, "M N"],
        gain: Optional[Float[torch.Tensor, "N"]],
        bias: Optional[Float[torch.Tensor, "N"]],
        epsilon: float = 1e-5,
    ):
        # Get dimensions
        M, N = x.shape

        # Enqueue fused kernel if less than 64KiB per feature
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KiB.")

        # Allocate output
        y = torch.empty_like(x, dtype=x.dtype)
        rrms = torch.empty((M,), dtype=torch.float32, device="cuda")

        # Run the kernel
        with torch.cuda.device(x.device.index):
            quant_rms_norm_fwd_kernel[(M,)](
                # fmt: off
                # Pointers to arrays
                x, y, gain, bias, rrms,
                # Strides
                x.stride(0), y.stride(0),
                # Some constants
                N, epsilon,
                # Meta-parameters
                BLOCK_SIZE_N, gain is not None, bias is not None
                # fmt: on
            )

        # Save tensors for backward pass later
        ctx.save_for_backward(x, gain, bias, rrms)

        # Return the result of the forward pass
        return y

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Float[torch.Tensor, "M N"]):
        # Retrieve stored tensors
        x, gain, bias, rrms = ctx.saved_tensors

        # Get dimensions
        M, N = x.shape

        # Enqueue fused kernel if less than 64KiB per feature
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KiB.")

        # Allocate output
        multi_processor_count = torch.cuda.get_device_properties(x.device).multi_processor_count

        dx = torch.empty_like(x, dtype=x.dtype)
        if gain is not None:
            # This is temporary as we still need to sum across the rows later
            dg_temp = torch.empty((multi_processor_count, N), dtype=torch.float32, device=gain.device)
        else:
            dg_temp = None
        if bias is not None:
            db_temp = torch.empty((multi_processor_count, N), dtype=torch.float32, device=bias.device)
        else:
            db_temp = None

        # Run the kernel
        # TODO: We could make this faster by using a technique like shown in
        #   https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#backward-pass
        rows_per_program = math.ceil(M / multi_processor_count)
        with torch.cuda.device(x.device.index):
            quant_rms_norm_bwd_kernel[(multi_processor_count,)](
                # fmt: off
                # Gradient inputs
                grad_output, dx, dg_temp, db_temp,
                # Original inputs
                x, gain, rrms,
                # Strides
                x.stride(0), grad_output.stride(0), dx.stride(0),
                # Some constants
                M, N, rows_per_program,
                # Meta-parameters
                BLOCK_SIZE_N, gain is not None, bias is not None
                # fmt: on
            )

        # Fix the summing of `dg` and `db`
        if gain is not None:
            dg = dg_temp.sum(0).to(gain.dtype)
        else:
            dg = None

        if bias is not None:
            db = db_temp.sum(0).to(bias.dtype)
        else:
            db = None

        # Return the gradients
        return dx, dg, db, None  # No gradient for `epsilon`


def quant_rms_norm_2d(
    x: Float[torch.Tensor, "M N"],
    gain: Optional[Float[torch.Tensor, "N"]],
    bias: Optional[Float[torch.Tensor, "N"]],
    epsilon: float = 1e-5,
):
    """
    Wrapper function for the quantized RMSNorm.

    Works only for the 2D case.

    Args:
        x: Input matrix.
        gain: Gain weight.
        bias: Bias weight.
        epsilon: Small value to avoid division by zero.

    Returns:
        Normalized and quantized inputs.
    """

    return QuantRMSNorm2DFn.apply(x, gain, bias, epsilon)
