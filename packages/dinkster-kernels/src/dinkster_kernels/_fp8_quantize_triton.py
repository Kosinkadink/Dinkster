"""Triton kernel for per-tensor FP8 input quantization."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _round_to_fp8(values, MANTISSA_BITS: tl.constexpr, MIN_EXPONENT: tl.constexpr):
    bits = values.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF).to(tl.int32) - 127
    quantum_exponent = tl.maximum(MIN_EXPONENT - MANTISSA_BITS, exponent - MANTISSA_BITS)
    quantum_bits = (quantum_exponent + 127).to(tl.uint32) << 23
    quantum = quantum_bits.to(tl.float32, bitcast=True)
    inverse_quantum_bits = (-quantum_exponent + 127).to(tl.uint32) << 23
    inverse_quantum = inverse_quantum_bits.to(tl.float32, bitcast=True)
    rounded = libdevice.rint(values * inverse_quantum) * quantum
    return tl.where(values == values, rounded, values)


@triton.jit
def _quantize_per_tensor_fp8_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    n_elements,
    FP8_MAX: tl.constexpr,
    MANTISSA_BITS: tl.constexpr,
    MIN_EXPONENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    scale = tl.load(scale_ptr)
    values = tl.div_rn(tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32), scale)
    values = tl.minimum(FP8_MAX, tl.maximum(-FP8_MAX, values))
    values = _round_to_fp8(values, MANTISSA_BITS, MIN_EXPONENT)
    tl.store(output_ptr + offsets, values, mask=mask)


def _pick_block(n_elements: int) -> int:
    if n_elements < 1 << 15:
        return 128
    if n_elements < 1 << 17:
        return 256
    if n_elements < 1 << 19:
        return 512
    return 1024


def launch_quantize_per_tensor_fp8(
    input: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Quantize contiguous input through one float32 scalar scale."""
    n_elements = input.numel()
    block = _pick_block(n_elements)
    grid = (triton.cdiv(n_elements, block),)
    _quantize_per_tensor_fp8_kernel[grid](
        input,
        scale,
        output,
        n_elements,
        FP8_MAX=torch.finfo(output.dtype).max,  # pyright: ignore[reportArgumentType]
        MANTISSA_BITS=3 if output.dtype == torch.float8_e4m3fn else 2,  # pyright: ignore[reportArgumentType]
        MIN_EXPONENT=-6 if output.dtype == torch.float8_e4m3fn else -14,  # pyright: ignore[reportArgumentType]
        BLOCK=block,  # pyright: ignore[reportArgumentType]
    )
