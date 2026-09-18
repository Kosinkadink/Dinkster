"""Triton device kernel for the combined RoPE rotation.

Imported lazily from :mod:`rope` inside the CUDA implementation so the
op schema registers without triton installed.

Each lane owns one channel pair and rotates q and k in the same
program, so the four rotation coefficients are loaded once per pair
and every input element is loaded exactly once.

The rounding sequence is the bit-identity contract with the reference
rotation (comfy/ldm/flux/math.py _apply_rope1 @ b78cec87): cast the
pair to float32, one rounded multiply ``f0 * x0``, then a fused
multiply-add ``fma(f1, x1, .)`` - exactly what ``torch.addcmul``
executes on both CPU and CUDA - and one rounding back to the input
dtype. The explicit ``tl.math.fma`` call keeps the compiler from
re-contracting the separate multiply, which is what makes the kitchen
Triton kernel's ``f0*x0 + f1*x1`` differ from the reference by one ulp.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Pairs per program. Small blocks tile better through L2 on mid-size
# tensors; large blocks amortize launch overhead once the tensor is
# firmly DRAM-bound (measured on an RTX 5090, threshold at 16M pairs).
_BLOCK_SMALL = 256
_BLOCK_LARGE = 1024
_LARGE_PAIRS = 1 << 24


def _pick_block(n_pairs_total: int) -> int:
    return _BLOCK_SMALL if n_pairs_total <= _LARGE_PAIRS else _BLOCK_LARGE


@triton.jit
def _rotate_pair(
    x_ptr,
    out_ptr,
    flat,
    flat_mask,
    f00,
    f01,
    f10,
    f11,
    out_dtype: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One contiguous vectorized load covers the block's pairs; split
    # deinterleaves the even/odd channels into registers.
    x = tl.load(x_ptr + flat, mask=flat_mask)
    x0, x1 = tl.split(tl.reshape(x, (BLOCK, 2)))
    x0 = x0.to(tl.float32)
    x1 = x1.to(tl.float32)
    out0 = tl.math.fma(f01, x1, f00 * x0)
    out1 = tl.math.fma(f11, x1, f10 * x0)
    out = tl.interleave(out0.to(out_dtype), out1.to(out_dtype))  # pyright: ignore[reportArgumentType]
    tl.store(out_ptr + flat, out, mask=flat_mask)


@triton.jit
def _rope_kernel(
    q_ptr,
    k_ptr,
    f_ptr,
    out_q_ptr,
    out_k_ptr,
    n_pairs_total,
    seq_len,
    f_batch_stride,
    HEAD_PAIRS: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs_total

    # offs walks (batch, heads, seq, pair); the pair's two channels sit
    # at the adjacent even-aligned input slots.
    pair = offs % HEAD_PAIRS
    position = (offs // HEAD_PAIRS) % seq_len
    batch = offs // (HEAD_PAIRS * seq_len * HEADS)

    f_base = batch * f_batch_stride + (position * HEAD_PAIRS + pair) * 4
    f00 = tl.load(f_ptr + f_base, mask=mask)
    f01 = tl.load(f_ptr + f_base + 1, mask=mask)
    f10 = tl.load(f_ptr + f_base + 2, mask=mask)
    f11 = tl.load(f_ptr + f_base + 3, mask=mask)

    flat = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    flat_mask = flat < 2 * n_pairs_total
    out_dtype: tl.constexpr = out_q_ptr.dtype.element_ty
    _rotate_pair(q_ptr, out_q_ptr, flat, flat_mask, f00, f01, f10, f11, out_dtype, BLOCK)
    _rotate_pair(k_ptr, out_k_ptr, flat, flat_mask, f00, f01, f10, f11, out_dtype, BLOCK)


def launch_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    out_q: torch.Tensor,
    out_k: torch.Tensor,
) -> None:
    """Rotate contiguous ``(B, H, S, D)`` q/k through contiguous
    ``(fb, 1, S, D/2, 2, 2)`` float32 frequencies into preallocated
    outputs. ``fb == 1`` broadcasts over the batch."""
    _, heads, seq_len, head_dim = xq.shape
    head_pairs = head_dim // 2
    f_batch_stride = 0 if freqs_cis.shape[0] == 1 else seq_len * head_pairs * 4
    n_pairs_total = xq.numel() // 2
    block = _pick_block(n_pairs_total)
    grid = (triton.cdiv(n_pairs_total, block),)
    _rope_kernel[grid](
        xq,
        xk,
        freqs_cis,
        out_q,
        out_k,
        n_pairs_total,
        seq_len,
        f_batch_stride,
        HEAD_PAIRS=head_pairs,  # pyright: ignore[reportArgumentType]
        HEADS=heads,  # pyright: ignore[reportArgumentType]
        BLOCK=block,  # pyright: ignore[reportArgumentType]
    )
