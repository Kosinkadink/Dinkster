"""Triton kernels behind the fused GGUF ops.

Import this module only where triton is importable; the host-side op
implementations import it lazily per call.

Both kernels decode through per-layout device functions selected by a
compile-time layout id (the ``triton_id`` of the layout descriptors in
``_common``), so each instantiation inlines exactly one decode with no
runtime dispatch. Every layout performs its single float32 rounding at
the same point as the corresponding vectorized torch decoder, so the
decode is bit-identical to it. The linear kernel then casts decoded
values to the input dtype and accumulates ``tl.dot`` tiles in
float32 - the same per-element weight values as decode-then-
``F.linear``, with only tile accumulation order differing.

Memory layout: every supported block size is even, so a block is
whole little-endian uint16 lanes (17 for Q8_0's 34 bytes, 9 for
Q4_0's 18, 72 for Q4_K's 144, 88 for Q5_K's 176, 105 for Q6_K's
210). Rows start 2-byte
aligned (even row size over an aligned base, which the host launchers
guarantee), so the kernels read uint16 lanes instead of single bytes,
halving load count and letting consecutive lanes coalesce into wider
transactions.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_DECODE_PAIRS = 512  # quant pairs per program: 1024 decoded elements
_LINEAR_BLOCK_N = 64

# Linear tile configs by input row count, chosen by measurement on
# text-encoder/diffusion shapes (RTX 5090): small M wants deep K tiles
# to keep the few programs busy, large M wants tall M tiles so each
# weight block decodes fewer times. (BLOCK_M, BLOCK_K, num_warps);
# BLOCK_N stays 64 and num_stages 3 throughout. The 256-element
# K-quant superblocks decode with more arithmetic per element than
# the 32-element layouts, so their mid/large-M sweet spot sits at
# wider K tiles ((64|128, 64, 4) instead of (64|128, 32, 4)).
_LINEAR_CONFIGS_32 = (
    (32, (16, 128, 8)),
    (128, (32, 64, 4)),
    (256, (64, 32, 4)),
)
_LINEAR_CONFIG_LARGE_32 = (128, 32, 4)
_LINEAR_CONFIGS_KQUANT = (
    (32, (16, 128, 8)),
    (256, (64, 64, 4)),
)
_LINEAR_CONFIG_LARGE_KQUANT = (128, 64, 4)


@triton.jit
def _decode_q8_0_pairs(blocks16_ptr, block_index, pair_index, mask):
    """Decode quant pair (2*pair, 2*pair + 1) of each indexed Q8_0 block.

    A block is 17 uint16 lanes: lane 0 bitcasts to the float16 scale,
    lanes 1..16 pack the 32 int8 quants two per lane (low byte first).
    Masked-off lanes load zero and decode to exact 0.0. Returns the
    decoded float32 values interleaved along the last axis
    (..., 2 * pairs), restoring element order.
    """
    base = blocks16_ptr + block_index.to(tl.int64) * 17
    scale = tl.load(base, mask=mask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
    pair = tl.load(base + 1 + pair_index, mask=mask, other=0)
    low = (pair & 0xFF).to(tl.uint8).to(tl.int8, bitcast=True)
    high = (pair >> 8).to(tl.uint8).to(tl.int8, bitcast=True)
    return tl.interleave(low.to(tl.float32) * scale, high.to(tl.float32) * scale)


@triton.jit
def _decode_q4_0_pairs(blocks16_ptr, block_index, pair_index, mask):
    """Decode quant pair (2*pair, 2*pair + 1) of each indexed Q4_0 block.

    A block is 9 uint16 lanes: lane 0 bitcasts to the float16 scale,
    lanes 1..8 hold the 16 packed quant bytes. Elements 0..15 are the
    low nibbles of bytes 0..15 and elements 16..31 the high nibbles,
    so pair p reads bytes 2*(p % 8) and 2*(p % 8) + 1 (one lane) and
    p // 8 selects the nibble half. Each element decodes as
    scale * (nibble - 8): the subtraction is exact and the multiply
    is the one rounding, matching the torch decoder. Masked-off lanes
    decode to scale 0.0 times -8.0, an exact +/-0.0 that is never
    stored and contributes exact zero to dot products.
    """
    base = blocks16_ptr + block_index.to(tl.int64) * 9
    scale = tl.load(base, mask=mask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
    lane = tl.load(base + 1 + pair_index % 8, mask=mask, other=0)
    even_byte = lane & 0xFF
    odd_byte = lane >> 8
    high_half = pair_index >= 8
    even = tl.where(high_half, even_byte >> 4, even_byte & 0xF).to(tl.int32) - 8
    odd = tl.where(high_half, odd_byte >> 4, odd_byte & 0xF).to(tl.int32) - 8
    return tl.interleave(even.to(tl.float32) * scale, odd.to(tl.float32) * scale)


@triton.jit
def _kquant_group_scale_min(base, group, mask):
    """Unpack K-quant group ``group``'s 6-bit (scale, min) pair and fold
    in the super-block's float16 scales.

    ``base`` points at each indexed super-block's first lane: lane 0
    bitcasts to the float16 super scale d, lane 1 to the min scale
    dmin, and lanes 2..7 hold the 12 packed scale bytes. Groups 0..3
    take their 6 scale bits from the low six of bytes 0..3 (mins from
    bytes 4..7); groups 4..7 pack the low nibble in bytes 8..11 (mins
    the high nibble) with the top two bits spilled into the high two
    of bytes 0..7. Returns (d * sc, dmin * mn) as float32; both
    products are exact (a float16 significand times a 6-bit integer),
    so the caller's subtraction is the single rounding. Masked-off
    lanes load zero and yield exact zero products.
    """
    d = tl.load(base, mask=mask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
    dmin = tl.load(base + 1, mask=mask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
    j = group & 3
    shift = (j & 1) * 8
    low = (tl.load(base + 2 + (j >> 1), mask=mask, other=0) >> shift) & 0xFF
    mid = (tl.load(base + 4 + (j >> 1), mask=mask, other=0) >> shift) & 0xFF
    high = (tl.load(base + 6 + (j >> 1), mask=mask, other=0) >> shift) & 0xFF
    spilled = group >= 4
    sc = tl.where(spilled, (high & 0x0F) | ((low >> 6) << 4), low & 0x3F)
    mn = tl.where(spilled, (high >> 4) | ((mid >> 6) << 4), mid & 0x3F)
    return d * sc.to(tl.float32), dmin * mn.to(tl.float32)


@triton.jit
def _decode_q4_k_pairs(blocks16_ptr, block_index, pair_index, mask):
    """Decode quant pair (2*pair, 2*pair + 1) of each indexed Q4_K block.

    A super-block is 72 uint16 lanes: two float16 scales, the packed
    6-bit (scale, min) pairs (see ``_kquant_group_scale_min``), then
    128 quant bytes as four 32-byte chunks at lanes 8 + 16c. Chunk c's
    32 low nibbles are elements 64c..64c+31 (group 2c) and its high
    nibbles the next 32 (group 2c + 1), so pair p sits in chunk
    p // 32 at lane offset p % 16, with bit 4 of p selecting the
    nibble half. Each element decodes as q * (d * sc) - (dmin * mn):
    both products are exact in float32 and the subtraction is the one
    rounding, matching the torch decoder. Masked-off lanes decode to
    0.0 - 0.0, an exact zero.
    """
    base = blocks16_ptr + block_index.to(tl.int64) * 72
    # The jit stub types device-function calls NoReturn; the tuple
    # unpack is real inside the kernel.
    gs, gm = _kquant_group_scale_min(base, pair_index >> 4, mask)  # pyright: ignore[reportGeneralTypeIssues]
    lane = tl.load(base + 8 + (pair_index >> 5) * 16 + (pair_index & 15), mask=mask, other=0)
    high_half = (pair_index >> 4) & 1
    even = tl.where(high_half == 1, (lane & 0xFF) >> 4, lane & 0xF)
    odd = tl.where(high_half == 1, lane >> 12, (lane >> 8) & 0xF)
    return tl.interleave(even.to(tl.float32) * gs - gm, odd.to(tl.float32) * gs - gm)


@triton.jit
def _decode_q5_k_pairs(blocks16_ptr, block_index, pair_index, mask):
    """Decode quant pair (2*pair, 2*pair + 1) of each indexed Q5_K block.

    A super-block is 88 uint16 lanes: the Q4_K scale layout, a
    32-byte high-bit plane at lanes 8..23, then 128 low-nibble bytes
    at lanes 24..87 chunked exactly like Q4_K. Element e's fifth quant
    bit is bit g of high-plane byte e % 32 - pair p's two elements
    share plane lane 8 + (p % 16) - where g is the element's
    group 2c + h. Each element decodes as q * (d * sc) - (dmin * mn)
    with q = nibble | (bit << 4); the subtraction is the one rounding,
    matching the torch decoder.
    """
    base = blocks16_ptr + block_index.to(tl.int64) * 88
    group = pair_index >> 4
    # The jit stub types device-function calls NoReturn; the tuple
    # unpack is real inside the kernel.
    gs, gm = _kquant_group_scale_min(base, group, mask)  # pyright: ignore[reportGeneralTypeIssues]
    lane = tl.load(base + 24 + (pair_index >> 5) * 16 + (pair_index & 15), mask=mask, other=0)
    bits = tl.load(base + 8 + (pair_index & 15), mask=mask, other=0)
    high_half = group & 1
    even = tl.where(high_half == 1, (lane & 0xFF) >> 4, lane & 0xF)
    odd = tl.where(high_half == 1, lane >> 12, (lane >> 8) & 0xF)
    even = even | ((((bits & 0xFF) >> group) & 1) << 4)
    odd = odd | ((((bits >> 8) >> group) & 1) << 4)
    return tl.interleave(even.to(tl.float32) * gs - gm, odd.to(tl.float32) * gs - gm)


@triton.jit
def _decode_q6_k_pairs(blocks16_ptr, block_index, pair_index, mask):
    """Decode quant pair (2*pair, 2*pair + 1) of each indexed Q6_K block.

    A super-block is 105 uint16 lanes: 128 low-nibble bytes at lanes
    0..63 (two 64-byte half planes), 64 two-bit bytes at lanes 64..95
    (two 32-byte half planes), 16 int8 group scales at lanes 96..103,
    and the float16 super scale d at lane 104. Within half h, row r's
    element l takes nibble r // 2 of byte 64h + 32(r % 2) + l and bits
    2r of two-bit byte 32h + l, under group scale 8h + 2r + l // 16 -
    for pair p that is h = p // 64, r = (p // 16) % 4, l = 2(p % 16)
    (+ 1), and scale index p // 8. Each element decodes as
    (d * sc) * (q - 32): the super-scale product (float16 times int8)
    and the integer subtraction are exact in float32, so the final
    multiply is the one rounding, matching the torch decoder.
    Masked-off lanes decode to 0.0 * -32.0, an exact -0.0 that is
    never stored and contributes exact zero to dot products.
    """
    base = blocks16_ptr + block_index.to(tl.int64) * 105
    d = tl.load(base + 104, mask=mask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
    sc_lane = tl.load(base + 96 + (pair_index >> 4), mask=mask, other=0)
    sc_byte = (sc_lane >> (((pair_index >> 3) & 1) * 8)) & 0xFF
    gs = d * sc_byte.to(tl.uint8).to(tl.int8, bitcast=True).to(tl.float32)
    half = pair_index >> 6
    row = (pair_index >> 4) & 3
    lane = tl.load(base + half * 32 + (row & 1) * 16 + (pair_index & 15), mask=mask, other=0)
    bits = tl.load(base + 64 + half * 16 + (pair_index & 15), mask=mask, other=0)
    high_nibble = row >> 1
    even = tl.where(high_nibble == 1, (lane & 0xFF) >> 4, lane & 0xF)
    odd = tl.where(high_nibble == 1, lane >> 12, (lane >> 8) & 0xF)
    shift = row * 2
    even = even | ((((bits & 0xFF) >> shift) & 3) << 4)
    odd = odd | ((((bits >> 8) >> shift) & 3) << 4)
    even_q = even.to(tl.int32) - 32
    odd_q = odd.to(tl.int32) - 32
    return tl.interleave(even_q.to(tl.float32) * gs, odd_q.to(tl.float32) * gs)


@triton.jit
def _decode_pairs(blocks16_ptr, block_index, pair_index, mask, LAYOUT: tl.constexpr):
    """Dispatch pair decode on the compile-time layout id.

    The constexpr comparison prunes at compile time, so each kernel
    instantiation contains exactly one layout's decode.
    """
    if LAYOUT == 0:  # _common.Q8_0.triton_id
        return _decode_q8_0_pairs(blocks16_ptr, block_index, pair_index, mask)
    elif LAYOUT == 1:  # _common.Q4_0.triton_id
        return _decode_q4_0_pairs(blocks16_ptr, block_index, pair_index, mask)
    elif LAYOUT == 2:  # _common.Q4_K.triton_id
        return _decode_q4_k_pairs(blocks16_ptr, block_index, pair_index, mask)
    elif LAYOUT == 3:  # _common.Q5_K.triton_id
        return _decode_q5_k_pairs(blocks16_ptr, block_index, pair_index, mask)
    else:  # _common.Q6_K.triton_id
        return _decode_q6_k_pairs(blocks16_ptr, block_index, pair_index, mask)


@triton.jit
def _gguf_decode_kernel(
    blocks16_ptr,
    out_ptr,
    total_pairs,
    LAYOUT: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    PAIRS: tl.constexpr,
):
    # int64 offsets: int32 index math would wrap past 2^31 elements.
    pairs = tl.program_id(0).to(tl.int64) * PAIRS + tl.arange(0, PAIRS)
    mask = pairs < total_pairs
    # Pair p holds elements 2p and 2p + 1, so the interleaved decode
    # is already in element order and the flat store stays coalesced.
    values = _decode_pairs(blocks16_ptr, pairs // BLOCK_PAIRS, pairs % BLOCK_PAIRS, mask, LAYOUT)
    offsets = tl.program_id(0).to(tl.int64) * (2 * PAIRS) + tl.arange(0, 2 * PAIRS)
    tl.store(out_ptr + offsets, values, mask=tl.interleave(mask, mask))


@triton.jit
def _gguf_linear_kernel(
    x_ptr,
    blocks16_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_om,
    stride_on,
    LAYOUT: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # int64 row/column offsets: they multiply strides and blocks_per_row,
    # where int32 math would wrap past 2^31 addressed elements.
    offs_m = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_pair = tl.arange(0, BLOCK_K // 2)
    m_mask = offs_m < M
    n_mask = offs_n < N
    pairs_per_row = K // 2
    blocks_per_row = K // (2 * BLOCK_PAIRS)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        k_mask = k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # Weight tile as (BLOCK_N, BLOCK_K): logical element (n, k)
        # lives in block n * blocks_per_row + k // block_elements, and
        # its quant pair is pair k // 2 of the row.
        pair = k_start // 2 + offs_pair
        # K is a multiple of the block size, so a pair never straddles
        # the K edge.
        pair_mask = n_mask[:, None] & (pair[None, :] < pairs_per_row)
        wt = _decode_pairs(
            blocks16_ptr,
            offs_n[:, None] * blocks_per_row + pair[None, :] // BLOCK_PAIRS,
            pair[None, :] % BLOCK_PAIRS,
            pair_mask,
            LAYOUT,
        )
        acc = tl.dot(x, tl.trans(wt).to(x.dtype), acc)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _as_uint16_lanes(blocks: torch.Tensor) -> torch.Tensor:
    """View the contiguous uint8 block table as uint16 lanes.

    The kernels' uint16 loads need a 2-byte-aligned base. Fresh CUDA
    allocations are always aligned; a view starting at an odd byte
    offset is realigned by cloning.
    """
    if blocks.data_ptr() % 2:
        blocks = blocks.clone()
    return blocks.view(torch.uint16)


def launch_decode(
    layout_id: int, block_pairs: int, blocks: torch.Tensor, out: torch.Tensor
) -> None:
    """Decode contiguous GGUF ``blocks`` into flat float32 ``out``."""
    total_pairs = out.numel() // 2
    grid = (triton.cdiv(total_pairs, _DECODE_PAIRS),)
    # Launch-time constexpr kwargs take plain ints/bools; triton wraps
    # them (the stubs declare tl.constexpr, hence the ignores).
    _gguf_decode_kernel[grid](
        _as_uint16_lanes(blocks),
        out,
        total_pairs,
        LAYOUT=layout_id,  # pyright: ignore[reportArgumentType]
        BLOCK_PAIRS=block_pairs,  # pyright: ignore[reportArgumentType]
        PAIRS=_DECODE_PAIRS,  # pyright: ignore[reportArgumentType]
    )


def launch_linear(
    layout_id: int,
    block_pairs: int,
    x: torch.Tensor,
    blocks: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
) -> None:
    """Run the fused linear: ``x (M, K)`` against blocks into ``out (M, N)``.

    The tile config follows the row count, so the float32 accumulation
    order over K can differ between row counts (as it already differs
    from decode-then-``F.linear``); for a fixed input shape the kernel
    is deterministic.
    """
    m, k = x.shape
    n = out.shape[1]
    if block_pairs > 16:  # 256-element K-quant superblocks
        configs, config_large = _LINEAR_CONFIGS_KQUANT, _LINEAR_CONFIG_LARGE_KQUANT
    else:
        configs, config_large = _LINEAR_CONFIGS_32, _LINEAR_CONFIG_LARGE_32
    for limit, config in configs:
        if m <= limit:
            block_m, block_k, num_warps = config
            break
    else:
        block_m, block_k, num_warps = config_large
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, _LINEAR_BLOCK_N))
    _gguf_linear_kernel[grid](
        x,
        _as_uint16_lanes(blocks),
        # Dead pointer when HAS_BIAS is False (the load is compiled out).
        bias if bias is not None else x,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        LAYOUT=layout_id,  # pyright: ignore[reportArgumentType]
        BLOCK_PAIRS=block_pairs,  # pyright: ignore[reportArgumentType]
        HAS_BIAS=bias is not None,  # pyright: ignore[reportArgumentType]
        BLOCK_M=block_m,  # pyright: ignore[reportArgumentType]
        BLOCK_N=_LINEAR_BLOCK_N,  # pyright: ignore[reportArgumentType]
        BLOCK_K=block_k,  # pyright: ignore[reportArgumentType]
        num_warps=num_warps,  # pyright: ignore[reportCallIssue] - launch meta-parameter
        num_stages=3,  # pyright: ignore[reportCallIssue] - launch meta-parameter
    )
