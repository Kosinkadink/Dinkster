"""Triton kernel for ConvRot INT8 weight dequantization."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_GROUP_SIZE = tl.constexpr(256)


@triton.jit
def _h4(x0, x1, x2, x3):
    return tl.inline_asm_elementwise(
        asm="""
        add.rn.f32 $0, $4, $5;
        add.rn.f32 $0, $0, $6;
        sub.rn.f32 $0, $0, $7;
        mul.rn.f32 $0, $0, 0f3F000000;
        add.rn.f32 $1, $4, $5;
        sub.rn.f32 $1, $1, $6;
        add.rn.f32 $1, $1, $7;
        mul.rn.f32 $1, $1, 0f3F000000;
        sub.rn.f32 $2, $4, $5;
        add.rn.f32 $2, $2, $6;
        add.rn.f32 $2, $2, $7;
        mul.rn.f32 $2, $2, 0f3F000000;
        neg.f32 $3, $4;
        add.rn.f32 $3, $3, $5;
        add.rn.f32 $3, $3, $6;
        add.rn.f32 $3, $3, $7;
        mul.rn.f32 $3, $3, 0f3F000000;
        """,
        constraints="=f,=f,=f,=f,f,f,f,f",
        args=[x0, x1, x2, x3],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _radix4_stage(values, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    blocks: tl.constexpr = _GROUP_SIZE // (4 * STRIDE)
    grouped = tl.reshape(values, (GROUPS, blocks, 4, STRIDE))
    grouped = tl.permute(grouped, (0, 1, 3, 2))

    pairs = tl.reshape(grouped, (GROUPS * blocks * STRIDE * 2, 2))
    even, odd = tl.split(pairs)
    even = tl.reshape(even, (GROUPS * blocks * STRIDE, 2))
    odd = tl.reshape(odd, (GROUPS * blocks * STRIDE, 2))
    x0, x2 = tl.split(even)
    x1, x3 = tl.split(odd)

    y0, y1, y2, y3 = _h4(x0, x1, x2, x3)  # pyright: ignore[reportGeneralTypeIssues]

    y02 = tl.interleave(y0, y2)
    y13 = tl.interleave(y1, y3)
    packed = tl.interleave(y02, y13)
    grouped = tl.reshape(packed, (GROUPS, blocks, STRIDE, 4))
    grouped = tl.permute(grouped, (0, 1, 3, 2))
    return tl.reshape(grouped, (GROUPS * _GROUP_SIZE,))


@triton.jit
def _dequantize_int8_convrot_kernel(
    q_ptr,
    scale_ptr,
    out_ptr,
    columns,
    GROUPS: tl.constexpr,
):
    group_block = tl.program_id(axis=0)
    row = tl.program_id(axis=1)
    offsets = (
        row * columns + group_block * GROUPS * _GROUP_SIZE + tl.arange(0, GROUPS * _GROUP_SIZE)
    )
    mask = offsets < (row + 1) * columns
    scale = tl.load(scale_ptr + row)
    values = tl.load(q_ptr + offsets, mask=mask, other=0).to(tl.float32) * scale
    values = _radix4_stage(values, GROUPS=GROUPS, STRIDE=1)
    values = _radix4_stage(values, GROUPS=GROUPS, STRIDE=4)
    values = _radix4_stage(values, GROUPS=GROUPS, STRIDE=16)
    values = _radix4_stage(values, GROUPS=GROUPS, STRIDE=64)
    tl.store(out_ptr + offsets, values, mask=mask)


def launch_dequantize_int8_convrot(
    qdata: torch.Tensor, scale: torch.Tensor, output: torch.Tensor
) -> None:
    """Dequantize contiguous rowwise INT8 storage into float32."""
    rows, columns = qdata.shape
    groups = columns // int(_GROUP_SIZE)
    groups_per_program = 4 if groups >= 4 else 2 if groups >= 2 else 1
    grid = (triton.cdiv(groups, groups_per_program), rows)
    _dequantize_int8_convrot_kernel[grid](
        qdata,
        scale,
        output,
        columns=columns,  # pyright: ignore[reportArgumentType]
        GROUPS=groups_per_program,  # pyright: ignore[reportArgumentType]
        num_warps=max(4, groups_per_program * 2),  # pyright: ignore[reportCallIssue]
    )
