"""Fused Q8_0 GGUF operators.

Q8_0 packs 32 logical weight elements into one 34-byte block: a
float16 scale followed by 32 int8 quants, decoding as quant times
scale. Both custom ops here consume the packed uint8 blocks directly
on the device:

- ``gguf_q8_0_decode`` materializes the float32 weight, bit-identical
  to the vectorized torch decoder (the quant-times-scale product is
  exact in float32, so there is exactly one rounding on both sides).
- ``gguf_q8_0_linear`` decodes blocks inside the matmul tiles instead
  of materializing the weight. Per-element weight values are the same
  exact decode cast to the input dtype; only the tile accumulation
  order differs from decode-then-``F.linear``, so outputs are
  value-close, not bit-equal, to that reference.

Triton imports lazily inside the CUDA implementations, so importing
this module (which registers the op schemas) needs neither triton nor
a CUDA device. Route eligibility goes through
:func:`gguf_q8_0_linear_available`, the shared cached host probe: a
host without a C compiler for triton's launcher build is thereby
ineligible, never broken.
"""

from __future__ import annotations

import torch

from . import _common

Q8_0_BLOCK_ELEMENTS = _common.Q8_0.block_elements
Q8_0_BLOCK_BYTES = _common.Q8_0.block_bytes


@torch.library.custom_op("dinkster_kernels::gguf_q8_0_decode", mutates_args=(), device_types="cuda")
def gguf_q8_0_decode(blocks: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Decode Q8_0 blocks to a float32 ``(out_features, in_features)`` weight."""

    return _common.run_decode(_common.Q8_0, blocks, out_features, in_features)


@gguf_q8_0_decode.register_fake
def _gguf_q8_0_decode_fake(  # pyright: ignore[reportUnusedFunction] - registered on the op
    blocks: torch.Tensor, out_features: int, in_features: int
) -> torch.Tensor:
    return blocks.new_empty((out_features, in_features), dtype=torch.float32)


@torch.library.custom_op("dinkster_kernels::gguf_q8_0_linear", mutates_args=(), device_types="cuda")
def gguf_q8_0_linear(
    input: torch.Tensor,
    blocks: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
) -> torch.Tensor:
    """``F.linear`` over a Q8_0-encoded weight, decoded inside the matmul.

    ``input`` is ``(..., in_features)`` float16 or bfloat16; the
    result is ``(..., out_features)`` in the input dtype. Weight
    values decode exactly as the reference decoder and are cast to
    the input dtype before float32 tile accumulation. The op is
    inference-only: it registers no autograd formula.
    """

    return _common.run_linear(_common.Q8_0, input, blocks, bias, out_features)


@gguf_q8_0_linear.register_fake
def _gguf_q8_0_linear_fake(  # pyright: ignore[reportUnusedFunction] - registered on the op
    input: torch.Tensor,
    blocks: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], out_features))


def gguf_q8_0_linear_available() -> bool:
    """Whether the fused Q8_0 ops can execute on this host (shared probe)."""

    return _common.fused_ops_available()
