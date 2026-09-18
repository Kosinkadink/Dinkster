"""Fused Q6_K GGUF operators.

Q6_K packs 256 logical weight elements into one 210-byte super-block:
128 low-nibble bytes (two 64-byte half planes), 64 two-bit bytes (two
32-byte half planes), sixteen int8 group scales, and a float16 super
scale d. Each element's six-bit quant is its nibble plus two high
bits, decoded as (d * sc) * (q - 32) under one of the sixteen
16-element group scales. Both custom ops here consume the packed
uint8 blocks directly on the device:

- ``gguf_q6_k_decode`` materializes the float32 weight, bit-identical
  to the vectorized torch decoder (the super-scale product and the
  integer subtraction are exact in float32; the final multiply is the
  one rounding on both sides).
- ``gguf_q6_k_linear`` decodes blocks inside the matmul tiles instead
  of materializing the weight. Per-element weight values are the same
  exact decode cast to the input dtype; only the tile accumulation
  order differs from decode-then-``F.linear``, so outputs are
  value-close, not bit-equal, to that reference.

Triton imports lazily inside the CUDA implementations, so importing
this module (which registers the op schemas) needs neither triton nor
a CUDA device. Route eligibility goes through
:func:`gguf_q6_k_linear_available`, the shared cached host probe: a
host without a C compiler for triton's launcher build is thereby
ineligible, never broken.
"""

from __future__ import annotations

import torch

from . import _common

Q6_K_BLOCK_ELEMENTS = _common.Q6_K.block_elements
Q6_K_BLOCK_BYTES = _common.Q6_K.block_bytes


@torch.library.custom_op("dinkster_kernels::gguf_q6_k_decode", mutates_args=(), device_types="cuda")
def gguf_q6_k_decode(blocks: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Decode Q6_K blocks to a float32 ``(out_features, in_features)`` weight."""

    return _common.run_decode(_common.Q6_K, blocks, out_features, in_features)


@gguf_q6_k_decode.register_fake
def _gguf_q6_k_decode_fake(  # pyright: ignore[reportUnusedFunction] - registered on the op
    blocks: torch.Tensor, out_features: int, in_features: int
) -> torch.Tensor:
    return blocks.new_empty((out_features, in_features), dtype=torch.float32)


@torch.library.custom_op("dinkster_kernels::gguf_q6_k_linear", mutates_args=(), device_types="cuda")
def gguf_q6_k_linear(
    input: torch.Tensor,
    blocks: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
) -> torch.Tensor:
    """``F.linear`` over a Q6_K-encoded weight, decoded inside the matmul.

    ``input`` is ``(..., in_features)`` float16 or bfloat16; the
    result is ``(..., out_features)`` in the input dtype. Weight
    values decode exactly as the reference decoder and are cast to
    the input dtype before float32 tile accumulation. The op is
    inference-only: it registers no autograd formula.
    """

    return _common.run_linear(_common.Q6_K, input, blocks, bias, out_features)


@gguf_q6_k_linear.register_fake
def _gguf_q6_k_linear_fake(  # pyright: ignore[reportUnusedFunction] - registered on the op
    input: torch.Tensor,
    blocks: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], out_features))


def gguf_q6_k_linear_available() -> bool:
    """Whether the fused Q6_K ops can execute on this host (shared probe)."""

    return _common.fused_ops_available()
