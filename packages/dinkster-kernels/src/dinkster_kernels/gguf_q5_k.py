"""Fused Q5_K GGUF operators.

Q5_K packs 256 logical weight elements into one 176-byte super-block:
the Q4_K scale layout (float16 d and dmin plus 12 bytes of packed
6-bit (scale, min) pairs), a 32-byte high-bit plane, and 128
low-nibble bytes as four 32-byte chunks. Each element's five-bit
quant is its nibble plus bit 2c + h of high-plane byte e % 32, and
decodes as q * (d * sc) - (dmin * mn). Both custom ops here consume
the packed uint8 blocks directly on the device:

- ``gguf_q5_k_decode`` materializes the float32 weight, bit-identical
  to the vectorized torch decoder (both products are exact in float32
  and the subtraction is the one rounding on both sides).
- ``gguf_q5_k_linear`` decodes blocks inside the matmul tiles instead
  of materializing the weight. Per-element weight values are the same
  exact decode cast to the input dtype; only the tile accumulation
  order differs from decode-then-``F.linear``, so outputs are
  value-close, not bit-equal, to that reference.

Triton imports lazily inside the CUDA implementations, so importing
this module (which registers the op schemas) needs neither triton nor
a CUDA device. Route eligibility goes through
:func:`gguf_q5_k_linear_available`, the shared cached host probe: a
host without a C compiler for triton's launcher build is thereby
ineligible, never broken.
"""

from __future__ import annotations

import torch

from . import _common

Q5_K_BLOCK_ELEMENTS = _common.Q5_K.block_elements
Q5_K_BLOCK_BYTES = _common.Q5_K.block_bytes


@torch.library.custom_op("dinkster_kernels::gguf_q5_k_decode", mutates_args=(), device_types="cuda")
def gguf_q5_k_decode(blocks: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Decode Q5_K blocks to a float32 ``(out_features, in_features)`` weight."""

    return _common.run_decode(_common.Q5_K, blocks, out_features, in_features)


@gguf_q5_k_decode.register_fake
def _gguf_q5_k_decode_fake(  # pyright: ignore[reportUnusedFunction] - registered on the op
    blocks: torch.Tensor, out_features: int, in_features: int
) -> torch.Tensor:
    return blocks.new_empty((out_features, in_features), dtype=torch.float32)


@torch.library.custom_op("dinkster_kernels::gguf_q5_k_linear", mutates_args=(), device_types="cuda")
def gguf_q5_k_linear(
    input: torch.Tensor,
    blocks: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
) -> torch.Tensor:
    """``F.linear`` over a Q5_K-encoded weight, decoded inside the matmul.

    ``input`` is ``(..., in_features)`` float16 or bfloat16; the
    result is ``(..., out_features)`` in the input dtype. Weight
    values decode exactly as the reference decoder and are cast to
    the input dtype before float32 tile accumulation. The op is
    inference-only: it registers no autograd formula.
    """

    return _common.run_linear(_common.Q5_K, input, blocks, bias, out_features)


@gguf_q5_k_linear.register_fake
def _gguf_q5_k_linear_fake(  # pyright: ignore[reportUnusedFunction] - registered on the op
    input: torch.Tensor,
    blocks: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], out_features))


def gguf_q5_k_linear_available() -> bool:
    """Whether the fused Q5_K ops can execute on this host (shared probe)."""

    return _common.fused_ops_available()
