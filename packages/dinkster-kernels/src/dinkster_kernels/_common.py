"""Shared host-side machinery for the fused GGUF ops.

Layout descriptors, argument validation, and the cached availability
probe live here so each per-layout op module stays a thin schema
wrapper. Nothing here imports triton at module import time.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch

_HALF_DTYPES = (torch.float16, torch.bfloat16)


@dataclass(frozen=True)
class GgufLayout:
    """One GGML block layout the fused kernels understand.

    ``triton_id`` selects the decode device function at kernel compile
    time (the LAYOUT constexpr in ``_gguf_triton``).
    """

    ggml_type: str
    triton_id: int
    block_bytes: int
    block_elements: int

    @property
    def block_pairs(self) -> int:
        return self.block_elements // 2


Q8_0 = GgufLayout("Q8_0", 0, 34, 32)
Q4_0 = GgufLayout("Q4_0", 1, 18, 32)
Q4_K = GgufLayout("Q4_K", 2, 144, 256)
Q5_K = GgufLayout("Q5_K", 3, 176, 256)
Q6_K = GgufLayout("Q6_K", 4, 210, 256)


def validate_blocks(
    layout: GgufLayout, blocks: torch.Tensor, out_features: int, in_features: int
) -> None:
    t = layout.ggml_type
    if blocks.dtype != torch.uint8 or blocks.ndim != 2 or blocks.shape[1] != layout.block_bytes:
        raise ValueError(
            f"{t} blocks must be uint8 of shape (block_count,"
            f" {layout.block_bytes}), got {blocks.dtype} of shape {tuple(blocks.shape)}"
        )
    if not blocks.is_contiguous():
        raise ValueError(f"{t} blocks must be contiguous")
    if out_features <= 0 or in_features <= 0 or in_features % layout.block_elements:
        raise ValueError(
            f"a fused {t} weight of {out_features}x{in_features} does not split"
            f" into rows of whole {layout.block_elements}-element blocks"
        )
    if blocks.shape[0] * layout.block_elements != out_features * in_features:
        raise ValueError(
            f"{blocks.shape[0]} {t} blocks do not hold a {out_features}x{in_features} weight"
        )


def validate_linear_args(
    layout: GgufLayout,
    input: torch.Tensor,
    blocks: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
) -> torch.Tensor | None:
    """Validate fused-linear arguments; returns the bias to use
    (made contiguous if needed, since the kernel reads it at unit
    stride)."""
    t = layout.ggml_type
    validate_blocks(layout, blocks, out_features, input.shape[-1])
    if input.dtype not in _HALF_DTYPES:
        raise ValueError(f"fused {t} linear computes at float16 or bfloat16, got {input.dtype}")
    if blocks.device != input.device:
        raise ValueError(
            f"fused {t} linear needs blocks on the input device, got {blocks.device}"
            f" vs {input.device}"
        )
    if bias is not None:
        if bias.dtype != input.dtype or bias.shape != (out_features,):
            raise ValueError(
                f"fused {t} linear bias must be a 1-D tensor of out_features"
                " values in the input dtype"
            )
        if bias.device != input.device:
            raise ValueError(
                f"fused {t} linear needs the bias on the input device, got {bias.device}"
                f" vs {input.device}"
            )
        if not bias.is_contiguous():
            bias = bias.contiguous()
    return bias


def run_linear(
    layout: GgufLayout,
    input: torch.Tensor,
    blocks: torch.Tensor,
    bias: torch.Tensor | None,
    out_features: int,
) -> torch.Tensor:
    """Shared fused-linear body: validate, flatten, launch, reshape."""
    bias = validate_linear_args(layout, input, blocks, bias, out_features)
    from . import _gguf_triton as kernels

    in_features = input.shape[-1]
    x = input.reshape(-1, in_features)
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty((x.shape[0], out_features), device=input.device, dtype=input.dtype)
    if x.shape[0]:
        kernels.launch_linear(layout.triton_id, layout.block_pairs, x, blocks, bias, out)
    return out.reshape(*input.shape[:-1], out_features)


def run_decode(
    layout: GgufLayout, blocks: torch.Tensor, out_features: int, in_features: int
) -> torch.Tensor:
    """Shared decode body: validate, launch, reshape."""
    validate_blocks(layout, blocks, out_features, in_features)
    from . import _gguf_triton as kernels

    out = torch.empty(out_features * in_features, device=blocks.device, dtype=torch.float32)
    kernels.launch_decode(layout.triton_id, layout.block_pairs, blocks, out)
    return out.reshape(out_features, in_features)


_available: bool | None = None


def fused_ops_available() -> bool:
    """Whether the fused GGUF ops can execute on this host.

    True only after one cached end-to-end probe succeeds: a CUDA
    device is visible, triton imports, and a one-block decode kernel
    compiles, runs, and returns the exact expected values (an
    all-zero block decodes to exact zeros). Kernel compilation needs
    a host C compiler for triton's launcher; any failure makes every
    fused route ineligible rather than raising. The probe checks host
    capability once through the Q8_0 decode; per-layout kernel
    correctness is pinned by the test suites, not probed per call.
    """

    global _available
    if _available is None:
        _available = _probe()
    return _available


def _probe() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        importlib.import_module("triton")
        from .gguf_q8_0 import gguf_q8_0_decode

        blocks = torch.zeros((1, Q8_0.block_bytes), dtype=torch.uint8, device="cuda")
        decoded = gguf_q8_0_decode(blocks, 1, Q8_0.block_elements)
        return decoded.shape == (1, Q8_0.block_elements) and bool((decoded == 0).all())
    except Exception:  # noqa: BLE001 - capability probes are best-effort
        return False
