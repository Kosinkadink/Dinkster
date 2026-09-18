"""Combined RoPE rotation as a Dinkster-owned custom op.

``apply_rope`` rotates query and key through precomputed 2x2 rotation
frequencies in one launch, bit-identical to the reference pure-torch
rotation (comfy/ldm/flux/math.py _apply_rope1 @ b78cec87): float32
compute with one rounded multiply, one fused multiply-add matching
``torch.addcmul``, and one rounding back to the input dtype. Kernel
details and the contraction analysis live in :mod:`_rope_triton`.

Triton imports lazily inside the CUDA implementation, so importing
this module (which registers the op schema) needs neither triton nor
a CUDA device. Route eligibility goes through
:func:`apply_rope_available` (the shared cached host probe) plus
:func:`apply_rope_supported` for the per-call shape contract;
consumers keep a reference fallback for anything unsupported. The op
is inference-only: it registers no autograd formula.
"""

from __future__ import annotations

import torch

from . import _common

_X_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _rope_args_error(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> str | None:
    """The shape contract shared by validation and eligibility: q/k are
    ``(B, H, S, D)`` half or float32 with even D, frequencies are
    float32 ``(fb, 1, S, D/2, 2, 2)`` with fb of 1 or B, all on one
    device."""
    if xq.shape != xk.shape or xq.dtype != xk.dtype:
        return (
            f"xq and xk must match, got {tuple(xq.shape)} {xq.dtype}"
            f" vs {tuple(xk.shape)} {xk.dtype}"
        )
    if xq.ndim != 4 or xq.shape[-1] % 2:
        return f"xq/xk must be (batch, heads, seq, even head_dim), got {tuple(xq.shape)}"
    if xq.dtype not in _X_DTYPES:
        return f"xq/xk must be float16, bfloat16, or float32, got {xq.dtype}"
    batch, _, seq_len, head_dim = xq.shape
    if freqs_cis.dtype != torch.float32:
        return f"freqs_cis must be float32, got {freqs_cis.dtype}"
    if (
        freqs_cis.ndim != 6
        or freqs_cis.shape[0] not in (1, batch)
        or freqs_cis.shape[1:] != (1, seq_len, head_dim // 2, 2, 2)
    ):
        return (
            f"freqs_cis must be (1 or {batch}, 1, {seq_len}, {head_dim // 2}, 2, 2),"
            f" got {tuple(freqs_cis.shape)}"
        )
    if xk.device != xq.device or freqs_cis.device != xq.device:
        return f"all inputs must share one device, got {xq.device}, {xk.device}, {freqs_cis.device}"
    return None


def apply_rope_supported(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> bool:
    """Whether these arguments fit the fused rotation's shape contract."""
    return xq.device.type == "cuda" and _rope_args_error(xq, xk, freqs_cis) is None


@torch.library.custom_op("dinkster_kernels::apply_rope", mutates_args=(), device_types="cuda")
def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k pairs through the 2x2 frequency matrices.

    Outputs match the input shape and dtype, are always C-contiguous
    with canonical strides, and are bit-identical to the reference
    pure-torch rotation on the same device.
    """
    error = _rope_args_error(xq, xk, freqs_cis)
    if error is not None:
        raise ValueError(error)
    from . import _rope_triton as kernels

    xq = xq.contiguous()
    xk = xk.contiguous()
    freqs_cis = freqs_cis.contiguous()
    # Outputs are canonical contiguous by contract. Canonical strides
    # are a pure function of shape, so the fake below matches this
    # allocation for every accepted input, including zero-size and
    # singleton-dimension layouts where preserve-format semantics of
    # empty_like differ between real and fake tensors.
    out_q = torch.empty_like(xq, memory_format=torch.contiguous_format)
    out_k = torch.empty_like(xk, memory_format=torch.contiguous_format)
    if xq.numel():
        kernels.launch_rope(xq, xk, freqs_cis, out_q, out_k)
    return out_q, out_k


@apply_rope.register_fake
def _apply_rope_fake(  # pyright: ignore[reportUnusedFunction] - registered on the op
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # The eager body allocates canonical contiguous outputs, and
    # canonical strides depend only on shape, so this matches the
    # eager allocation for every accepted input.
    return (
        torch.empty_like(xq, memory_format=torch.contiguous_format),
        torch.empty_like(xk, memory_format=torch.contiguous_format),
    )


def apply_rope_available() -> bool:
    """Whether the fused rotation can execute on this host (shared probe)."""

    return _common.fused_ops_available()
