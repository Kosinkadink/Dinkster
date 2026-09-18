"""Storage-dtype writeback: stochastic rounding + per-key seeds.

Ports of comfy/float.py stochastic_rounding / manual_stochastic_round_
to_float8 and comfy/utils.py string_to_seed @ b78cec87. Stochastic
rounding is LOAD-BEARING for patch application (ROADMAP): writing a
patched fp8 weight back with plain nearest rounding loses the LoRA's
effect; the stochastic path preserves it in expectation.

Like the reference, fp8 rounding prefers the comfy-kitchen accelerated
kernel when the installed comfy_kitchen exposes it (capability probe,
never a version assumption) and falls back to the manual torch path
otherwise. The two paths draw randomness differently and are not
bit-equal - not even upstream, where both are accepted. The
accelerated path's randomness (a seeded uint8 ``torch.randint``
stream) is byte-identical across the torch builds Dinkster spans, so it
is also the more reproducible of the two; the manual path's float16
``torch.rand`` stream is torch-build-specific. The torch package
README covers installing kitchen into ``.venv-torch``.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock

import torch

_Fp8Kernel = Callable[[torch.Tensor, torch.Tensor, torch.dtype], torch.Tensor]
_UNPROBED = object()


def _probe_kitchen_fp8_kernel() -> _Fp8Kernel | None:
    """Capability probe for comfy_kitchen.stochastic_rounding_fp8
    (comfy/float.py's optional-import block @ b78cec87): present and
    exposing the kernel -> accelerated path; anything else -> manual.
    Never a version check."""
    try:
        import comfy_kitchen  # pyright: ignore[reportMissingTypeStubs]

        kernel: _Fp8Kernel = comfy_kitchen.stochastic_rounding_fp8
        return kernel
    except (AttributeError, ImportError):
        return None


_ck_stochastic_rounding_fp8: _Fp8Kernel | None | object = _UNPROBED
_ck_stochastic_rounding_fp8_lock = Lock()


def _kitchen_fp8_kernel() -> _Fp8Kernel | None:
    global _ck_stochastic_rounding_fp8
    if _ck_stochastic_rounding_fp8 is _UNPROBED:
        with _ck_stochastic_rounding_fp8_lock:
            if _ck_stochastic_rounding_fp8 is _UNPROBED:
                _ck_stochastic_rounding_fp8 = _probe_kitchen_fp8_kernel()
    if _ck_stochastic_rounding_fp8 is None:
        return None
    return _ck_stochastic_rounding_fp8  # pyright: ignore[reportReturnType]


def string_to_seed(data: str | bytes) -> int:
    """CRC32 of ``data`` (comfy/utils.py string_to_seed @ b78cec87):
    the deterministic per-parameter-key seed for stochastic rounding,
    so patching the same key twice rounds identically."""
    crc = 0xFFFFFFFF
    for byte in data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF


def _calc_mantissa(
    abs_x: torch.Tensor,
    exponent: torch.Tensor,
    normal_mask: torch.Tensor,
    mantissa_bits: int,
    exponent_bias: int,
    generator: torch.Generator,
) -> torch.Tensor:
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - exponent_bias)) - 1.0) * (2**mantissa_bits),
        (abs_x / (2.0 ** (-exponent_bias + 1 - mantissa_bits))),
    )
    mantissa_scaled += torch.rand(
        mantissa_scaled.size(),
        dtype=mantissa_scaled.dtype,
        layout=mantissa_scaled.layout,
        device=mantissa_scaled.device,
        generator=generator,
    )
    return mantissa_scaled.floor() / (2**mantissa_bits)


def _manual_stochastic_round_to_float8(
    x: torch.Tensor, dtype: torch.dtype, generator: torch.Generator
) -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    # DELIBERATE DIVERGENCE from the reference (comfy/float.py
    # @ b78cec87), which computes floor(log2(abs_x)) with fp16 log2.
    # For inputs just below a power of two, fp16 log2 rounds UP to
    # the integer (log2(7.996) -> 3.0), the exponent comes out one
    # too high, and the low rng draws produce a result one ulp BELOW
    # the correct lower neighbor - off the adjacent-value grid
    # entirely (docs/comfyui-issues/
    # stochastic-rounding-fp16-log2-boundary.md; the comfy-kitchen
    # CUDA kernel gets these right, so upstream's two paths disagree
    # bitwise). frexp is exact: abs_x = m * 2**e with m in [0.5, 1),
    # so floor(log2(abs_x)) == e - 1 for every finite nonzero input.
    # All other inputs round bit-identically to the reference (the
    # executed-reference goldens in test_patches.py stay green).
    # (cast back to fp16: the downstream mantissa arithmetic and the
    # torch.rand draw dtype must stay exactly the reference's)
    exponent = torch.clamp(
        torch.frexp(abs_x).exponent - 1 + exponent_bias,
        0,
        2**exponent_bits - 1,
    ).to(abs_x.dtype)
    normal_mask = ~(exponent == 0)

    abs_x[:] = _calc_mantissa(abs_x, exponent, normal_mask, mantissa_bits, exponent_bias, generator)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - exponent_bias)) * (1.0 + abs_x),
        (2.0 ** (-exponent_bias + 1)) * abs_x,
    )

    inf = torch.finfo(dtype)
    torch.clamp(sign, min=inf.min, max=inf.max, out=sign)
    return sign


def stochastic_rounding(value: torch.Tensor, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """Round ``value`` to storage ``dtype``; fp8 targets round
    stochastically with a ``seed``-deterministic generator, everything
    else is a plain cast (comfy/float.py stochastic_rounding
    @ b78cec87). Accelerated kitchen kernel when available, manual
    torch path otherwise - see module docstring. The manual slice loop
    is kept verbatim because it determines the RNG draw order."""
    if dtype == torch.float32:
        return value.to(dtype=torch.float32)
    if dtype == torch.float16:
        return value.to(dtype=torch.float16)
    if dtype == torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        kitchen_fp8_kernel = _kitchen_fp8_kernel()
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        if kitchen_fp8_kernel is not None:
            rng = torch.randint(
                0,
                256,
                value.size(),
                dtype=torch.uint8,
                layout=value.layout,
                device=value.device,
                generator=generator,
            )
            if value.device.type == "cuda":
                # kitchen's compiled CUDA kernel DLPack-exports its
                # arguments at the CURRENT device (stream=-1), so a
                # cuda:1 tensor while cuda:0 is current raises
                # BufferError. Pin the context to the value's own
                # device (upstream issue: docs/comfyui-issues/
                # comfy-kitchen-cuda-stochastic-rounding-wrong-device.md).
                with torch.cuda.device(value.device):
                    return kitchen_fp8_kernel(value, rng, dtype)
            return kitchen_fp8_kernel(value, rng, dtype)
        output = torch.empty_like(value, dtype=dtype)
        num_slices = max(1, (value.numel() / (4096 * 4096)))
        slice_size = max(1, round(value.shape[0] / num_slices))
        for i in range(0, value.shape[0], slice_size):
            output[i : i + slice_size].copy_(
                _manual_stochastic_round_to_float8(value[i : i + slice_size], dtype, generator)
            )
        return output

    return value.to(dtype=dtype)


__all__ = [
    "stochastic_rounding",
    "string_to_seed",
]
