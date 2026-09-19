"""Per-tensor FP8 input quantization as a Dinkster-owned custom op.

The CUDA route converts float32, float16, or bfloat16 input through
float32 division by one float32 scale, clamps to the finite range of
the requested FP8 dtype, and casts once to FP8. This is the arithmetic
used by dinkster-kitchen's CUDA quantizer and by the eager reference in
``dinkster_inference_torch.quant_linear``.
"""

from __future__ import annotations

import importlib

import torch

_INPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_OUTPUT_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_MIN_PERFORMANCE_ELEMENTS = 256
_available: bool | None = None


def _args_error(
    input: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype,
) -> str | None:
    if input.dtype not in _INPUT_DTYPES:
        return f"input must be float32, float16, or bfloat16, got {input.dtype}"
    if scale.dtype != torch.float32 or scale.numel() != 1:
        return (
            f"scale must contain one float32 value, got {scale.dtype} of shape {tuple(scale.shape)}"
        )
    if scale.device != input.device:
        return f"input and scale must share one device, got {input.device} and {scale.device}"
    if output_type not in _OUTPUT_DTYPES:
        return f"output_type must be float8_e4m3fn or float8_e5m2, got {output_type}"
    return None


def quantize_per_tensor_fp8_supported(
    input: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> bool:
    """Whether these arguments fit the owned CUDA quantizer and its
    measured performance domain."""
    return (
        input.device.type == "cuda"
        and input.numel() >= _MIN_PERFORMANCE_ELEMENTS
        and _args_error(input, scale, output_type) is None
    )


@torch.library.custom_op(
    "dinkster_kernels::quantize_per_tensor_fp8",
    mutates_args=(),
    device_types="cuda",
)
def quantize_per_tensor_fp8(
    input: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Quantize one tensor into a contiguous FP8 output."""
    error = _args_error(input, scale, output_type)
    if error is not None:
        raise ValueError(error)
    from . import _fp8_quantize_triton as kernels

    input = input.contiguous()
    scale = scale.contiguous().reshape(())
    output = torch.empty_like(
        input,
        dtype=output_type,
        memory_format=torch.contiguous_format,
    )
    if input.numel():
        kernels.launch_quantize_per_tensor_fp8(input, scale, output)
    return output


@quantize_per_tensor_fp8.register_fake
def _quantize_per_tensor_fp8_fake(  # pyright: ignore[reportUnusedFunction] - op registration
    input: torch.Tensor,
    scale: torch.Tensor,
    output_type: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    return torch.empty_like(
        input,
        dtype=output_type,
        memory_format=torch.contiguous_format,
    )


def quantize_per_tensor_fp8_available() -> bool:
    """Whether the owned quantizer compiles and runs bit-exactly on this host."""
    global _available
    if getattr(torch.version, "hip", None) is not None:
        return False
    if _available is None:
        _available = _probe()
    return _available


def _probe() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        importlib.import_module("triton")
        input = torch.tensor(
            [-1000.0, -10.083683967590332, -1.0, -0.0, 0.0, 1.0, 2.0410664081573486, 1000.0],
            device="cuda",
            dtype=torch.float32,
        )
        scale = torch.tensor(0.03, device="cuda", dtype=torch.float32)
        for output_type in _OUTPUT_DTYPES:
            fp8_max = torch.finfo(output_type).max
            expected = torch.clamp(input / scale, -fp8_max, fp8_max).to(output_type)
            actual = quantize_per_tensor_fp8(input, scale, output_type)
            if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
                return False
        return True
    except Exception:  # noqa: BLE001 - capability probes are best-effort
        return False
