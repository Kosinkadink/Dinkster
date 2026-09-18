"""ConvRot INT8 weight dequantization as a Dinkster-owned custom op.

The fused CUDA route dequantizes each row and applies the normalized
256-wide regular Hadamard inverse rotation in one launch. Four radix-4
stages preserve the reference operation order and materialize only the
final float32 weight.
"""

from __future__ import annotations

import torch

from . import _common

CONVROT_GROUP_SIZE = 256


def _args_error(qdata: torch.Tensor, scale: torch.Tensor, group_size: int) -> str | None:
    if qdata.dtype != torch.int8 or qdata.ndim != 2:
        return f"qdata must be rank-2 int8, got {qdata.dtype} of shape {tuple(qdata.shape)}"
    if group_size != CONVROT_GROUP_SIZE:
        return f"owned ConvRot dequantization requires group size {CONVROT_GROUP_SIZE}"
    rows, columns = qdata.shape
    if columns == 0 or columns % CONVROT_GROUP_SIZE:
        return f"qdata width must be a positive multiple of {CONVROT_GROUP_SIZE}, got {columns}"
    if scale.dtype != torch.float32 or scale.shape != (rows, 1):
        return f"scale must be float32 of shape ({rows}, 1), got {scale.dtype} {tuple(scale.shape)}"
    if scale.device != qdata.device:
        return f"qdata and scale must share one device, got {qdata.device} and {scale.device}"
    return None


def dequantize_int8_convrot_weight_supported(
    qdata: torch.Tensor, scale: torch.Tensor, group_size: int
) -> bool:
    """Whether these arguments fit the owned fused route."""
    return qdata.device.type == "cuda" and _args_error(qdata, scale, group_size) is None


@torch.library.custom_op(
    "dinkster_kernels::dequantize_int8_convrot_weight",
    mutates_args=(),
    device_types="cuda",
)
def dequantize_int8_convrot_weight(
    qdata: torch.Tensor, scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Dequantize rowwise INT8 ConvRot weights into contiguous float32."""
    error = _args_error(qdata, scale, group_size)
    if error is not None:
        raise ValueError(error)
    from . import _int8_convrot_triton as kernels

    qdata = qdata.contiguous()
    scale = scale.contiguous().reshape(-1)
    output = torch.empty_like(
        qdata,
        dtype=torch.float32,
        memory_format=torch.contiguous_format,
    )
    if qdata.numel():
        kernels.launch_dequantize_int8_convrot(qdata, scale, output)
    return output


@dequantize_int8_convrot_weight.register_fake
def _dequantize_int8_convrot_weight_fake(  # pyright: ignore[reportUnusedFunction] - op registration
    qdata: torch.Tensor, scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    return torch.empty_like(
        qdata,
        dtype=torch.float32,
        memory_format=torch.contiguous_format,
    )


def dequantize_int8_convrot_weight_available() -> bool:
    """Whether Dinkster-owned fused kernels can execute on this host."""
    return getattr(torch.version, "hip", None) is None and _common.fused_ops_available()
