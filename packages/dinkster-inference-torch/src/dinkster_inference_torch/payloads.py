"""Strict little-endian torch tensor adapters for conditioning payloads."""

from __future__ import annotations

import ctypes
import sys

import torch
from dinkster_inference import (
    ConditioningPayloadBinding,
    LivePayloadBinding,
    PayloadBinding,
    ResidentPayloadBinding,
)

_TORCH_TO_WIRE: dict[torch.dtype, str] = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
_WIRE_TO_TORCH = {wire: dtype for dtype, wire in _TORCH_TO_WIRE.items()}
_WIDTHS = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


class TensorPayloadError(ValueError):
    """A strict tensor adapter refusal."""


def _little_endian(data: bytes, width: int) -> bytes:
    if sys.byteorder == "little" or width == 1:
        return data
    return b"".join(data[offset : offset + width][::-1] for offset in range(0, len(data), width))


def _native_endian(data: bytes, width: int) -> bytearray:
    little = _little_endian(data, width)
    return bytearray(little)


def tensor_to_payload_binding(
    reference_id: str,
    tensor: torch.Tensor,
    *,
    space: str,
) -> PayloadBinding:
    """Detach and normalize one strided tensor into owned canonical bytes."""
    if not isinstance(tensor, torch.Tensor):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TensorPayloadError("value must be a torch Tensor")
    if tensor.ndim == 0:
        raise TensorPayloadError("scalar tensors are not supported")
    if tensor.layout is not torch.strided:
        raise TensorPayloadError("sparse and non-strided tensors are not supported")
    if tensor.is_quantized:
        raise TensorPayloadError("quantized tensors are not supported")
    if tensor.is_complex():
        raise TensorPayloadError("complex tensors are not supported")
    dtype = _TORCH_TO_WIRE.get(tensor.dtype)
    if dtype is None:
        raise TensorPayloadError(f"unsupported tensor dtype: {tensor.dtype}")
    normalized = tensor.detach().to(device="cpu").contiguous()
    if dtype == "BOOL":
        normalized = normalized.to(dtype=torch.uint8)
    raw = ctypes.string_at(normalized.data_ptr(), normalized.numel() * normalized.element_size())
    data = _little_endian(raw, _WIDTHS[dtype])
    return PayloadBinding(reference_id, tuple(normalized.shape), dtype, space, data)


def payload_binding_to_tensor(binding: ConditioningPayloadBinding) -> torch.Tensor:
    """Decode one binding into a tensor backed by newly owned writable storage."""
    if isinstance(binding, ResidentPayloadBinding):
        raise TensorPayloadError(f"resident payload {binding.reference_id!r} cannot materialize")
    if isinstance(binding, LivePayloadBinding):
        try:
            data = binding.materialize(binding.payload)
        except Exception as error:
            raise TensorPayloadError(
                f"live payload {binding.reference_id!r} failed to materialize: {error}"
            ) from error
        binding = PayloadBinding(
            binding.reference_id, binding.shape, binding.dtype, binding.space, data
        )
    dtype = _WIRE_TO_TORCH.get(binding.dtype)
    if dtype is None:
        raise TensorPayloadError(f"unsupported payload dtype: {binding.dtype}")
    if not binding.data:
        return torch.empty(binding.shape, dtype=dtype)
    width = _WIDTHS[binding.dtype]
    owned = _native_endian(binding.data, width)
    storage_dtype = (
        torch.uint16
        if binding.dtype == "BF16"
        else torch.uint8
        if binding.dtype == "BOOL"
        else dtype
    )
    tensor = torch.frombuffer(owned, dtype=storage_dtype).clone()
    if binding.dtype == "BF16":
        tensor = tensor.view(torch.bfloat16)
    elif binding.dtype == "BOOL":
        if bool(torch.any((tensor != 0) & (tensor != 1))):
            raise TensorPayloadError("BOOL payload bytes must be canonical 0 or 1")
        tensor = tensor.to(dtype=torch.bool)
    return tensor.reshape(binding.shape)


__all__ = [
    "TensorPayloadError",
    "payload_binding_to_tensor",
    "tensor_to_payload_binding",
]
