"""Shared tensor helpers for the torch patch layer.

Near-transcriptions of comfy/weight_adapter/base.py and
comfy/model_management.py cast_to_device @ b78cec87. Ported tensor
code stays close to the source on purpose (the torch-free boundary
rules in docs/native-inference-plan.md): these functions ARE the
reference math, minus device-management machinery Dinkster does not
carry (non_blocking probing, streams).
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol

import torch


class TransferStager(Protocol):
    def transfer(
        self,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype | None,
    ) -> torch.Tensor: ...


_TRANSFER_STAGER: ContextVar[TransferStager | None] = ContextVar(
    "dinkster_transfer_stager", default=None
)


@contextmanager
def use_transfer_stager(stager: TransferStager) -> Generator[None]:
    token = _TRANSFER_STAGER.set(stager)
    try:
        yield
    finally:
        _TRANSFER_STAGER.reset(token)


def identity(tensor: torch.Tensor) -> torch.Tensor:
    """The default per-entry delta hook (the reference's
    ``lambda a: a`` when a patch tuple's function slot is None)."""
    return tensor


def transfer_to_device(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
    copy: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Move a tensor, using an active bounded CUDA stager when available."""
    if (
        non_blocking
        and device.type == "cuda"
        and tensor.device.type == "cpu"
        and not tensor.is_pinned()
    ):
        stager = _TRANSFER_STAGER.get()
        if stager is not None:
            return stager.transfer(tensor, dtype=dtype)
    return tensor.to(device=device, dtype=dtype, copy=copy, non_blocking=non_blocking)


def cast_to_device(
    tensor: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    copy: bool = False,
) -> torch.Tensor:
    """comfy cast_to_device @ b78cec87 with pinned CUDA staging."""
    if tensor.device == device and tensor.dtype == dtype and not copy:
        return tensor
    return transfer_to_device(
        tensor,
        device,
        dtype=dtype,
        copy=copy,
        non_blocking=device.type == "cuda" and tensor.device.type == "cpu",
    )


def pad_tensor_to_shape(tensor: torch.Tensor, new_shape: tuple[int, ...]) -> torch.Tensor:
    """Zero-pad ``tensor`` up to ``new_shape``
    (comfy/weight_adapter/base.py pad_tensor_to_shape @ b78cec87)."""
    if any(new_shape[i] < tensor.shape[i] for i in range(len(new_shape))):
        raise ValueError(
            "the new shape must be larger than the original tensor in"
            f" all dimensions, got {tuple(tensor.shape)} -> {new_shape}"
        )
    if len(new_shape) != len(tensor.shape):
        raise ValueError(
            "the new shape must have the same number of dimensions as"
            f" the original tensor, got {tuple(tensor.shape)} -> {new_shape}"
        )
    padded = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    slices = tuple(slice(0, dim) for dim in tensor.shape)
    padded[slices] = tensor[slices]
    return padded


def weight_decompose(
    dora_scale: torch.Tensor,
    weight: torch.Tensor,
    lora_diff: torch.Tensor,
    alpha: float,
    strength: float,
    intermediate_dtype: torch.dtype,
    function: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """DoRA application (comfy/weight_adapter/base.py weight_decompose
    @ b78cec87), including the per-entry delta ``function`` hook (None
    means identity). Mutates and returns ``weight`` like the
    reference."""
    if function is None:
        function = identity
    dora_scale = cast_to_device(dora_scale, weight.device, intermediate_dtype)
    lora_diff *= alpha
    weight_calc = weight + function(lora_diff).type(weight.dtype)

    wd_on_output_axis = dora_scale.shape[0] == weight_calc.shape[0]
    if wd_on_output_axis:
        weight_norm = (
            weight.reshape(weight.shape[0], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight.shape[0], *[1] * (weight.dim() - 1))
        )
    else:
        weight_norm = (
            weight_calc.transpose(0, 1)
            .reshape(weight_calc.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1))
            .transpose(0, 1)
        )
    weight_norm = weight_norm + torch.finfo(weight.dtype).eps

    weight_calc *= (dora_scale / weight_norm).type(weight.dtype)
    if strength != 1.0:
        weight_calc -= weight
        weight += strength * weight_calc
    else:
        weight[:] = weight_calc
    return weight


__all__ = [
    "cast_to_device",
    "identity",
    "pad_tensor_to_shape",
    "transfer_to_device",
    "weight_decompose",
]
