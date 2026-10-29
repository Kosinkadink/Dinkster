"""Explicit contiguous device placement for Qwen transformer layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from dinkster_inference.patches import PatchSet

from .module_residency import (
    ComponentResidencyPlacement,
    ResidencyMechanismFactory,
    enroll_component_placement,
)
from .operations import bound_compute_device
from .qwen_text import QwenBlock, QwenTextModel
from .residency import ResidentWeights


def _placement_device(value: object) -> torch.device:
    try:
        device = torch.device(cast("str | torch.device", value))
    except (TypeError, RuntimeError) as error:
        raise TypeError("Qwen layer placement device must be a torch device or string") from error
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Qwen layer placement supports only CPU, CUDA, and MPS devices")
    if device.type == "cpu" and str(device) != "cpu":
        raise ValueError("Qwen CPU layer placement does not accept a device index")
    if device.type == "cuda" and ":" not in str(device):
        raise ValueError("Qwen CUDA layer placement requires an explicit device index")
    return device


@dataclass(frozen=True, slots=True)
class QwenLayerRange:
    """A half-open contiguous layer range assigned to one device."""

    start: int
    stop: int
    device: torch.device

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.stop) is not int:
            raise TypeError("Qwen layer range bounds must be exact integers")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("Qwen layer range must be non-empty with non-negative bounds")
        object.__setattr__(self, "device", _placement_device(self.device))


@dataclass(frozen=True, slots=True)
class QwenLayerPlacement:
    """Complete ordered layer placement with no implicit device policy."""

    ranges: tuple[QwenLayerRange, ...]

    def __post_init__(self) -> None:
        if type(self.ranges) is not tuple or not self.ranges:
            raise ValueError("Qwen layer placement requires a non-empty tuple of ranges")
        expected = 0
        seen: set[torch.device] = set()
        for item in self.ranges:
            if type(item) is not QwenLayerRange:
                raise TypeError("Qwen layer placement entries must be QwenLayerRange")
            if item.start != expected:
                raise ValueError("Qwen layer placement ranges must be ordered and contiguous")
            if item.device in seen:
                raise ValueError("each Qwen placement device must own one contiguous range")
            expected = item.stop
            seen.add(item.device)
        if len(self.ranges) > 1 and any(item.device.type != "cuda" for item in self.ranges):
            raise ValueError("multi-range Qwen layer placement requires indexed CUDA devices")

    @property
    def layer_count(self) -> int:
        return self.ranges[-1].stop

    @property
    def devices(self) -> tuple[torch.device, ...]:
        return tuple(item.device for item in self.ranges)

    def device_for_layer(self, layer: int) -> torch.device:
        if type(layer) is not int or not 0 <= layer < self.layer_count:
            raise ValueError("Qwen layer index is outside the placement")
        return next(item.device for item in self.ranges if item.start <= layer < item.stop)

    def module_placements(self, model: QwenTextModel) -> dict[str, torch.device]:
        if len(model.layers) != self.layer_count:
            raise ValueError(
                f"Qwen placement covers {self.layer_count} layers but model has {len(model.layers)}"
            )
        result = {
            "embed_tokens": self.ranges[0].device,
            "norm": self.ranges[-1].device,
        }
        result.update(
            {f"layers.{index}": self.device_for_layer(index) for index in range(self.layer_count)}
        )
        return result


def enroll_qwen_layer_placement(
    model: QwenTextModel,
    placement: QwenLayerPlacement,
    *,
    offload_device: torch.device | str,
    patch_set: PatchSet[torch.Tensor] | None = None,
    intermediate_dtype: torch.dtype = torch.float32,
    patch_weight_dtype: torch.dtype | None = None,
    patch_key_prefix: str = "",
    mechanism_factory: ResidencyMechanismFactory = ResidentWeights,
) -> ComponentResidencyPlacement:
    """Bind Qwen state to per-device mechanisms using an explicit placement."""
    if type(model) is not QwenTextModel:
        raise TypeError("Qwen layer placement requires a QwenTextModel")
    if type(placement) is not QwenLayerPlacement:
        raise TypeError("placement must be QwenLayerPlacement")
    enrolled = enroll_component_placement(
        model,
        placement.module_placements(model),
        offload_device=offload_device,
        patch_set=patch_set,
        intermediate_dtype=intermediate_dtype,
        patch_weight_dtype=patch_weight_dtype,
        patch_key_prefix=patch_key_prefix,
        mechanism_factory=mechanism_factory,
    )
    model.__dict__["_dinkster_qwen_layer_placement"] = placement
    return enrolled


def qwen_layer_placement(model: QwenTextModel) -> QwenLayerPlacement | None:
    value = model.__dict__.get("_dinkster_qwen_layer_placement")
    if value is None:
        return None
    if type(value) is not QwenLayerPlacement:
        raise TypeError("Qwen model has invalid layer placement metadata")
    return value


def _module_device(module: torch.nn.Module) -> torch.device:
    bound = bound_compute_device(module)
    if bound is not None:
        return bound
    state = next(module.parameters(recurse=False), None)
    if state is None:
        state = next(module.buffers(recurse=False), None)
    if state is None:
        raise ValueError("Qwen placement module has no direct state")
    return state.device


def resolve_qwen_layer_placement(model: QwenTextModel) -> QwenLayerPlacement:
    """Return and verify explicit placement, or the homogeneous default."""
    placement = qwen_layer_placement(model)
    embedding_device = _module_device(model.embed_tokens)
    if type(model) is not QwenTextModel:
        if placement is not None:
            raise TypeError("explicit Qwen layer placement requires a QwenTextModel")
        return QwenLayerPlacement((QwenLayerRange(0, len(model.layers), embedding_device),))
    layer_devices = tuple(
        _module_device(cast(QwenBlock, layer).input_layernorm) for layer in model.layers
    )
    norm_device = _module_device(model.norm)
    if placement is None:
        if any(device != embedding_device for device in (*layer_devices, norm_device)):
            raise ValueError("Qwen models spanning devices require explicit layer placement")
        placement = QwenLayerPlacement((QwenLayerRange(0, len(model.layers), embedding_device),))
    if placement.layer_count != len(model.layers):
        raise ValueError("Qwen layer placement does not cover the model")
    expected = tuple(placement.device_for_layer(index) for index in range(len(model.layers)))
    if embedding_device != placement.ranges[0].device:
        raise ValueError("Qwen embedding placement does not match its residency binding")
    if layer_devices != expected:
        raise ValueError("Qwen layer placement does not match its residency bindings")
    if norm_device != placement.ranges[-1].device:
        raise ValueError("Qwen output normalization placement does not match its residency binding")
    return placement


__all__ = [
    "QwenLayerPlacement",
    "QwenLayerRange",
    "enroll_qwen_layer_placement",
    "qwen_layer_placement",
    "resolve_qwen_layer_placement",
]
