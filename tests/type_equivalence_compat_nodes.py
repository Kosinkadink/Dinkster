"""Translated v1 compatibility pack for type-equivalence crossings."""

from __future__ import annotations

import sys
import types

import numpy as np
from dinkster_compat_comfy import translate_mappings
from dinkster_values import TypeRegistry


class FakeTensor:
    """Boundary-relevant torch tensor surface for the torch-free test worker."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array


fake_torch = types.ModuleType("torch")
fake_torch.from_numpy = FakeTensor  # pyright: ignore[reportAttributeAccessIssue]
sys.modules["torch"] = fake_torch


class CompatImageProducer:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "produce"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:  # noqa: N802
        return {"required": {}}

    def produce(self) -> tuple[FakeTensor]:
        return (FakeTensor(np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3)),)


class CompatMaskProducer:
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "produce"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:  # noqa: N802
        return {"required": {}}

    def produce(self) -> tuple[FakeTensor]:
        return (FakeTensor(np.arange(6, dtype=np.float32).reshape(1, 2, 3)),)


def _shape(value: FakeTensor) -> str:
    if not isinstance(value, FakeTensor):
        raise TypeError(f"compat input was not restored through the torch codec: {type(value)}")
    return "x".join(str(dimension) for dimension in value.shape)


class CompatImageConsumer:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("shape",)
    FUNCTION = "consume"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:  # noqa: N802
        return {"required": {"image": ("IMAGE",)}}

    def consume(self, image: FakeTensor) -> tuple[str]:
        return (_shape(image),)


class CompatMaskConsumer:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("shape",)
    FUNCTION = "consume"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:  # noqa: N802
        return {"required": {"mask": ("MASK",)}}

    def consume(self, mask: FakeTensor) -> tuple[str]:
        return (_shape(mask),)


NODE_CLASS_MAPPINGS = {
    "ImageProducer": CompatImageProducer,
    "MaskProducer": CompatMaskProducer,
    "ImageConsumer": CompatImageConsumer,
    "MaskConsumer": CompatMaskConsumer,
}
_TRANSLATION = translate_mappings(NODE_CLASS_MAPPINGS, namespace="custom")
NODES = tuple(_TRANSLATION.node_classes)


def register_types(registry: TypeRegistry) -> None:
    _TRANSLATION.register_types(registry)
