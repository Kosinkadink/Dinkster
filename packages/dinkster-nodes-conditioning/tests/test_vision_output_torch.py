"""Actual reset CLIP vision outputs round-trip through ordinary pack hosts."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    TypeRegistry,
    materialize_clip_vision_output,
    register_inference_types,
)
from dinkster_nodes_conditioning.clip import CLIPVisionEncode


def _source(variant: int) -> Any:
    attributes = {
        "last_hidden_state": torch.arange(6, dtype=torch.float16).reshape(2, 3).T,
        "image_embeds": torch.tensor([0x3F80, 0x8000, 0x7FC1], dtype=torch.uint16).view(
            torch.bfloat16
        ),
        "penultimate_hidden_states": torch.tensor([[1.0, -2.0]], dtype=torch.float64),
        "image_sizes": [(224, 336, 3), (448, 224, 3)],
        "mm_projected": None,
    }
    if variant:
        attributes["all_hidden_states"] = (
            torch.empty((2, 0, 3), dtype=torch.float32),
            {"layer": 4, "label": "last", "flag": True, "weight": 0.25, "empty": None},
        )
    return materialize_clip_vision_output(attributes)


class VisionEncoder:
    def __init__(self) -> None:
        self.image: Any = None
        self.crop: bool | None = None

    def encode_image(self, image: Any, *, crop: bool) -> Any:
        self.image = image
        self.crop = crop
        return _source(int(crop))


@pytest.mark.parametrize(("crop", "expected_crop"), (("none", False), ("center", True)))
def test_clip_vision_encode_materializes_image_and_selects_crop(
    crop: str, expected_crop: bool
) -> None:
    image = np.arange(24, dtype=np.float32).reshape(1, 2, 4, 3)
    encoder = VisionEncoder()

    output = CLIPVisionEncode.execute(clip_vision=encoder, image=image, crop=crop)[
        "CLIP_VISION_OUTPUT"
    ]

    assert isinstance(encoder.image, torch.Tensor)
    assert torch.equal(encoder.image, torch.from_numpy(image))
    assert encoder.crop is expected_crop
    assert (
        type(materialize_clip_vision_output(output))
        is importlib.import_module("dinkster_inference.clip_vision").Output
    )


def _inspect(value: Any) -> dict[str, Any]:
    output = materialize_clip_vision_output(value)
    assert type(output) is importlib.import_module("dinkster_inference.clip_vision").Output

    def plain(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return {
                "dtype": str(item.dtype),
                "shape": list(item.shape),
                "bytes": bytes(item.contiguous().reshape(-1).view(torch.uint8).tolist()).hex(),
            }
        if isinstance(item, dict):
            return {key: plain(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return {"tuple": [plain(child) for child in item]}
        if isinstance(item, list):
            return [plain(child) for child in item]
        return item

    return plain(vars(output))


@pytest.mark.parametrize("variant", (0, 1))
def test_actual_output_materialization_is_lossless_and_does_not_mutate(variant: int) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(CLIP_VISION_OUTPUT_TYPE)
    source = _source(variant)
    original_attributes = vars(source).copy()
    before = _inspect(source)
    decoded = spec.decode(spec.encode(source))
    assert _inspect(decoded) == before
    assert spec.fingerprint is not None
    assert spec.fingerprint(source) == spec.fingerprint(decoded)
    for name, item in original_attributes.items():
        assert getattr(source, name) is item
    assert _inspect(source) == before
    if variant:
        assert "all_hidden_states" in before
    else:
        assert "all_hidden_states" not in before
