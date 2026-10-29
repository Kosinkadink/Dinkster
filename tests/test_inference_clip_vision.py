"""Torch-free Wan 2.1 CLIP vision checkpoint admission tests."""

from __future__ import annotations

import pytest
from dinkster_inference.clip_vision import (
    WAN21_CLIP_VISION,
    ClipVisionConfig,
    ClipVisionDetectError,
    clip_vision_layout,
    detect_wan21_clip_vision,
)
from dinkster_inference.devices import FLOAT16, FLOAT32, INT64
from dinkster_inference.weights import TensorGeometry


def _header() -> dict[str, TensorGeometry]:
    return {
        key: TensorGeometry(shape, INT64 if key.endswith("position_ids") else FLOAT16)
        for key, shape in clip_vision_layout().items()
    }


def test_exact_official_header_is_admitted() -> None:
    header = _header()
    assert len(header) == 521
    assert detect_wan21_clip_vision(header) == WAN21_CLIP_VISION
    assert WAN21_CLIP_VISION.tokens == 257
    assert header["vision_model.encoder.layers.31.mlp.fc1.weight"].shape == (5120, 1280)


@pytest.mark.parametrize("eps", (0.0, float("nan"), float("inf"), 1))
def test_config_refuses_invalid_layer_norm_epsilon(eps: float) -> None:
    with pytest.raises(ValueError, match="finite positive float"):
        ClipVisionConfig(layer_norm_eps=eps)


@pytest.mark.parametrize(
    ("key", "replacement"),
    (
        ("vision_model.encoder.layers.31.layer_norm1.weight", None),
        ("vision_model.embeddings.position_embedding.weight", (577, 1280)),
        ("vision_model.embeddings.patch_embedding.weight", (1280, 3, 16, 16)),
        ("visual_projection.weight", (1280, 1280)),
    ),
)
def test_non_vith_variants_and_incomplete_headers_fail_closed(
    key: str, replacement: tuple[int, ...] | None
) -> None:
    header = _header()
    if replacement is None:
        del header[key]
    else:
        header[key] = TensorGeometry(replacement, FLOAT16)
    with pytest.raises(ClipVisionDetectError, match="not the Wan 2.1"):
        detect_wan21_clip_vision(header)


def test_dtype_and_extra_key_fail_closed() -> None:
    header = _header()
    header["vision_model.encoder.layers.0.mlp.fc1.weight"] = TensorGeometry((5120, 1280), FLOAT32)
    header["other.weight"] = TensorGeometry((1,), FLOAT16)
    with pytest.raises(ClipVisionDetectError, match="expected dtype float16"):
        detect_wan21_clip_vision(header)
