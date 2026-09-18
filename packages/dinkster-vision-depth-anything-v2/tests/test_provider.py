from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest

pytest.importorskip("torch")

from dinkster_assets import install_declared_assets, use_declared_asset_pack
from dinkster_vision_depth_anything_v2 import model as depth_model
from dinkster_vision_depth_anything_v2.model import (
    _model_config,
    _model_input,
    _resize_hint,
    execute_depth_anything_v2,
)
from dinkster_workers import load_manifest
from transformers import Dinov2Config

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "packages" / "dinkster-vision-depth-anything-v2" / "dinkster-pack.toml"
GOLDEN_PATH = ROOT / "tests" / "goldens" / "depth_anything_v2_controlnet_aux_59b1fc4.json"
MODEL_SHA256 = "4e01e34ed5549b529b70b92d53226bc370f03041977b390d3dde45d47f516cf9"


class _ModelResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        del digest
        return self.path


def _model_path() -> Path:
    value = os.environ.get("DINKSTER_DEPTH_ANYTHING_V2_TEST_MODEL")
    if not value:
        pytest.skip("DINKSTER_DEPTH_ANYTHING_V2_TEST_MODEL is not configured")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"DINKSTER_DEPTH_ANYTHING_V2_TEST_MODEL does not exist: {path}")
    return path


def _decode(record: object) -> np.ndarray:
    payload = cast("dict[str, object]", record)
    shape = cast("list[int]", payload["shape"])
    encoded = cast("str", payload["uint8Base64"])
    return np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).reshape(shape)


def _golden() -> dict[str, object]:
    return cast("dict[str, object]", json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))


def test_large_model_config_matches_converted_artifact_architecture() -> None:
    config = _model_config()
    backbone = cast("Dinov2Config", config.backbone_config)
    assert config.depth_estimation_type == "relative"
    assert backbone.hidden_size == 1024
    assert backbone.num_hidden_layers == 24
    assert backbone.num_attention_heads == 16
    assert backbone.patch_size == 14
    assert backbone.out_indices == [5, 12, 18, 24]
    assert config.neck_hidden_sizes == [256, 512, 1024, 1024]


def test_source_preprocessing_and_hint_sizing_are_pinned() -> None:
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    tensor = _model_input(frame)
    assert tuple(tensor.shape) == (1, 3, 518, 686)
    np.testing.assert_allclose(
        tensor[0, :, 0, 0].numpy(),
        np.array((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225)),
        rtol=0.0,
        atol=1e-6,
    )
    depth = np.arange(96 * 128, dtype=np.uint8).reshape(96, 128)
    hint = _resize_hint(depth, 64)
    assert hint.shape == (64, 85, 3)
    assert np.array_equal(hint[:, :, 0], hint[:, :, 1])
    assert np.array_equal(hint[:, :, 1], hint[:, :, 2])


def test_invalid_inputs_fail_before_loading_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_load() -> object:
        raise AssertionError("model must not load before input validation")

    monkeypatch.setattr(depth_model, "_load_model", unexpected_load)
    image = np.zeros((1, 8, 12, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="resolution must be an integer"):
        execute_depth_anything_v2(image, resolution=32)
    with pytest.raises(ValueError, match="1, 3, or 4 channels"):
        execute_depth_anything_v2(np.zeros((1, 8, 12, 2), dtype=np.float32), resolution=64)


def test_multi_image_batches_preserve_frame_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(depth_model, "_load_model", object)
    monkeypatch.setattr(
        depth_model,
        "_relative_depth",
        lambda _model, frame: frame[:, :, 0],
    )
    image = np.stack(
        (
            np.zeros((8, 12, 3), dtype=np.float32),
            np.ones((8, 12, 3), dtype=np.float32),
        )
    )
    output = execute_depth_anything_v2(image, resolution=64)
    assert output.shape == (2, 64, 96, 3)
    assert np.count_nonzero(output[0]) == 0
    assert np.count_nonzero(output[1] != 1.0) == 0


def test_depth_output_matches_pinned_controlnet_aux_vector() -> None:
    golden = _golden()
    assert golden["baseline"] == "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
    assert golden["modelSha256"] == (
        "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345"
    )
    assert golden["numpy"] == "2.5.1"
    assert golden["opencv"] == "5.0.0"
    assert golden["torch"] == "2.13.0+cpu"
    source = _decode(golden["source"])[None].astype(np.float32) / 255.0
    expected = _decode(golden["output"])
    manifest = load_manifest(MANIFEST)
    install_declared_assets(manifest.name, manifest.assets, _ModelResolver(_model_path()))
    with use_declared_asset_pack(manifest.name):
        output = execute_depth_anything_v2(source, resolution=64)
    actual = np.rint(output[0] * 255.0).astype(np.uint8)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def test_converted_model_artifact_is_the_expected_bytes() -> None:
    with _model_path().open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == MODEL_SHA256
