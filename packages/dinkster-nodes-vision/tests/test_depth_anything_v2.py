from __future__ import annotations

from typing import cast

import numpy as np
import pytest

pytest.importorskip("torch")

from dinkster_nodes_vision.depth_anything_v2 import model as depth_model
from dinkster_nodes_vision.depth_anything_v2.model import (
    _model_config,
    _model_input,
    _resize_hint,
    execute_depth_anything_v2,
)
from transformers import Dinov2Config


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
