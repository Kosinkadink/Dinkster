"""CPU behavior coverage for ModelPatchLoader helpers and DurationHead loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DINKSTER_ACCELERATOR"] = "cpu"

import torch
from dinkster_api.v1 import LocalAssetRef
from dinkster_nodes_comfy.model_patch_loader import ModelPatchLoader, _z_image_convert


class LocalResolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve_local(self, binding: str) -> tuple[Path, tuple[int, int, int, int, int]] | None:
        stat = self.path.stat()
        return self.path, (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_dev,
            stat.st_ino,
        )

    def reconcile_local(self, binding: str) -> None:
        return None


def test_z_image_conversion_preserves_qkv_order_for_asymmetric_weights() -> None:
    state = {
        "x.attention.to_k.weight": torch.tensor([[10.0], [20.0]]),
        "x.attention.to_q.weight": torch.tensor([[1.0], [2.0]]),
        "x.attention.to_v.weight": torch.tensor([[100.0], [200.0]]),
        "x.attention.to_out.0.weight": torch.tensor([[7.0]]),
    }
    converted = _z_image_convert(state, torch)
    assert torch.equal(
        converted["x.attention.qkv.weight"],
        torch.tensor([[1.0], [2.0], [10.0], [20.0], [100.0], [200.0]]),
    )
    assert torch.equal(converted["x.attention.out.weight"], torch.tensor([[7.0]]))


def test_loader_rejects_unbound_asset() -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        ModelPatchLoader.execute(model=object())


def test_loader_builds_core_model_patcher_from_duration_head_checkpoint(tmp_path: Path) -> None:
    from dinkster_inference.ldm.lightricks.duration_head import DurationHead

    checkpoint = tmp_path / "duration.safetensors"
    state = {f"duration_head.{key}": value for key, value in DurationHead().state_dict().items()}
    from safetensors.torch import save_file

    save_file(state, str(checkpoint))
    asset = LocalAssetRef(
        "duration",
        checkpoint.name,
        checkpoint.stat().st_size,
        checkpoint.name,
        resolver=LocalResolver(checkpoint),
    )
    patcher = ModelPatchLoader.execute(model=asset)["MODEL_PATCH"]
    assert type(patcher).__name__ == "ModelPatcher"
    assert type(patcher.model).__name__ == "DurationHead"
    assert patcher.model.attention_pooler.query_tokens.dtype is torch.float32
