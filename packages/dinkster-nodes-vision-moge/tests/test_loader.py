"""Checkpoint loading through host asset grants, without model downloads."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
from dinkster_api.v1 import initialize_inference_process
from dinkster_assets import MountDef, MountTable

LoadMoGeModel = importlib.import_module("dinkster_vision_moge.loader").LoadMoGeModel


def test_loader_reconstructs_synthetic_checkpoint_from_granted_asset(tmp_path: Path) -> None:
    initialize_inference_process()
    torch = importlib.import_module("torch")
    module = importlib.import_module("dinkster_inference.ldm.moge.model")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(640)
        original = module.MoGeModelV1(
            backbone={
                "hidden_size": 64,
                "num_attention_heads": 1,
                "num_hidden_layers": 4,
                "layer_norm_eps": 1e-6,
                "use_swiglu_ffn": False,
            },
            dim_upsample=[32, 32, 32],
            operations=torch.nn,
        )
        # DINO tokens and layer scales use torch.empty, even with torch.nn operations.
        with torch.no_grad():
            for parameter in original.parameters():
                parameter.uniform_(-0.25, 0.75)
    state: dict[str, Any] = original.state_dict()
    importlib.import_module("safetensors.torch").save_file(state, tmp_path / "moge.safetensors")
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef("geometry", tmp_path), kind="model/geometry-estimation")
    asset = table.bind_name("moge.safetensors", "model/geometry-estimation")
    loaded: Any = LoadMoGeModel.execute(model_name=asset)["MOGE_MODEL"]
    assert loaded.version == "v1"
    assert loaded.num_tokens_range == (1200, 2500)
    assert loaded.mask_threshold == 0.5
    assert loaded.patcher.model is loaded.model
    restored: dict[str, Any] = loaded.model.state_dict()
    assert restored.keys() == state.keys()
    for name, expected in state.items():
        torch.testing.assert_close(
            restored[name].cpu(), expected.to(restored[name].dtype), rtol=0, atol=0
        )


def test_loader_rejects_ungranted_path_before_reading_checkpoint() -> None:
    with pytest.raises(TypeError, match="model_name must be an AssetRef or LocalAssetRef"):
        LoadMoGeModel.execute(model_name="/tmp/moge.safetensors")
