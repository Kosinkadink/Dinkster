"""Schema and loader dispatch contracts for latent-upscale assets."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import LATENT_UPSCALE_MODEL_TYPE, AssetRef, initialize_inference_process
from dinkster_assets import digest_file
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.latent_upscale_loader import LatentUpscaleModelLoader
from safetensors.torch import save_file


class _Asset:
    def __init__(self, path: Path) -> None:
        self.path = path

    def local_path(self) -> Path:
        return self.path


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, _digest: str) -> Path:
        return self.path


def _asset(path: Path) -> AssetRef:
    return AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=_Resolver(path))


def test_loader_is_registered_with_the_canonical_core_schema_without_torch() -> None:
    assert LatentUpscaleModelLoader in COMFY_NODES
    schema = LatentUpscaleModelLoader.schema()
    assert schema.node_type == "LatentUpscaleModelLoader"
    assert schema.display_name == "Load Latent Upscale Model"
    assert schema.category == "model/loaders"
    assert [input.id for input in schema.inputs] == ["model_name"]
    assert schema.outputs[0].type.types == (LATENT_UPSCALE_MODEL_TYPE,)
    assert schema.inputs[0].widget is not None
    assert schema.inputs[0].widget.accept == ("application/octet-stream",)
    assert schema.inputs[0].widget.kind == "model/latent-upscaler"


def test_loader_rejects_unbound_non_asset_values() -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        LatentUpscaleModelLoader.execute(model_name=object())


def test_hunyuan_720p_dispatch_derives_the_reference_config(monkeypatch, tmp_path: Path) -> None:
    state = {
        "in_conv.conv.weight": SimpleNamespace(shape=(8, 3, 3, 3, 3)),
        "out_conv.conv.weight": SimpleNamespace(shape=(5, 8, 3, 3, 3)),
        "blocks.0.block.0.conv.weight": object(),
        "blocks.1.block.0.conv.weight": object(),
    }
    captured: dict[str, Any] = {}

    class Model:
        def __init__(self, model_type: str, config: dict[str, object], fast_disk: bool) -> None:
            captured.update(model_type=model_type, config=config, fast_disk=fast_disk)

        def load_sd(self, received: object) -> None:
            captured["state"] = received

    modules = {
        "dinkster_inference.utils": SimpleNamespace(
            load_torch_file=lambda *args, **kwargs: (state, {})
        ),
        "dinkster_inference.storage": SimpleNamespace(
            state_dict_fast_disk=lambda received: received is state
        ),
        "dinkster_inference.ldm.hunyuan_video.upsampler": SimpleNamespace(
            HunyuanVideo15SRModel=Model
        ),
    }
    monkeypatch.setattr(
        "dinkster_nodes_comfy.latent_upscale_loader.initialize_inference_process", lambda: None
    )
    monkeypatch.setattr(
        "dinkster_nodes_comfy.latent_upscale_loader._asset_path", lambda _: str(tmp_path / "model")
    )
    monkeypatch.setattr(
        "dinkster_nodes_comfy.latent_upscale_loader.importlib.import_module", modules.__getitem__
    )
    result = LatentUpscaleModelLoader.execute(model_name=_Asset(tmp_path / "model"))
    assert isinstance(result["LATENT_UPSCALE_MODEL"], Model)
    assert captured == {
        "model_type": "720p",
        "config": {
            "in_channels": 3,
            "out_channels": 5,
            "hidden_channels": 8,
            "num_blocks": 2,
            "global_residual": False,
        },
        "fast_disk": True,
        "state": state,
    }


@pytest.mark.parametrize("model_format", ("720p", "1080p", "ltx"))
def test_real_safetensors_checkpoints_load_the_expected_pinned_models(
    tmp_path: Path, model_format: str
) -> None:
    initialize_inference_process()
    from dinkster_inference import ops
    from dinkster_inference.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel
    from dinkster_inference.ldm.lightricks.latent_upsampler import LatentUpsampler
    from dinkster_inference.model_patcher import CoreModelPatcher

    if model_format == "720p":
        source = HunyuanVideo15SRModel(
            "720p",
            {
                "in_channels": 2,
                "out_channels": 2,
                "hidden_channels": 2,
                "num_blocks": 1,
                "global_residual": False,
            },
        )
        state = source.get_sd()
        metadata = None
    elif model_format == "1080p":
        source = HunyuanVideo15SRModel(
            "1080p", {"z_channels": 4, "out_channels": 4, "block_out_channels": (32,)}
        )
        state = {
            key.replace("nin_shortcut.conv", "nin_shortcut", 1): value
            for key, value in source.get_sd().items()
        }
        metadata = None
    else:
        source = LatentUpsampler.from_config(
            {
                "in_channels": 4,
                "mid_channels": 32,
                "num_blocks_per_stage": 1,
                "dims": 2,
                "spatial_upsample": True,
                "temporal_upsample": False,
            },
            operations=ops.disable_weight_init,
        )
        state = source.state_dict()
        metadata = {"config": __import__("json").dumps(source.config())}
    state = {key: torch.full_like(value, 0.125) for key, value in state.items()}
    checkpoint = tmp_path / f"{model_format}.safetensors"
    save_file(state, checkpoint, metadata=metadata)

    loaded = LatentUpscaleModelLoader.execute(model_name=_asset(checkpoint))["LATENT_UPSCALE_MODEL"]
    if model_format == "ltx":
        assert type(loaded) is CoreModelPatcher
        actual = loaded.model.state_dict()
    else:
        assert type(loaded) is HunyuanVideo15SRModel
        actual = loaded.get_sd()
    expected = (
        {key.replace("nin_shortcut", "nin_shortcut.conv", 1): value for key, value in state.items()}
        if model_format == "1080p"
        else state
    )
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
