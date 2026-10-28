"""Schema and loader dispatch contracts for latent-upscale assets."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_UPSCALE_MODEL_TYPE,
    AssetRef,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_assets import AssetVault, digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.latent_upscale_loader import LatentUpscaleModelLoader
from safetensors.torch import save_file

from dinkster.compose import PackSpec, ServingComposer


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


class LatentUpscaleConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="latent-upscale-fixture.consumer",
            inputs=(InputSpec("model", TypeExpr.concrete(LATENT_UPSCALE_MODEL_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: object) -> dict[str, object]:
        model_format = os.environ["LATENT_UPSCALE_FORMAT"]
        hunyuan = importlib.import_module("dinkster_inference.ldm.hunyuan_video.upsampler")
        patcher = importlib.import_module("dinkster_inference.model_patcher")
        if model_format == "ltx":
            assert type(model) is patcher.CoreModelPatcher
            state = model.model.state_dict()
            assert next(iter(state.values())).dtype is torch.float32
            actual_type = type(model.model).__name__
        else:
            assert type(model) is hunyuan.HunyuanVideo15SRModel
            state = model.get_sd()
            assert type(model.model).__name__ == (
                "SRModel3DV2" if model_format == "720p" else "Upsampler"
            )
            actual_type = type(model).__name__
        assert state
        for value in state.values():
            torch.testing.assert_close(value, torch.full_like(value, 0.125), rtol=0, atol=0)
        return cls.outputs(summary=json.dumps({"format": model_format, "type": actual_type}))


def _checkpoint(path: Path, model_format: str) -> None:
    initialize_inference_process()
    from dinkster_inference import ops
    from dinkster_inference.ldm.hunyuan_video.upsampler import HunyuanVideo15SRModel
    from dinkster_inference.ldm.lightricks.latent_upsampler import LatentUpsampler

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
        state, metadata = source.get_sd(), None
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
        state, metadata = source.state_dict(), {"config": json.dumps(source.config())}
    save_file(
        {key: torch.full_like(value, 0.125) for key, value in state.items()},
        path,
        metadata=metadata,
    )


@pytest.mark.parametrize(
    "model_format,in_process", (("720p", True), ("1080p", True), ("ltx", True), ("ltx", False))
)
def test_latent_upscale_loader_crosses_ordinary_hosts_twice(
    tmp_path: Path, in_process: bool, model_format: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / f"{model_format}.safetensors"
    _checkpoint(checkpoint, model_format)
    vault = AssetVault(tmp_path / "vault")
    digest = digest_file(checkpoint)
    with vault.writer(digest) as writer, checkpoint.open("rb") as source:
        writer.write(source.read())
        writer.commit()
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    monkeypatch.setenv("LATENT_UPSCALE_FORMAT", model_format)

    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        comfy_manifest = root / "dinkster-pack.toml"
        fixture_manifest = root / "tests/fixtures/latent_upscale/dinkster-pack.toml"
        environment = {
            "CUDA_VISIBLE_DEVICES": "",
            "DINKSTER_ACCELERATOR": "cpu",
            "DINKSTER_ASSET_VAULT": str(vault.root),
            "LATENT_UPSCALE_FORMAT": model_format,
            "PYTHONPATH": os.pathsep.join(
                (
                    str(tmp_path),
                    str(Path(__file__).parent),
                    str(root / "src"),
                    os.environ.get("PYTHONPATH", ""),
                )
            ),
        }
        composer = ServingComposer(worker_env=environment)
        try:
            for manifest in (comfy_manifest, fixture_manifest):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "latent-upscale-fixture",
                        group_manifests=() if in_process else (comfy_manifest, fixture_manifest),
                    )
                )
            graph = Graph(
                nodes={
                    "loader": GraphNode(
                        "LatentUpscaleModelLoader",
                        {
                            "model_name": AssetRef(
                                digest, checkpoint.name, checkpoint.stat().st_size
                            ).to_wire()
                        },
                    ),
                    "consumer": GraphNode(
                        "latent-upscale-fixture.consumer",
                        {"model": Link("loader", "LATENT_UPSCALE_MODEL")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert (
                    json.loads(result.outputs["consumer"]["summary"].resolve())["format"]
                    == model_format
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
