"""Real tiny upscale checkpoints through granted files and ordinary hosts."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    UPSCALE_MODEL_TYPE,
    AssetError,
    AssetRef,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_assets import MountDef, MountTable
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.upscale_model_loader import UpscaleModelLoader
from safetensors.torch import save_file
from spandrel import ImageModelDescriptor, UnsupportedModelError
from spandrel.architectures.Compact import Compact

from dinkster.compose import PackSpec, ServingComposer


def seed(tmp_path: Path, *, verified: bool) -> tuple[MountTable, AssetRef | LocalAssetRef]:
    model = Compact(num_feat=2, num_conv=0, upscale=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.body[-1].bias.copy_(torch.tensor([0.125] * 4 + [0.25] * 4 + [0.5] * 4))
    save_file(model.state_dict(), str(tmp_path / "tiny.safetensors"))
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="upscalers", path=tmp_path), kind="model/upscaler")
    if verified:
        table.scan("upscalers")
    asset = table.bind_name("tiny.safetensors", "model/upscaler")
    assert isinstance(asset, AssetRef if verified else LocalAssetRef)
    return table, asset


def check_model(model: Any) -> None:
    from dinkster_inference.model_patcher import CoreModelPatcher

    assert isinstance(model, ImageModelDescriptor)
    assert isinstance(model.patcher, CoreModelPatcher)
    assert model.patcher.model is model.model
    assert not model.model.training
    assert model.scale == 2
    assert model.patcher.load_device.type == "cpu"
    assert model.patcher.offload_device.type == "cpu"
    image = torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3) / 64
    before = image.clone()
    # Hand-derived Compact result: zero convolutions leave the final RGB biases,
    # then the architecture adds its nearest-neighbor residual to the input.
    expected = image.repeat_interleave(2, -2).repeat_interleave(2, -1)
    expected = expected + torch.tensor([0.125, 0.25, 0.5]).reshape(1, 3, 1, 1)
    with torch.inference_mode():
        actual = model(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(image, before, rtol=0, atol=0)


@pytest.mark.parametrize("verified", (False, True))
def test_loader_consumes_granted_real_checkpoint_without_mutation(
    tmp_path: Path, verified: bool
) -> None:
    _table, asset = seed(tmp_path, verified=verified)
    path = tmp_path / "tiny.safetensors"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    check_model(UpscaleModelLoader.execute(model_name=asset)["UPSCALE_MODEL"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_loader_rejects_raw_paths_changed_grants_and_unknown_weights(tmp_path: Path) -> None:
    table, asset = seed(tmp_path, verified=False)
    path = tmp_path / "tiny.safetensors"
    for value in (path, str(path), "../tiny.safetensors"):
        with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
            UpscaleModelLoader.execute(model_name=value)
    save_file({"unrecognized": torch.tensor([7.0])}, str(path))
    with pytest.raises(AssetError):
        UpscaleModelLoader.execute(model_name=asset)
    changed = table.bind_name(path.name, "model/upscaler")
    assert isinstance(changed, LocalAssetRef)
    with pytest.raises(UnsupportedModelError):
        UpscaleModelLoader.execute(model_name=changed)


class UpscaleLoaderConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="upscale-loader-fixture.consumer",
            inputs=(InputSpec("model", TypeExpr.concrete(UPSCALE_MODEL_TYPE)),),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, model: Any) -> dict[str, object]:
        check_model(model)
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_loader_resident_output_executes_and_caches_in_ordinary_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    _table, asset = seed(tmp_path, verified=False)
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(tmp_path / "mounts.json"))
    path = tmp_path / "tiny.safetensors"
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "upscale_loader_fixture.py").write_text(
            "from test_upscale_model_loader_torch import UpscaleLoaderConsumer\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (UpscaleLoaderConsumer,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="upscale-loader-fixture"\npublisher="dinkster"\n'
            'namespaces=["upscale-loader-fixture"]\n[pack.entry]\n'
            'nodes="upscale_loader_fixture:NODES"\n'
            'types="upscale_loader_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "DINKSTER_ACCELERATOR": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_MOUNTS_SNAPSHOT": str(tmp_path / "mounts.json"),
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(native.parent / "src"))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest, worker_group="upscale-loader", group_manifests=(native, fixture)
                    )
                )
            graph = Graph(
                nodes={
                    "load": GraphNode("UpscaleModelLoader", {"model_name": asset.to_wire()}),
                    "consume": GraphNode(
                        "upscale-loader-fixture.consumer", {"model": Link("load", "UPSCALE_MODEL")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for run in range(2):
                result = await engine.run(graph, ["consume"])
                assert result.outputs["consume"]["valid"].resolve() is True
                if run == 1:
                    assert set(result.cached) == {"load", "consume"}
                assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        finally:
            await composer.close()

    asyncio.run(scenario())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
