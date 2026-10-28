"""Real checkpoint bytes at the MoGe loader's granted file boundary."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    MOGE_MODEL_TYPE,
    AssetRef,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_assets import AssetError, MountDef, MountTable, digest_file
from dinkster_assets.model import LocalFileSignature, local_file_signature
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.moge_loader import LoadMoGeModel
from safetensors.torch import save_file

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize("format", ["safetensors", "pt"])
@pytest.mark.parametrize("local", [False, True])
def test_moge_loader_preserves_checkpoint_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format: str, local: bool
) -> None:
    initialize_inference_process()
    from dinkster_inference.ldm.moge import model

    class LoadedModel:
        def __init__(self, state: dict[str, torch.Tensor]) -> None:
            self.state = state

    monkeypatch.setattr(model, "MoGeModel", LoadedModel)
    state = {
        "encoder.weight": torch.tensor([[1.25, -3.5], [7.0, 0.125]], dtype=torch.float16),
        "mask_threshold": torch.tensor(0.375),
        "indices": torch.tensor([4, 1, 9], dtype=torch.int64),
    }
    path = tmp_path / f"geometry.{format}"
    if format == "safetensors":
        save_file(state, path)
    else:
        torch.save(state, path)

    class Resolver:
        def resolve(self, digest: str) -> Path:
            return path

        def resolve_local(self, binding: str) -> tuple[Path, LocalFileSignature]:
            return path, local_file_signature(path.stat())

        def reconcile_local(self, binding: str) -> None:
            return None

    asset: AssetRef | LocalAssetRef
    if local:
        asset = LocalAssetRef(
            "granted", path.name, path.stat().st_size, path.name, resolver=Resolver()
        )
    else:
        asset = AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=Resolver())
    loaded = LoadMoGeModel.execute(model_name=asset)["MOGE_MODEL"]
    assert isinstance(loaded, LoadedModel)
    assert loaded.state.keys() == state.keys()
    for key, tensor in state.items():
        assert loaded.state[key].dtype == tensor.dtype
        assert torch.equal(loaded.state[key], tensor)


@pytest.mark.parametrize("value", ["ungranted.safetensors", Path("ungranted.pt"), None])
def test_moge_loader_rejects_ungranted_paths(value: object) -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        LoadMoGeModel.execute(model_name=value)


def test_moge_loader_rejects_revoked_binding() -> None:
    asset = LocalAssetRef("revoked", "missing.pt", 4, "geometry/missing.pt")
    with pytest.raises(AssetError, match="unavailable, revoked or changed"):
        LoadMoGeModel.execute(model_name=asset)


class CheckpointModel:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self.state = state


def install_constructor() -> None:
    initialize_inference_process()
    from dinkster_inference.ldm.moge import model

    model.MoGeModel = CheckpointModel


class InspectMoGeModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="moge-loader-fixture.inspect",
            inputs=(InputSpec("model", TypeExpr.concrete(MOGE_MODEL_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: Any) -> dict[str, object]:
        assert isinstance(model, CheckpointModel)
        assert set(model.state) == {"marker"}
        assert model.state["marker"].dtype == torch.float16
        assert model.state["marker"].shape == (2, 3)
        assert torch.equal(
            model.state["marker"],
            torch.tensor([[1.25, -3.5, 7], [0.125, 9, -2]], dtype=torch.float16),
        )
        return cls.outputs(summary="checkpoint state reached MoGe observer")


@pytest.mark.parametrize("verified", [False, True])
@pytest.mark.parametrize("in_process", [False, True])
def test_moge_loader_transports_resident_model_through_ordinary_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verified: bool, in_process: bool
) -> None:
    initialize_inference_process()
    from dinkster_inference.ldm.moge import model

    monkeypatch.setattr(model, "MoGeModel", CheckpointModel)
    path = tmp_path / "geometry.safetensors"
    save_file(
        {"marker": torch.tensor([[1.25, -3.5, 7], [0.125, 9, -2]], dtype=torch.float16)}, path
    )
    before = digest_file(path)
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="geometry", path=tmp_path), kind="model/geometry-estimation")
    if verified:
        table.scan("geometry")
    asset = table.bind_name(path.name, "model/geometry-estimation")
    assert isinstance(asset, AssetRef if verified else LocalAssetRef)
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(tmp_path / "mounts.json"))
    module = "moge_loader_fixture_" + tmp_path.name.replace("-", "_")
    (tmp_path / f"{module}.py").write_text(
        "from test_moge_loader_torch import InspectMoGeModel, install_constructor\n"
        "from dinkster_nodes_comfy import register_inference_types\n"
        "install_constructor()\n"
        "NODES = (InspectMoGeModel,)\n",
        encoding="ascii",
    )
    fixture = tmp_path / "dinkster-pack.toml"
    fixture.write_text(
        '[pack]\nname = "moge-loader-fixture"\npublisher = "dinkster"\n'
        'namespaces = ["moge-loader-fixture"]\n[pack.entry]\n'
        f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
        encoding="ascii",
    )
    package = Path(__file__).resolve().parents[1]
    checkout = package.parents[1]
    native = package / "dinkster-pack.toml"

    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(tmp_path / "mounts.json"),
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        str(checkout / "src"),
                        *(str(p) for p in sorted(checkout.glob("packages/*/src"))),
                        *(p for p in sys.path if p),
                    )
                ),
            }
        )
        try:
            manifests = (native, fixture)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "moge-loader-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "load": GraphNode("LoadMoGeModel", {"model_name": asset.to_wire()}),
                    "inspect": GraphNode(
                        "moge-loader-fixture.inspect", {"model": Link("load", "MOGE_MODEL")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == (
                    "checkpoint state reached MoGe observer"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
    assert digest_file(path) == before
