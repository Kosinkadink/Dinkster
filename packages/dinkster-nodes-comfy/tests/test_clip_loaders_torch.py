"""CPU multi-checkpoint loader asset admission and ordering."""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any

import pytest
from dinkster_api.v1 import (
    ASSET_TYPE,
    CLIP_TYPE,
    CORE_INT,
    CORE_STRING,
    AssetRef,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    resolver_from_env,
)
from dinkster_assets import MountDef, MountTable, digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.clip_loaders import QuadrupleCLIPLoader, TripleCLIPLoader

from dinkster.compose import PackSpec, ServingComposer


class LoaderAssets(Node):
    original_load_clip: Any = None

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "clip-loader-fixture.assets",
            inputs=(InputSpec("root", TypeExpr.concrete(CORE_STRING)),),
            outputs=tuple(
                OutputSpec(f"asset{i}", TypeExpr.concrete(ASSET_TYPE)) for i in range(1, 5)
            ),
        )

    @classmethod
    def execute(cls, *, root: str) -> dict[str, object]:
        initialize_inference_process()
        sd = importlib.import_module("dinkster_inference.sd")
        cls.original_load_clip = sd.load_clip
        sd.load_clip = lambda **options: options
        return cls.outputs(
            **{
                f"asset{i}": AssetRef(digest_file(path), path.name, 1, resolver=resolver_from_env())
                for i, index in enumerate((8, 2, 11, 3), 1)
                for path in (Path(root) / f"encoder-{index}.safetensors",)
            }
        )


class LoaderInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "clip-loader-fixture.inspect",
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("count", TypeExpr.concrete(CORE_INT)),
                InputSpec("root", TypeExpr.concrete(CORE_STRING)),
            ),
            outputs=(OutputSpec("count", TypeExpr.concrete(CORE_INT)),),
        )

    @classmethod
    def execute(cls, *, clip: Any, count: int, root: str) -> dict[str, object]:
        try:
            assert clip == {
                "ckpt_paths": [
                    str(Path(root) / f"encoder-{index}.safetensors")
                    for index in (8, 2, 11, 3)[:count]
                ],
                "embedding_directory": [str(Path(root) / name) for name in ("early", "late")],
            }
            return cls.outputs(count=count)
        finally:
            importlib.import_module(
                "dinkster_inference.sd"
            ).load_clip = LoaderAssets.original_load_clip


@pytest.mark.parametrize("in_process", (True,))
@pytest.mark.parametrize("count", (3, 4))
def test_clip_loaders_execute_in_ordinary_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, count: int
) -> None:
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot, index_root=tmp_path / "index")
    for index in (8, 2, 11, 3):
        (tmp_path / f"encoder-{index}.safetensors").write_bytes(bytes([index]))
    table.add(MountDef(id="encoders", path=tmp_path), kind="model/text-encoder")
    for name, priority in (("late", 8), ("early", 2)):
        path = tmp_path / name
        path.mkdir()
        table.add(MountDef(id=name, path=path, priority=priority), kind="model/embedding")
        table.scan(name)
    table.scan("encoders")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    initialize_inference_process()
    sd = importlib.import_module("dinkster_inference.sd")
    original = sd.load_clip

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"clip_loader_fixture_{in_process}_{count}"
        (tmp_path / f"{module}.py").write_text(
            "from test_clip_loaders_torch import LoaderAssets, LoaderInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (LoaderAssets, LoaderInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname = "clip-loader-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["clip-loader-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="clip-loader-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "assets": GraphNode("clip-loader-fixture.assets", {"root": str(tmp_path)}),
                    "load": GraphNode(
                        "TripleCLIPLoader" if count == 3 else "QuadrupleCLIPLoader",
                        {f"clip_name{i}": Link("assets", f"asset{i}") for i in range(1, count + 1)},
                    ),
                    "inspect": GraphNode(
                        "clip-loader-fixture.inspect",
                        {
                            "clip": Link("load", "CLIP"),
                            "count": count,
                            "root": str(tmp_path),
                        },
                    ),
                }
            )
            for _ in range(2):
                engine = composer.composition.make_engine(lambda _event: None)
                assert (await engine.run(graph, ["inspect"])).outputs["inspect"][
                    "count"
                ].resolve() == count
        finally:
            await composer.close()

    try:
        asyncio.run(scenario())
    finally:
        sd.load_clip = original


@pytest.mark.parametrize("count", (3, 4))
@pytest.mark.parametrize("verified", (False, True))
def test_clip_loader_keeps_asset_order_and_exact_embedding_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, verified: bool
) -> None:
    initialize_inference_process()
    sd = importlib.import_module("dinkster_inference.sd")
    paths = [tmp_path / f"encoder-{index}.safetensors" for index in (8, 2, 11, 3)[:count]]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index + 1]))
    snapshot = tmp_path / "mounts.json"
    grants = MountTable(snapshot, index_root=tmp_path / "index")
    grants.add(MountDef(id="encoders", path=tmp_path), kind="model/text-encoder")
    first, second, unrelated = (tmp_path / name for name in ("first", "second", "unrelated"))
    for root in (first, second, unrelated):
        root.mkdir()
    grants.add(MountDef(id="late", path=first, priority=8), kind="model/embedding")
    grants.add(MountDef(id="early", path=second, priority=2), kind="model/embedding")
    grants.add(MountDef(id="other", path=unrelated, priority=0), kind="model/checkpoint")
    for mount_id in ("late", "early", "other"):
        grants.scan(mount_id)
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))

    class Resolver:
        def __init__(self, path: Path) -> None:
            self.path = path

        def resolve(self, digest: str) -> Path:
            return self.path

    assets: list[AssetRef | LocalAssetRef] = []
    for path in paths:
        if verified:
            assets.append(AssetRef(digest_file(path), path.name, 1, resolver=Resolver(path)))
        else:
            ref = grants.bind_name(path.name, "model/text-encoder")
            assert isinstance(ref, LocalAssetRef)
            assets.append(ref)
    result = object()

    def load_clip(**options: Any) -> object:
        assert options == {
            "ckpt_paths": [str(path) for path in paths],
            "embedding_directory": [str(second), str(first)],
        }
        return result

    monkeypatch.setattr(sd, "load_clip", load_clip)
    node = TripleCLIPLoader if count == 3 else QuadrupleCLIPLoader
    assert (
        node.execute(**{f"clip_name{i}": value for i, value in enumerate(assets, 1)})["CLIP"]
        is result
    )
    with pytest.raises(TypeError, match=f"clip_name{count}"):
        node.execute(
            **{
                f"clip_name{i}": value if i != count else paths[-1].name
                for i, value in enumerate(assets, 1)
            }
        )
