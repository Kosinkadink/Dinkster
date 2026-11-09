"""CPU multi-checkpoint loader asset admission and ordering."""

from __future__ import annotations

import importlib
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
from dinkster_models.clip_loaders import QuadrupleCLIPLoader, TripleCLIPLoader


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
