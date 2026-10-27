"""CPU contracts for admitted background models and foreground masks."""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    BACKGROUND_REMOVAL_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    MASK_TYPE,
    AssetRef,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)
from dinkster_assets import AssetError, MountDef, MountTable, digest_file
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy import background_removal
from dinkster_nodes_comfy.background_removal import LoadBackgroundRemovalModel, RemoveBackground

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize("local", (False, True))
@pytest.mark.parametrize("valid", (False, True))
def test_loader_preserves_admitted_model_identity_and_rejects_invalid_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local: bool, valid: bool
) -> None:
    path = tmp_path / "foreground.safetensors"
    path.write_bytes(b"controlled checkpoint")

    class Resolver:
        def resolve(self, digest: str) -> Path:
            return path

        def resolve_local(self, binding: str) -> tuple[Path, tuple[int, int, int, int, int]]:
            stat = path.stat()
            return path, (
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_dev,
                stat.st_ino,
            )

        def reconcile_local(self, binding: str) -> AssetRef | None:
            return None

    resolver = Resolver()
    asset = (
        LocalAssetRef("host-binding", path.name, path.stat().st_size, path.name, resolver=resolver)
        if local
        else AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=resolver)
    )
    model = object()

    def load(filename: str) -> object:
        assert filename == str(path)
        return model if valid else None

    monkeypatch.setattr(background_removal, "initialize_inference_process", lambda: None)
    monkeypatch.setattr(
        background_removal.importlib, "import_module", lambda name: SimpleNamespace(load=load)
    )
    if valid:
        assert LoadBackgroundRemovalModel.execute(bg_removal_name=asset)["bg_model"] is model
    else:
        with pytest.raises(RuntimeError) as error:
            LoadBackgroundRemovalModel.execute(bg_removal_name=asset)
        assert str(error.value) == (
            "ERROR: background model file is invalid and does not contain a valid "
            "background removal model."
        )


@pytest.mark.parametrize(
    "asset", ("foreground.safetensors", AssetRef("blake3:" + "b" * 64, "missing", 1))
)
def test_loader_rejects_unadmitted_input_before_inference_initialization(
    monkeypatch: pytest.MonkeyPatch, asset: object
) -> None:
    def forbidden() -> None:
        pytest.fail("unadmitted path initialized inference")

    monkeypatch.setattr(background_removal, "initialize_inference_process", forbidden)
    with pytest.raises((TypeError, AssetError)):
        LoadBackgroundRemovalModel.execute(bg_removal_name=asset)


@pytest.mark.parametrize("portable", (False, True))
def test_remove_background_materializes_batch_and_preserves_returned_mask_identity(
    portable: bool,
) -> None:
    source = torch.arange(2 * 3 * 5 * 3, dtype=torch.float32).reshape(2, 3, 5, 3) / 91
    original = source.clone()
    mask = source[..., 0] * 0.25 + source[..., 2] * 0.75

    class Model:
        def encode_image(self, image: torch.Tensor) -> torch.Tensor:
            torch.testing.assert_close(image, original, rtol=0, atol=0)
            if not portable:
                assert image is source
            return mask

    result = RemoveBackground.execute(
        bg_removal_model=Model(), image=source.numpy() if portable else source
    )
    assert result["mask"] is mask
    torch.testing.assert_close(source, original, rtol=0, atol=0)


class ControlledBackgroundModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        assert image.shape == (2, 3, 5, 3)
        self.calls += 1
        return image[..., 0] * 0.25 + image[..., 2] * 0.75


def install_controlled_loader() -> None:
    initialize_inference_process()
    loader = importlib.import_module("dinkster_inference.bg_removal_model")

    def load(filename: str) -> ControlledBackgroundModel:
        assert Path(filename).read_bytes() == b"controlled checkpoint"
        return ControlledBackgroundModel()

    loader.load = load


class BackgroundSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "background-fixture.source",
            outputs=(OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        image = torch.arange(90, dtype=torch.float32).reshape(2, 3, 5, 3) / 91
        return cls.outputs(image=image.numpy())


class BackgroundInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "background-fixture.inspect",
            inputs=(
                InputSpec("model", TypeExpr.concrete(BACKGROUND_REMOVAL_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec("mask", TypeExpr.concrete(MASK_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls, *, model: ControlledBackgroundModel, image: Any, mask: Any
    ) -> dict[str, object]:
        source = materialize_image(image)
        actual = materialize_image(mask)
        expected_source = torch.arange(90, dtype=torch.float32).reshape(2, 3, 5, 3) / 91
        torch.testing.assert_close(source, expected_source, rtol=0, atol=0)
        expected = expected_source[..., 0] * 0.25 + expected_source[..., 2] * 0.75
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert model.calls == 1
        return cls.outputs(summary="resident model and exact foreground masks")


@pytest.mark.parametrize("verified", (False, True))
@pytest.mark.parametrize("in_process", (True, False))
def test_admitted_background_loader_to_mask_through_ordinary_manifest_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verified: bool, in_process: bool
) -> None:
    checkpoint = tmp_path / "foreground.safetensors"
    checkpoint.write_bytes(b"controlled checkpoint")
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="foreground", path=tmp_path), kind="model/background-removal")
    if verified:
        table.scan("foreground")
    asset = table.bind_name(checkpoint.name, "model/background-removal")
    assert isinstance(asset, AssetRef if verified else LocalAssetRef)
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(tmp_path / "mounts.json"))
    initialize_inference_process()
    loader = importlib.import_module("dinkster_inference.bg_removal_model")
    original_load = loader.load

    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        module = f"background_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_background_removal_torch import (\n"
            "    BackgroundSource, BackgroundInspect, install_controlled_loader\n)\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "install_controlled_loader()\n"
            "NODES = (BackgroundSource, BackgroundInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname = "background-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["background-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "DINKSTER_MOUNTS_SNAPSHOT": str(tmp_path / "mounts.json"),
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
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
                        worker_group="background-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("background-fixture.source"),
                    "load": GraphNode(
                        "LoadBackgroundRemovalModel", {"bg_removal_name": asset.to_wire()}
                    ),
                    "remove": GraphNode(
                        "RemoveBackground",
                        {
                            "bg_removal_model": Link("load", "bg_model"),
                            "image": Link("source", "image"),
                        },
                    ),
                    "inspect": GraphNode(
                        "background-fixture.inspect",
                        {
                            "model": Link("load", "bg_model"),
                            "image": Link("source", "image"),
                            "mask": Link("remove", "mask"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                output = await engine.run(graph, ["inspect"])
                assert output.outputs["inspect"]["summary"].resolve() == (
                    "resident model and exact foreground masks"
                )
        finally:
            await composer.close()

    try:
        asyncio.run(scenario())
    finally:
        loader.load = original_load
