"""Granted image folders preserve source ordering and repeated caption pairs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_STRING,
    IMAGE_TYPE,
    AssetError,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.dataset_folders import (
    LoadImageDataSetFromFolder,
    LoadImageTextDataSetFromFolder,
)
from PIL import Image

from dinkster.compose import PackSpec, ServingComposer


def seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "input"
    directory = root / "dataset"
    directory.mkdir(parents=True)
    Image.fromarray(np.array([[[17, 31, 61, 0], [127, 191, 251, 255]]], np.uint8)).save(
        directory / "z.PNG"
    )
    Image.fromarray(np.array([[41], [83], [167]], np.uint8)).save(directory / "a.png")
    (directory / "z.txt").write_text("  caption z\n", encoding="utf-8")
    repeated = directory / "2_repeat"
    repeated.mkdir()
    Image.fromarray(np.array([[[3, 5, 11]]], np.uint8)).save(repeated / "b.png")
    (repeated / "b.txt").write_text(" repeated\t", encoding="utf-8")
    skipped = directory / "0_skip"
    skipped.mkdir()
    Image.fromarray(np.zeros((1, 1, 3), np.uint8)).save(skipped / "skip.png")
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {
                        "id": "input",
                        "kind": "directory/input",
                        "root": str(root),
                        "index": str(tmp_path / "no-index.json"),
                        "priority": 0,
                    }
                ]
            }
        ),
        encoding="ascii",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return directory, snapshot


def expected(directory: Path, paired: bool) -> list[str]:
    # Hand-derived order: raw directory entries, with b.png repeated twice by 2_repeat.
    order = []
    for name in os.listdir(directory):
        if name in ("z.PNG", "a.png"):
            order.append(name)
        elif paired and name == "2_repeat":
            order.extend(["b.png", "b.png"])
    return order


def check_images(images: Any, names: list[str]) -> None:
    # Hand-derived from seed pixels: discard alpha, repeat grayscale over RGB, divide by 255.
    values = {
        "z.PNG": [[[17, 31, 61], [127, 191, 251]]],
        "a.png": [[[41, 41, 41]], [[83, 83, 83]], [[167, 167, 167]]],
        "b.png": [[[3, 5, 11]]],
    }
    assert len(images) == len(names)
    for image, name in zip(images, names, strict=True):
        tensor = materialize_image(image)
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor, torch.tensor(values[name], dtype=torch.float32)[None] / 255)


def source_tree_digests(directory: Path) -> dict[Path, str]:
    return {
        path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_folder_images_and_repeated_caption_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, _ = seed(tmp_path, monkeypatch)
    images = LoadImageDataSetFromFolder.execute(folder="dataset")["images"]
    check_images(images, expected(directory, False))
    result = LoadImageTextDataSetFromFolder.execute(folder="dataset")
    names = expected(directory, True)
    check_images(result["images"], names)
    # Hand-derived: strip the seeded sidecars and use empty text for the missing a.txt.
    assert result["texts"] == [
        {"z.PNG": "caption z", "a.png": "", "b.png": "repeated"}[n] for n in names
    ]


def test_folder_loads_revalidate_grants_and_reject_empty_and_escaped_subtrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, snapshot = seed(tmp_path, monkeypatch)
    (directory.parent / "empty").mkdir()
    for node in (LoadImageDataSetFromFolder, LoadImageTextDataSetFromFolder):
        with pytest.raises(ValueError, match="No valid images"):
            node.execute(folder="empty")
        with pytest.raises(AssetError):
            node.execute(folder="../dataset")
    (directory / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(AssetError):
        LoadImageDataSetFromFolder.execute(folder="dataset")
    (directory / "outside").unlink()
    snapshot.write_text('{"mounts":[]}', encoding="ascii")
    with pytest.raises(AssetError):
        LoadImageTextDataSetFromFolder.execute(folder="dataset")


class FolderInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="folder-dataset-fixture.inspect",
            inputs=(
                InputSpec("images", TypeExpr.list_of(TypeExpr.concrete(IMAGE_TYPE))),
                InputSpec("paired", TypeExpr.list_of(TypeExpr.concrete(IMAGE_TYPE))),
                InputSpec("texts", TypeExpr.list_of(TypeExpr.concrete(CORE_STRING))),
            ),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, images: Any, paired: Any, texts: Any) -> Any:
        directory = Path(os.environ["FOLDER_DATASET_FIXTURE"])
        check_images(images, expected(directory, False))
        names = expected(directory, True)
        check_images(paired, names)
        # Hand-derived from the seeded sidecars, as in the direct caption assertion.
        assert texts == [{"z.PNG": "caption z", "a.png": "", "b.png": "repeated"}[n] for n in names]
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_folder_dataset_ordinary_hosts_reread_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    directory, snapshot = seed(tmp_path, monkeypatch)
    source_before = source_tree_digests(directory)
    monkeypatch.setenv("FOLDER_DATASET_FIXTURE", str(directory))

    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        module = f"folder_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_dataset_folders_torch import FolderInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (FolderInspect,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="folder-dataset-fixture"\npublisher="dinkster"\n'
            'namespaces=["folder-dataset-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        paths = (checkout / "src", *sorted(checkout.glob("packages/*/src")))
        composer = ServingComposer(
            worker_env={
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "FOLDER_DATASET_FIXTURE": str(directory),
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        *(str(p) for p in paths),
                        *(p for p in sys.path if p),
                    )
                ),
            }
        )
        try:
            manifests = (package / "dinkster-pack.toml", fixture)
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "folder-dataset-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "images": GraphNode("LoadImageDataSetFromFolder", {"folder": "dataset"}),
                    "pairs": GraphNode("LoadImageTextDataSetFromFolder", {"folder": "dataset"}),
                    "inspect": GraphNode(
                        "folder-dataset-fixture.inspect",
                        {
                            "images": Link("images", "images"),
                            "paired": Link("pairs", "images"),
                            "texts": Link("pairs", "texts"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["valid"].resolve() is True
                assert not ({"images", "pairs"} & set(result.cached))
                assert source_tree_digests(directory) == source_before
        finally:
            await composer.close()

    asyncio.run(scenario())
    assert source_tree_digests(directory) == source_before
