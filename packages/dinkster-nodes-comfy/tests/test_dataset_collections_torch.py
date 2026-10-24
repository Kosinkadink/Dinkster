"""CPU behavior for native dataset collection nodes."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import PORTS_NODE_ID, Graph, GraphNode, Link, RegionNode, RegionOutput
from dinkster_nodes_comfy.dataset_collections import (
    ImageDeduplication,
    ImageGrid,
    MergeTextLists,
    ShuffleDataset,
    ShuffleImageTextDataset,
)

from dinkster.compose import PackSpec, ServingComposer

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_IMAGES = TypeExpr.list_of(_IMAGE)
_STRING = TypeExpr.concrete(CORE_STRING)
_STRINGS = TypeExpr.list_of(_STRING)


def _images() -> list[torch.Tensor]:
    return [torch.full((2, 3, 4, 3), value / 10, dtype=torch.float32) for value in (1, 2, 3)]


def test_shuffle_flattens_batches_with_reference_numpy_order() -> None:
    images = _images()
    result = ShuffleDataset.execute(images=images, seed=0)["images"]
    assert [float(image[0, 0, 0, 0]) for image in result] == [
        0.30000001192092896,
        0.20000000298023224,
        0.10000000149011612,
        0.20000000298023224,
        0.10000000149011612,
        0.30000001192092896,
    ]


def test_shuffle_image_text_preserves_pairing() -> None:
    images = [torch.full((1, 2, 2, 3), value, dtype=torch.float32) for value in (1, 2, 3)]
    result = ShuffleImageTextDataset.execute(images=images, texts=["one", "two", "three"], seed=0)
    assert [float(image[0, 0, 0, 0]) for image in result["images"]] == [3, 2, 1]
    assert result["texts"] == ["three", "two", "one"]


def test_deduplication_uses_the_reference_threshold_boundary() -> None:
    image = torch.arange(3 * 9 * 11, dtype=torch.float32).reshape(1, 9, 11, 3) / (3 * 9 * 11)
    altered = image.clone()
    altered[..., 0] = 1.0 - altered[..., 0]
    result = ImageDeduplication.execute(
        images=[image, image.clone(), altered], similarity_threshold=0.95
    )
    assert len(result["images"]) == 1
    assert ImageDeduplication.execute(images=[], similarity_threshold=0.95) == {"images": []}
    half = torch.zeros((1, 8, 8, 3))
    half[:, 4:] = 1
    opposite = half.flip(1)
    assert (
        len(ImageDeduplication.execute(images=[half, opposite], similarity_threshold=0)["images"])
        == 1
    )
    kept = ImageDeduplication.execute(
        images=[half, opposite, half.clone()], similarity_threshold=1
    )["images"]
    assert len(kept) == 2
    assert torch.equal(kept[0], half) and torch.equal(kept[1], opposite)


def test_image_grid_quantizes_and_pads_like_the_reference() -> None:
    red = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    blue = torch.tensor([[[[0.0, 0.0, 1.0]]]])
    result = ImageGrid.execute(
        images=[red, blue], columns=2, cell_width=2, cell_height=2, padding=1
    )["images"]
    assert tuple(result.shape) == (1, 2, 5, 3)
    assert torch.equal(result[0, :, :2], torch.tensor([[[1.0, 0.0, 0.0]]]).expand(2, 2, 3))
    assert torch.equal(result[0, :, 2], torch.zeros((2, 3)))
    assert torch.equal(result[0, :, 3:], torch.tensor([[[0.0, 0.0, 1.0]]]).expand(2, 2, 3))
    try:
        ImageGrid.execute(
            images=[torch.empty((0, 1, 1, 3))],
            columns=2,
            cell_width=2,
            cell_height=2,
            padding=1,
        )
    except ValueError as error:
        assert str(error) == "Cannot create grid from empty image list"
    else:
        raise AssertionError("Empty image batches must not produce a grid")


def test_merge_text_lists_preserves_the_reference_nested_payload() -> None:
    assert MergeTextLists.execute(texts=["one", "two"]) == {"texts": [["one", "two"]]}


class CollectionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "collection-fixture.source",
            outputs=(
                OutputSpec("batches", _IMAGES),
                OutputSpec("paired_images", _IMAGES),
                OutputSpec("texts", _STRINGS),
                OutputSpec("grid_images", _IMAGES),
                OutputSpec("seeds", TypeExpr.list_of(TypeExpr.concrete(CORE_INT))),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(
            batches=_images(),
            paired_images=[torch.full((1, 2, 2, 3), float(value)) for value in (1, 2, 3)],
            texts=["one", "two", "three"],
            grid_images=[torch.tensor([[[[1.0, 0.0, 0.0]]]]), torch.tensor([[[[0.0, 0.0, 1.0]]]])],
            seeds=[0, 19],
        )


class ObservePair(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "collection-fixture.pair",
            inputs=(InputSpec("image", _IMAGE), InputSpec("text", _STRING)),
            outputs=(OutputSpec("pair", _STRING),),
        )

    @classmethod
    def execute(cls, *, image: Any, text: str) -> Any:
        return cls.outputs(pair=f"{text}:{int(materialize_image(image)[0, 0, 0, 0])}")


class ObserveCollections(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "collection-fixture.observe",
            inputs=(
                InputSpec("shuffled", _IMAGES),
                InputSpec("deduplicated", _IMAGES),
                InputSpec("grid", _IMAGE),
                InputSpec("pairs", _STRINGS),
            ),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, shuffled: Any, deduplicated: Any, grid: Any, pairs: Any) -> Any:
        assert len(shuffled) == 6
        for item, expected in zip(shuffled, (0.3, 0.2, 0.1, 0.2, 0.1, 0.3), strict=True):
            assert torch.equal(materialize_image(item), torch.full((1, 3, 4, 3), expected))
        assert len(deduplicated) == 1
        assert torch.equal(materialize_image(deduplicated[0]), torch.full((1, 3, 4, 3), 0.1))
        assert pairs == ["three:3", "two:2", "one:1"]
        grid = materialize_image(grid)
        assert tuple(grid.shape) == (1, 32, 65, 3)
        assert torch.equal(grid[0, :, :32], torch.tensor([1.0, 0.0, 0.0]).expand(32, 32, 3))
        assert torch.equal(grid[0, :, 32], torch.zeros((32, 3)))
        assert torch.equal(grid[0, :, 33:], torch.tensor([0.0, 0.0, 1.0]).expand(32, 32, 3))
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_collections_ordinary_manifest_selection_and_paired_map(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(Path.cwd() / "src")))
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "collection_fixture.py").write_text(
                "from test_dataset_collections_torch import (\n"
                "    CollectionSource, ObservePair, ObserveCollections,\n"
                ")\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, CollectionSource, ObservePair, ObserveCollections)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["collection-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "collection_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "collection_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(
                PackSpec(
                    manifest=manifest.parent.parent
                    / "dinkster-nodes-foundation/dinkster-pack.toml",
                    in_process=in_process,
                    trust_reserved=True,
                )
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("collection-fixture.source"),
                    "seed": GraphNode(
                        "std.list.element", {"list": Link("source", "seeds"), "index": 0}
                    ),
                    "shuffle": GraphNode(
                        "ShuffleDataset",
                        {"images": Link("source", "batches"), "seed": Link("seed", "item")},
                    ),
                    "pair_shuffle": GraphNode(
                        "ShuffleImageTextDataset",
                        {
                            "images": Link("source", "paired_images"),
                            "texts": Link("source", "texts"),
                            "seed": Link("seed", "item"),
                        },
                    ),
                    "deduplicate": GraphNode(
                        "ImageDeduplication",
                        {"images": Link("source", "batches"), "similarity_threshold": 0.95},
                    ),
                    "grid": GraphNode(
                        "ImageGrid",
                        {
                            "images": Link("source", "grid_images"),
                            "columns": 2,
                            "cell_width": 32,
                            "cell_height": 32,
                            "padding": 1,
                        },
                    ),
                    "pairs": RegionNode(
                        kind="map",
                        binding="zip",
                        body=Graph(
                            nodes={
                                "observe": GraphNode(
                                    "collection-fixture.pair",
                                    {
                                        "image": Link(PORTS_NODE_ID, "image"),
                                        "text": Link(PORTS_NODE_ID, "text"),
                                    },
                                )
                            }
                        ),
                        ports={"image": _IMAGE, "text": _STRING},
                        inputs={
                            "image": Link("pair_shuffle", "images"),
                            "text": Link("pair_shuffle", "texts"),
                        },
                        element_ports=("image", "text"),
                        outputs={"pairs": RegionOutput(Link("observe", "pair"))},
                    ),
                    "observe": GraphNode(
                        "collection-fixture.observe",
                        {
                            "shuffled": Link("shuffle", "images"),
                            "deduplicated": Link("deduplicate", "images"),
                            "grid": Link("grid", "images"),
                            "pairs": Link("pairs", "pairs"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["observe"])
                assert result.outputs["observe"]["valid"].resolve() is True
        finally:
            await composer.close()

    asyncio.run(scenario())
