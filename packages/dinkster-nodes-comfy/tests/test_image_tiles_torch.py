"""CPU behavior for native image tiles and mask preview."""

from __future__ import annotations

import asyncio
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_STRING,
    IMAGE_TYPE,
    MASK_TYPE,
    VIDEO_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    assemble_video,
    effective_video_facts,
    list_children,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.image_tiles import ImageMergeTileList, SplitImageToTileList
from dinkster_nodes_comfy.mask_preview import MaskPreview
from dinkster_nodes_comfy.utilities import PreviewAny
from dinkster_schema import use_reporter

from dinkster.compose import PackSpec, ServingComposer, compose_serving


def test_split_and_merge_asymmetric_tiles_preserve_dtype_device_and_pixels() -> None:
    image = torch.arange(5 * 7, dtype=torch.float64).reshape(1, 5, 7, 1)
    tiles = SplitImageToTileList.execute(image=image, tile_width=4, tile_height=3, overlap=1)[
        "IMAGE"
    ]
    assert [tile.shape for tile in tiles] == [(1, 3, 4, 1)] * 4
    assert all(tile.dtype == image.dtype and tile.device == image.device for tile in tiles)
    merged = ImageMergeTileList.execute(image_list=tiles, final_width=7, final_height=5, overlap=1)[
        "IMAGE"
    ]
    assert torch.allclose(merged, image)


def test_merge_stops_at_missing_tiles_and_crops_partial_tiles() -> None:
    tile = torch.ones((1, 2, 2, 1), dtype=torch.float32)
    merged = ImageMergeTileList.execute(
        image_list=[tile[:, :1]], final_width=4, final_height=2, overlap=0
    )["IMAGE"]
    assert torch.equal(merged[0, 0, :, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.equal(merged[0, 1, :, 0], torch.zeros(4))


def test_mask_preview_passes_through_mask_and_reports_png() -> None:
    mask = torch.tensor([[[0.0, 1.0], [0.5, 0.25]]])
    reports: list[tuple[str, dict[str, object], bytes | None]] = []
    with use_reporter(lambda name, data, blob: reports.append((name, dict(data), blob))):
        result = MaskPreview.execute(mask=mask)["mask"]
    assert result is mask
    assert reports[0][:2] == ("preview", {"mime": "image/png", "width": 2, "height": 2})
    assert reports[0][2] is not None and reports[0][2].startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (None, "None"),
        (False, "False"),
        ("first\nsecond", "first\nsecond"),
        ({"n": [2, 7]}, '{\n    "n": [\n        2,\n        7\n    ]\n}'),
    ],
)
def test_preview_any_returns_and_reports_the_formatted_value(source: object, expected: str) -> None:
    reports: list[dict[str, object]] = []
    with use_reporter(lambda _name, data, _blob: reports.append(dict(data))):
        assert PreviewAny.execute(source=source) == {"STRING": expected}
    assert reports == [{"step": 1, "total": 1, "text": expected}]


def test_preview_any_formats_tensor_edges_and_unserializable_objects() -> None:
    tensor = torch.arange(50)
    expected = "tensor([ 0,  1,  2,  3,  4,  5,  ..., 44, 45, 46, 47, 48, 49])"
    torch.set_printoptions(threshold=20)
    try:
        assert PreviewAny.execute(source=tensor) == {"STRING": expected}
    finally:
        torch.set_printoptions(profile="default")

    class BrokenString:
        def __str__(self) -> str:
            raise ValueError("cannot stringify")

    assert PreviewAny.execute(source=BrokenString()) == {
        "STRING": "source exists, but could not be serialized."
    }


_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)
_VIDEO = TypeExpr.concrete(VIDEO_TYPE)
_STRINGS = TypeExpr.list_of(TypeExpr.concrete(CORE_STRING))


class ToolSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "image-tools-fixture.source",
            outputs=(
                OutputSpec("image", _IMAGE),
                OutputSpec("mask", _MASK),
                OutputSpec("video", _VIDEO),
                OutputSpec("videos", TypeExpr.list_of(_VIDEO)),
                OutputSpec("texts", _STRINGS),
                OutputSpec("first", _STRINGS),
                OutputSpec("last", _STRINGS),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(
            image=torch.arange(2 * 5 * 7 * 3, dtype=torch.float64).reshape(2, 5, 7, 3) / 211,
            mask=torch.tensor([[[0.0, 0.5, 1.0], [0.25, 0.75, 0.125]]]),
            video=assemble_video(np.zeros((7, 4, 6, 3), dtype=np.float32), fps=Fraction(5)),
            videos=[
                assemble_video(np.zeros((n, 4, 6, 3), dtype=np.float32), fps=Fraction(5))
                for n in (1, 2, 3)
            ],
            texts=["one", "two", "three"],
            first=["a", "b"],
            last=["c"],
        )


class ToolObserver(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "image-tools-fixture.observe",
            inputs=(
                InputSpec("image", _IMAGE),
                InputSpec("mask", _MASK),
                InputSpec("video", _VIDEO),
                InputSpec("random", _VIDEO),
                InputSpec("videos", TypeExpr.list_of(_VIDEO)),
                InputSpec("paired", TypeExpr.list_of(_VIDEO)),
                InputSpec("texts", _STRINGS),
                InputSpec("text", TypeExpr.concrete(CORE_STRING)),
            ),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        image: Any,
        mask: Any,
        video: Any,
        random: Any,
        videos: Any,
        paired: Any,
        texts: list[str],
        text: str,
    ) -> Any:
        # The IMAGE consumer boundary normalizes floating storage to float32.
        expected = (
            torch.arange(2 * 5 * 7 * 3, dtype=torch.float64).reshape(2, 5, 7, 3) / 211
        ).float()
        image = materialize_image(image)
        assert image.dtype == torch.float32
        assert torch.equal(image, expected), (
            image.dtype,
            image.shape,
            float((image - expected).abs().max()),
        )
        assert torch.equal(
            materialize_image(mask), torch.tensor([[[0.0, 0.5, 1.0], [0.25, 0.75, 0.125]]])
        ), "mask pixels"
        assert effective_video_facts(video)["frame_count"] == 2, effective_video_facts(video)
        assert effective_video_facts(random)["frame_count"] == 3, effective_video_facts(random)
        assert [effective_video_facts(v)["frame_count"] for v in videos] == [3, 2, 1], "shuffle"
        assert [effective_video_facts(v)["frame_count"] for v in paired] == [3, 2, 1], "paired"
        assert texts == ["three", "two", "one"], texts
        assert text == "c", text
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_registered_image_dataset_and_list_tools_execute_twice_in_ordinary_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(Path.cwd() / "src")))
            }
        )
        manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "image_tools_consumer_entry.py").write_text(
            "from test_image_tiles_torch import ToolSource, ToolObserver\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (ToolSource, ToolObserver)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="image-tools-fixture"\npublisher="dinkster"\n'
            'namespaces=["image-tools-fixture"]\n[pack.entry]\n'
            'nodes="image_tools_consumer_entry:NODES"\n'
            'types="image_tools_consumer_entry:register_inference_types"\n',
            encoding="ascii",
        )
        try:
            await composer.add_pack(
                PackSpec(
                    manifest=manifest.parent.parent
                    / "dinkster-nodes-foundation/dinkster-pack.toml",
                    in_process=in_process,
                    trust_reserved=True,
                )
            )
            await composer.add_pack(
                PackSpec(manifest=manifest, in_process=in_process, trust_reserved=True)
            )
            await composer.add_pack(PackSpec(manifest=fixture, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("image-tools-fixture.source"),
                    "split": GraphNode(
                        "SplitImageToTileList",
                        {
                            "image": Link("source", "image"),
                            "tile_width": 4,
                            "tile_height": 3,
                            "overlap": 0,
                        },
                    ),
                    "merge": GraphNode(
                        "ImageMergeTileList",
                        {
                            "image_list": Link("split", "IMAGE"),
                            "final_width": 7,
                            "final_height": 5,
                            "overlap": 0,
                        },
                    ),
                    "mask": GraphNode("MaskPreview", {"mask": Link("source", "mask")}),
                    "crop": GraphNode(
                        "VideoTemporalCrop",
                        {"video": Link("source", "video"), "start_frame": 5, "length": 99},
                    ),
                    "random": GraphNode(
                        "VideoRandomTemporalCrop",
                        {"video": Link("source", "video"), "length": 3, "seed": 19},
                    ),
                    "shuffle": GraphNode(
                        "ShuffleVideoDataset", {"videos": Link("source", "videos"), "seed": 0}
                    ),
                    "paired": GraphNode(
                        "ShuffleVideoTextDataset",
                        {
                            "videos": Link("source", "videos"),
                            "texts": Link("source", "texts"),
                            "seed": 0,
                        },
                    ),
                    "list": GraphNode(
                        "CreateList",
                        {
                            "inputs.input0": Link("source", "first"),
                            "inputs.input1": Link("source", "last"),
                        },
                    ),
                    "item": GraphNode(
                        "GetItemFromList", {"list": Link("list", "list"), "index": -1}
                    ),
                    "preview": GraphNode("PreviewAny", {"source": Link("item", "item")}),
                    "observe": GraphNode(
                        "image-tools-fixture.observe",
                        {
                            "image": Link("merge", "IMAGE"),
                            "mask": Link("mask", "mask"),
                            "video": Link("crop", "video"),
                            "random": Link("random", "video"),
                            "videos": Link("shuffle", "videos"),
                            "paired": Link("paired", "videos"),
                            "texts": Link("paired", "texts"),
                            "text": Link("preview", "STRING"),
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


class MixedListSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "mixed-list-fixture.source",
            outputs=(
                OutputSpec("texts", _STRINGS),
                OutputSpec("numbers", TypeExpr.list_of(TypeExpr.concrete("INT"))),
                OutputSpec("nested", TypeExpr.list_of(TypeExpr.list_of(TypeExpr.concrete("INT")))),
                OutputSpec("empty", _STRINGS),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(texts=["seven", "eleven"], numbers=[29], nested=[[3, 17]], empty=[])


@pytest.mark.parametrize("in_process", (True, False))
def test_native_mixed_lists_cross_hosts_and_persistent_cache(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "mixed_list_entry.py").write_text(
            "from test_image_tiles_torch import MixedListSource\nNODES = (MixedListSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="mixed-list-fixture"\npublisher="dinkster"\n'
            'namespaces=["mixed-list-fixture"]\n[pack.entry]\n'
            'nodes="mixed_list_entry:NODES"\n',
            encoding="ascii",
        )
        packs = [
            PackSpec(manifest, in_process=in_process, trust_reserved=True),
            PackSpec(fixture, in_process=in_process),
        ]
        graph = Graph(
            nodes={
                "source": GraphNode("mixed-list-fixture.source"),
                "mixed": GraphNode(
                    "CreateList",
                    {
                        "inputs.input2": Link("source", "texts"),
                        "inputs.input0": Link("source", "empty"),
                        "inputs.input3": Link("source", "nested"),
                        "inputs.input1": Link("source", "numbers"),
                    },
                ),
                "number": GraphNode(
                    "GetItemFromList", {"list": Link("mixed", "list"), "index": -1}
                ),
                "nested": GraphNode("GetItemFromList", {"list": Link("mixed", "list"), "index": 2}),
            }
        )
        fingerprints: dict[str, str] = {}
        for run in range(2):
            composition = await compose_serving(
                packs,
                include_default_packs=False,
                worker_env={
                    "PYTHONPATH": os.pathsep.join(
                        (str(Path(__file__).parent), str(Path.cwd() / "src"))
                    )
                },
                cache_mode="layered",
                cache_dir=tmp_path / "cache",
            )
            try:
                engine = composition.make_engine(lambda _event: None)
                result = await engine.run(graph, ["mixed", "number", "nested"])
                mixed = result.outputs["mixed"]["list"]
                assert mixed.type_id == "list<*>"
                assert mixed.resolve() == ["seven", "eleven", [3, 17], 29]
                children = list_children(mixed)
                assert children is not None
                assert [child.type_id for child in children] == [
                    "STRING",
                    "STRING",
                    "list<INT>",
                    "INT",
                ]
                expected = {"number": ("INT", 29), "nested": ("list<INT>", [3, 17])}
                for name, (type_id, payload) in expected.items():
                    selected = result.outputs[name]["item"]
                    assert selected.type_id == type_id
                    assert selected.resolve() == payload
                    if run:
                        assert selected.fingerprint == fingerprints[name]
                    fingerprints[name] = selected.fingerprint
                if run:
                    assert result.executed == ()
                    assert set(result.cached) == set(graph.nodes)
            finally:
                await composition.close()

    asyncio.run(scenario())
