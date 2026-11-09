"""Observable crop geometry, padding and empty-input behavior on CPU."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    materialize_image,
    register_box_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.bbox_crop import CropByBBoxes

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize("nested", (False, True))
def test_crop_unions_boxes_broadcasts_last_frame_and_preserves_source(nested: bool) -> None:
    image = torch.arange(3 * 6 * 8 * 4, dtype=torch.float32).reshape(3, 6, 8, 4)
    original = image.clone()
    boxes = [{"x": 1, "y": 2, "width": 2, "height": 2}, {"x": 4, "y": 1, "width": 2, "height": 2}]
    registry = TypeRegistry()
    register_box_types(registry)
    spec = registry.spec(BOUNDING_BOX_TYPE)
    restored = spec.decode(spec.encode([boxes] if nested else boxes))
    output = CropByBBoxes.execute(
        image=image.numpy(), bboxes=restored, output_width=5, output_height=3, padding=0
    )["IMAGE"]
    torch.testing.assert_close(output, original[:, 1:4, 1:6], rtol=0, atol=0)
    torch.testing.assert_close(image, original, rtol=0, atol=0)


def test_crop_pad_keeps_aspect_and_centers_black_border() -> None:
    image = torch.tensor([[[[2.0], [4.0], [6.0], [8.0]], [[12.0], [14.0], [16.0], [18.0]]]])
    output = CropByBBoxes.execute(
        image=image,
        bboxes={"x": 0, "y": 0, "width": 4, "height": 2},
        output_width=4,
        output_height=6,
        padding=0,
        keep_aspect="pad",
    )["IMAGE"]
    expected = torch.zeros(1, 6, 4, 1)
    expected[:, 2:4] = image
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_crop_padding_clamps_and_skips_empty_frames() -> None:
    image = torch.arange(2 * 4 * 5 * 3, dtype=torch.float64).reshape(2, 4, 5, 3)
    output = CropByBBoxes.execute(
        image=image,
        bboxes=[[], [{"x": 0, "y": 1, "width": 2, "height": 1}]],
        output_width=3,
        output_height=3,
        padding=1,
    )["IMAGE"]
    torch.testing.assert_close(output, image[1:2, :3, :3], rtol=0, atol=0)
    assert output.dtype == torch.float64


@pytest.mark.parametrize("boxes", ([], [[]]))
def test_empty_crop_preserves_tensor_identity(boxes: list) -> None:
    image = torch.ones(2, 3, 7, 3)
    assert (
        CropByBBoxes.execute(image=image, bboxes=boxes, output_width=8, output_height=5, padding=0)[
            "IMAGE"
        ]
        is image
    )


def test_degenerate_box_uses_reference_fallback_and_tiny_black_result() -> None:
    box = {"x": 9, "y": 9, "width": 0, "height": 0}
    image = torch.arange(10 * 20, dtype=torch.float32).reshape(1, 10, 20, 1)
    output = CropByBBoxes.execute(
        image=image, bboxes=box, output_width=3, output_height=3, padding=0
    )["IMAGE"]
    torch.testing.assert_close(output, image[:, 1:4, 8:11], rtol=0, atol=0)
    tiny = CropByBBoxes.execute(
        image=torch.ones(1, 2, 2, 4), bboxes=box, output_width=5, output_height=3, padding=0
    )["IMAGE"]
    assert tiny.shape == (1, 3, 5, 4) and torch.count_nonzero(tiny) == 0


class CropSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "bbox-crop-fixture.source",
            outputs=(
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("boxes", TypeExpr.concrete(BOUNDING_BOX_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        image = torch.arange(2 * 6 * 8 * 3, dtype=torch.float32).reshape(2, 6, 8, 3) / 288
        return cls.outputs(
            image=image.numpy(),
            boxes=[
                [
                    {"x": 1, "y": 2, "width": 2, "height": 2},
                    {"x": 4, "y": 1, "width": 2, "height": 2},
                ]
            ],
        )


class CropInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "bbox-crop-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(IMAGE_TYPE))
                for name in ("source", "stretch", "pad")
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, source: Any, stretch: Any, pad: Any) -> dict[str, object]:
        source, stretch, pad = map(materialize_image, (source, stretch, pad))
        expected_source = torch.arange(2 * 6 * 8 * 3, dtype=torch.float32).reshape(2, 6, 8, 3) / 288
        expected_crop = expected_source[:, 1:4, 1:6]
        torch.testing.assert_close(source, expected_source, rtol=0, atol=0)
        torch.testing.assert_close(stretch, expected_crop, rtol=0, atol=0)
        expected_pad = torch.zeros(2, 7, 5, 3)
        expected_pad[:, 2:5] = expected_crop
        torch.testing.assert_close(pad, expected_pad, rtol=0, atol=0)
        return cls.outputs(result="union pixels, broadcast, source and black padding")


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_crops_twice_in_both_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"bbox_crop_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from test_bbox_crop_torch import CropSource, CropInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (CropSource, CropInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname = "bbox-crop-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["bbox-crop-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="bbox-crop-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            inputs = {
                "image": Link("source", "image"),
                "bboxes": Link("source", "boxes"),
                "output_width": 5,
                "output_height": 3,
                "padding": 0,
            }
            graph = Graph(
                nodes={
                    "source": GraphNode("bbox-crop-fixture.source"),
                    "stretch": GraphNode("CropByBBoxes", inputs),
                    "pad": GraphNode(
                        "CropByBBoxes", {**inputs, "output_height": 7, "keep_aspect": "pad"}
                    ),
                    "inspect": GraphNode(
                        "bbox-crop-fixture.inspect",
                        {
                            "source": Link("source", "image"),
                            "stretch": Link("stretch", "IMAGE"),
                            "pad": Link("pad", "IMAGE"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                output = (
                    (await engine.run(graph, ["inspect"])).outputs["inspect"]["result"].resolve()
                )
                assert output == "union pixels, broadcast, source and black padding"
        finally:
            await composer.close()

    asyncio.run(scenario())
