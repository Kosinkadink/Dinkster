"""Public CPU behavior for native image crop primitives."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.image_crop import ImageCropV2, PrimitiveBoundingBox
from dinkster_schema import use_reporter

from dinkster.compose import PackSpec, ServingComposer


def _image() -> torch.Tensor:
    return torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5, 1) / 30


def test_image_crop_v2_uses_asymmetric_bounds_and_reports_the_cropped_preview() -> None:
    image = _image()
    reports: list[tuple[str, dict[str, object], bytes | None]] = []

    with use_reporter(lambda name, data, blob: reports.append((name, dict(data), blob))):
        result = ImageCropV2.execute(
            image=image, crop_region={"x": 3, "y": 1, "width": 2, "height": 1}
        )

    assert torch.equal(result["IMAGE"], image[:, 1:2, 3:5, :])
    assert len(reports) == 1
    name, data, blob = reports[0]
    assert (name, data) == ("preview", {"mime": "image/png", "width": 2, "height": 1})
    assert blob is not None and blob.startswith(b"\x89PNG\r\n\x1a\n")


def test_image_crop_v2_uses_region_defaults_and_clamps_only_the_origin() -> None:
    image = _image()

    defaults = ImageCropV2.execute(image=image, crop_region={})["IMAGE"]
    clamped = ImageCropV2.execute(
        image=image, crop_region={"x": 99, "y": 99, "width": 10, "height": 10}
    )["IMAGE"]

    assert torch.equal(defaults, image)
    assert torch.equal(clamped, image[:, 2:3, 4:5, :])


def test_primitive_bounding_box_returns_input_values_without_normalization() -> None:
    assert PrimitiveBoundingBox.execute(x=3, y=1, width=2, height=1) == {
        "BOUNDING_BOX": {"x": 3, "y": 1, "width": 2, "height": 1}
    }


class CropSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "crop-fixture.source", outputs=(OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),)
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(image=_image().numpy())


class CropInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "crop-fixture.inspect",
            inputs=(InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, image: Any) -> Any:
        actual = materialize_image(image)
        assert actual.dtype == torch.float32
        assert actual.device.type == "cpu"
        assert torch.equal(actual, _image()[:, 1:2, 3:5, :])
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_crop_and_bounding_box_execute_twice_through_ordinary_manifest(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        fixture_module = "crop_ordinary_host_fixture"
        (tmp_path / f"{fixture_module}.py").write_text(
            "from test_image_crop_torch import CropSource, CropInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (CropSource, CropInspect)\n",
            encoding="ascii",
        )
        fixture_manifest = tmp_path / "dinkster-pack.toml"
        fixture_manifest.write_text(
            '[pack]\nname = "crop-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["crop-fixture"]\n\n[pack.entry]\n'
            f'nodes = "{fixture_module}:NODES"\n'
            f'types = "{fixture_module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(Path.cwd() / "src"))
                ),
            }
        )
        try:
            for manifest in (native_manifest, fixture_manifest):
                await composer.add_pack(PackSpec(manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("crop-fixture.source"),
                    "box": GraphNode(
                        "PrimitiveBoundingBox", {"x": 3, "y": 1, "width": 2, "height": 1}
                    ),
                    "crop": GraphNode(
                        "ImageCropV2",
                        {
                            "image": Link("source", "image"),
                            "crop_region": Link("box", "BOUNDING_BOX"),
                        },
                    ),
                    "inspect": GraphNode("crop-fixture.inspect", {"image": Link("crop", "IMAGE")}),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["valid"].resolve() is True
        finally:
            await composer.close()

    asyncio.run(scenario())
