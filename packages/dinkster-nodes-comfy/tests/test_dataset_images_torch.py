"""CPU behavior for native ComfyUI dataset image transform nodes."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import numpy as np
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
from dinkster_nodes_comfy.dataset_images import (
    AdjustBrightness,
    AdjustContrast,
    CenterCropImages,
    NormalizeImages,
    RandomCropImages,
    ResizeImagesByLongerEdge,
    ResizeImagesByShorterEdge,
)

from dinkster.compose import PackSpec, ServingComposer


def test_spatial_dataset_transforms_quantize_and_process_each_frame() -> None:
    image = torch.linspace(0.0, 1.0, 3 * 5 * 4).reshape(1, 3, 5, 4)
    images = image.repeat((2, 1, 1, 1))
    before = images.clone()

    shorter = ResizeImagesByShorterEdge.execute(images=images, shorter_edge=4)["images"]
    longer = ResizeImagesByLongerEdge.execute(images=images, longer_edge=4)["images"]
    center = CenterCropImages.execute(images=images, width=9, height=2)["images"]
    random = RandomCropImages.execute(images=images, width=3, height=2, seed=511)["images"]

    assert shorter.shape == (2, 4, 6, 4)
    assert longer.shape == (2, 2, 4, 4)
    assert center.shape == (2, 2, 5, 4)
    assert random.shape == (2, 2, 3, 4)
    torch.testing.assert_close(random[0], random[1])
    torch.testing.assert_close(images, before)


def test_tensor_dataset_transforms_preserve_alpha_without_mutating_input() -> None:
    images = torch.tensor([[[[0.2, 0.4, 0.6, 0.3], [0.8, 0.1, 0.5, 0.9]]]], dtype=torch.float32)
    before = images.clone()

    normalized = NormalizeImages.execute(images=images, mean=0.5, std=0.5)["images"]
    brightness = AdjustBrightness.execute(images=images, factor=0.5)["images"]
    contrast = AdjustContrast.execute(images=images, factor=2.0)["images"]

    torch.testing.assert_close(normalized[..., :3], (images[..., :3] - 0.5) / 0.5)
    torch.testing.assert_close(brightness[..., :3], images[..., :3] * 0.5)
    torch.testing.assert_close(
        contrast[..., :3], ((images[..., :3] - 0.5) * 2.0 + 0.5).clamp(0.0, 1.0)
    )
    for result in (normalized, brightness, contrast):
        torch.testing.assert_close(result[..., 3], images[..., 3])
    torch.testing.assert_close(images, before)


def test_center_crop_preserves_quantized_pixels_and_singleton_parameters() -> None:
    image = torch.arange(84, dtype=torch.float32).reshape(1, 4, 7, 3) / 84
    result = CenterCropImages.execute(images=image, width=[3], height=[2])["images"]
    expected = (image[:, 1:3, 2:5] * 255).to(torch.uint8).to(torch.float32) / 255
    assert torch.equal(result, expected)


_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_TRANSFORMS = {
    "ResizeImagesByShorterEdge": {"shorter_edge": 4},
    "ResizeImagesByLongerEdge": {"longer_edge": 4},
    "CenterCropImages": {"width": 3, "height": 2},
    "RandomCropImages": {"width": 9, "height": 8, "seed": 511},
    "NormalizeImages": {"mean": 0.2, "std": 0.4},
    "AdjustBrightness": {"factor": 1.5},
    "AdjustContrast": {"factor": 0.5},
}


class DatasetImageSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema("dataset-image-fixture.source", outputs=(OutputSpec("images", _IMAGE),))

    @classmethod
    def execute(cls) -> Any:
        images = np.empty((2, 3, 5, 3), dtype=np.float32)
        images[0] = [0.2, 0.4, 0.6]
        images[1] = [0.8, 0.1, 0.5]
        return cls.outputs(images=images)


class DatasetImageInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dataset-image-fixture.inspect",
            inputs=tuple(InputSpec(name, _IMAGE) for name in _TRANSFORMS),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, **outputs: Any) -> Any:
        colors = torch.tensor([[0.2, 0.4, 0.6], [0.8, 0.1, 0.5]])
        sizes = [(4, 6), (2, 4), (2, 3), (3, 5), (3, 5), (3, 5), (3, 5)]
        for (name, output), (height, width) in zip(outputs.items(), sizes, strict=True):
            actual = materialize_image(output)
            assert tuple(actual.shape) == (2, height, width, 3)
            if name == "NormalizeImages":
                expected = (colors - 0.2) / 0.4
            elif name == "AdjustBrightness":
                expected = (colors * 1.5).clamp(0, 1)
            elif name == "AdjustContrast":
                expected = ((colors - 0.5) * 0.5 + 0.5).clamp(0, 1)
            else:
                expected = torch.tensor([[51, 102, 153], [204, 25, 127]]) / 255
            torch.testing.assert_close(actual, expected[:, None, None, :].expand_as(actual))
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True,))
def test_dataset_images_ordinary_manifest_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(Path.cwd() / "src")))
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "dataset_image_fixture.py").write_text(
                "from test_dataset_images_torch import DatasetImageSource, DatasetImageInspect\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, DatasetImageSource, DatasetImageInspect)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["dataset-image-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "dataset_image_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "dataset_image_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("dataset-image-fixture.source"),
                    **{
                        name: GraphNode(name, {"images": Link("source", "images"), **parameters})
                        for name, parameters in _TRANSFORMS.items()
                    },
                    "inspect": GraphNode(
                        "dataset-image-fixture.inspect",
                        {name: Link(name, "images") for name in _TRANSFORMS},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["valid"].resolve() is True
        finally:
            await composer.close()

    asyncio.run(scenario())
