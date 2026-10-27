"""Public CPU behavior for native image crop primitives."""

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
    MASK_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.image_crop import ImageCropToMask, ImageCropV2, PrimitiveBoundingBox
from dinkster_schema import use_reporter
from PIL import Image

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


def _crop_to_mask_images() -> torch.Tensor:
    return (
        torch.tensor([[0.2, 0.4, 0.8, 0.01], [0.8, 0.2, 0.4, 0.99]])
        .reshape(2, 1, 1, 4)
        .expand(2, 8, 12, 4)
        .clone()
    )


def _expected_crop_to_mask() -> torch.Tensor:
    return (
        (torch.tensor([[76, 76, 102], [153, 51, 51]], dtype=torch.float32) / 255)
        .reshape(2, 1, 1, 3)
        .expand(2, 64, 80, 3)
    )


def test_crop_to_mask_composites_soft_mask_broadcasts_resizes_and_discards_alpha() -> None:
    images = _crop_to_mask_images()
    original = images.clone()
    masks = torch.full((1, 3, 5), 0.5)
    result = ImageCropToMask.execute(
        images=images,
        masks=masks,
        width=80,
        height=64,
        pad_factor=1.3,
        grow_mask=0,
        background="#663300",
    )["images"]
    assert torch.equal(result, _expected_crop_to_mask())
    assert result.dtype == torch.float32 and result.device.type == "cpu"
    assert torch.equal(images, original)
    assert torch.equal(masks, torch.full((1, 3, 5), 0.5))


def test_crop_to_mask_polarity_growth_and_outside_margin() -> None:
    image = torch.zeros((1, 8, 12, 3))
    image[:, 2:6, 3:9] = torch.tensor([0.2, 0.6, 1.0])
    mask = torch.zeros((1, 8, 12))
    mask[:, 2:6, 3:9] = 1
    kwargs = dict(
        images=image, width=80, height=64, pad_factor=1.6, grow_mask=1, background="#000000"
    )
    positive = ImageCropToMask.execute(masks=mask, **kwargs)["images"]
    negative = ImageCropToMask.execute(masks=1 - mask, **kwargs)["images"]
    assert torch.equal(positive, negative)
    expected_pixels = np.zeros((10, 12, 3), dtype=np.uint8)
    expected_pixels[3:7, 3:9] = [51, 153, 255]
    expected = Image.fromarray(expected_pixels).resize((80, 64), Image.Resampling.LANCZOS)
    assert torch.equal(
        positive, torch.from_numpy(np.array(expected).astype(np.float32) / 255).unsqueeze(0)
    )
    shrunk = ImageCropToMask.execute(
        images=torch.ones_like(image),
        masks=mask,
        width=80,
        height=64,
        pad_factor=1.6,
        grow_mask=-1,
        background="#000000",
    )["images"]
    expected_pixels = np.zeros((6, 6, 3), dtype=np.uint8)
    expected_pixels[3:5, 2:6] = 255
    expected = Image.fromarray(expected_pixels).resize((80, 64), Image.Resampling.LANCZOS)
    assert torch.equal(
        shrunk, torch.from_numpy(np.array(expected).astype(np.float32) / 255).unsqueeze(0)
    )


def test_crop_to_mask_rejects_nonbroadcast_batch_and_malformed_six_digit_color() -> None:
    kwargs = dict(images=_crop_to_mask_images(), width=80, height=64, pad_factor=1.0, grow_mask=0)
    with pytest.raises(ValueError, match="Mask batch 3 does not match image batch 2"):
        ImageCropToMask.execute(masks=torch.ones(3, 8, 12), background="#000000", **kwargs)
    with pytest.raises(ValueError, match="invalid literal"):
        ImageCropToMask.execute(masks=torch.ones(1, 8, 12), background="#gg0000", **kwargs)


class MaskCropSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "mask-crop-fixture.source",
            outputs=(
                OutputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("masks", TypeExpr.concrete(MASK_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(images=_crop_to_mask_images(), masks=torch.full((1, 3, 5), 0.5))


class MaskCropInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "mask-crop-fixture.inspect",
            inputs=(InputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, images: Any) -> Any:
        actual = materialize_image(images)
        assert torch.equal(actual, _expected_crop_to_mask())
        assert actual.dtype == torch.float32 and actual.device.type == "cpu"
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_crop_to_mask_crosses_ordinary_manifest_hosts_and_caches(
    tmp_path: Path, in_process: bool
) -> None:

    async def scenario() -> None:
        native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "mask_crop_fixture.py").write_text(
            "from test_image_crop_torch import MaskCropSource, MaskCropInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (MaskCropSource, MaskCropInspect)\n",
            encoding="ascii",
        )
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "mask-crop-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["mask-crop-fixture"]\n\n[pack.entry]\n'
            'nodes = "mask_crop_fixture:NODES"\n'
            'types = "mask_crop_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        str(native_manifest.parent / "src"),
                        str(Path.cwd() / "src"),
                    )
                ),
            }
        )
        try:
            for pack in (manifest, native_manifest):
                await composer.add_pack(PackSpec(pack, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("mask-crop-fixture.source"),
                    "crop": GraphNode(
                        "ImageCropToMask",
                        {
                            "images": Link("source", "images"),
                            "masks": Link("source", "masks"),
                            "width": 80,
                            "height": 64,
                            "pad_factor": 1.3,
                            "grow_mask": 0,
                            "background": "#663300",
                        },
                    ),
                    "inspect": GraphNode(
                        "mask-crop-fixture.inspect", {"images": Link("crop", "images")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            first = await engine.run(graph, ["inspect"])
            assert first.outputs["inspect"]["valid"].resolve() is True
            second = await engine.run(graph, ["inspect"])
            assert second.outputs["inspect"]["valid"].resolve() is True
            assert set(second.cached) == {"source", "crop", "inspect"}
        finally:
            await composer.close()

    asyncio.run(scenario())
