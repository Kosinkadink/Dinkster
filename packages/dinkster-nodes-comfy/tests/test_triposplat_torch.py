"""CPU TripoSplat foreground geometry and ordinary host transport."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
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
from dinkster_nodes_comfy.triposplat import TripoSplatPreprocessImage

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize("erode_radius", (0, 2))
@pytest.mark.parametrize("mask_size", ((64, 96), (32, 48)))
def test_primary_colors_repeat_resize_masks_and_preserve_inputs(
    erode_radius: int, mask_size: tuple[int, int]
) -> None:
    # Hand-derived: RGB values 2/-1 clamp to primary colors at the center;
    # a centered rectangular alpha region leaves black corners after square padding.
    # The declared stride rounds size 271 down to 256, independently of input size.
    colors = torch.eye(3)
    image = (colors * 3 - 1)[:, None, None, :].expand(3, 64, 96, 3).clone()
    mask = torch.zeros(1, *mask_size)
    height, width = mask_size
    mask[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1
    original_image, original_mask = image.clone(), mask.clone()
    result = TripoSplatPreprocessImage.execute(
        image=image.numpy(), mask=mask.numpy(), erode_radius=erode_radius, size=271
    )["image"]
    assert result.shape == (3, 256, 256, 3) and result.dtype == torch.float32
    torch.testing.assert_close(result[:, 128, 128], colors, rtol=0, atol=0)
    assert torch.count_nonzero(result[:, 0, 0]) == 0
    assert torch.all((result >= 0) & (result <= 1))
    torch.testing.assert_close(image, original_image, rtol=0, atol=0)
    torch.testing.assert_close(mask, original_mask, rtol=0, atol=0)


def test_empty_mask_reports_foreground_error() -> None:
    # Hand-derived: an all-zero mask has no foreground, regardless of RGB values.
    with pytest.raises(ValueError, match="mask is empty .*no foreground pixels"):
        TripoSplatPreprocessImage.execute(
            image=torch.ones(1, 48, 80, 3),
            mask=torch.zeros(1, 48, 80),
            erode_radius=1,
            size=256,
        )


def test_erosion_removes_a_foreground_strip_narrower_than_the_kernel() -> None:
    # Hand-derived: a 5x5 minimum filter removes a three-row strip. Input and
    # requested size are both 256, so the first resize cannot widen the strip.
    image = torch.ones(1, 256, 256, 3)
    mask = torch.zeros(1, 256, 256)
    mask[:, 100:103, 50:200] = 1
    un_eroded = TripoSplatPreprocessImage.execute(image=image, mask=mask, erode_radius=0, size=256)[
        "image"
    ]
    assert torch.count_nonzero(un_eroded) > 0
    with pytest.raises(ValueError, match="mask is empty .*no foreground pixels"):
        TripoSplatPreprocessImage.execute(image=image, mask=mask, erode_radius=2, size=256)


class PreprocessSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "triposplat-fixture.source",
            outputs=(
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("mask", TypeExpr.concrete(MASK_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        image = torch.zeros(1, 64, 96, 3)
        image[..., 0] = 1
        mask = torch.zeros(1, 64, 96)
        mask[:, 16:48, 24:72] = 1
        return cls.outputs(image=image.numpy(), mask=mask.numpy())


class PreprocessInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "triposplat-fixture.inspect",
            inputs=(InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, image: Any) -> Any:
        # Hand-derived from PreprocessSource's uniform red RGB, centered alpha
        # rectangle and requested size 256: red center, black corners, no green/blue.
        image = materialize_image(image)
        assert image.shape == (1, 256, 256, 3)
        torch.testing.assert_close(
            image[0, 128, 128], torch.tensor([1.0, 0.0, 0.0]), rtol=0, atol=0
        )
        assert torch.count_nonzero(image[0, 0, 0]) == 0
        assert torch.count_nonzero(image[..., 1:]) == 0
        return cls.outputs(summary="foreground crop transported")


@pytest.mark.parametrize("in_process", (True, False))
def test_native_manifest_preprocesses_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "triposplat_fixture.py").write_text(
            "from test_triposplat_torch import PreprocessSource, PreprocessInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (PreprocessSource, PreprocessInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="triposplat-fixture"\npublisher="dinkster"\n'
            'namespaces=["triposplat-fixture"]\n[pack.entry]\n'
            'nodes="triposplat_fixture:NODES"\ntypes="triposplat_fixture:register_inference_types"\n',
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
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "triposplat-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("triposplat-fixture.source"),
                    "preprocess": GraphNode(
                        "TripoSplatPreprocessImage",
                        {
                            "image": Link("source", "image"),
                            "mask": Link("source", "mask"),
                            "size": 256,
                        },
                    ),
                    "inspect": GraphNode(
                        "triposplat-fixture.inspect", {"image": Link("preprocess", "image")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve() == "foreground crop transported"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
