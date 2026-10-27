"""Public text rendering, compositing and ordinary native-host coverage."""

from __future__ import annotations

import asyncio
import os
import sys
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
from dinkster_nodes_comfy.text_overlay import TextOverlay

from dinkster.compose import PackSpec, ServingComposer


def _image(channels: int = 4) -> Any:
    image = torch.empty((2, 96, 160, channels), dtype=torch.float64)
    image[0] = torch.tensor([0.2, 0.4, 0.7, 0.25][:channels])
    image[1] = torch.tensor([0.8, 0.3, 0.1, 0.75][:channels])
    return image


def _render(images: Any, **options: Any) -> Any:
    return TextOverlay.execute(
        images=images,
        text=options.get("text", "MMMM"),
        font_size=options.get("font_size", 40),
        color=options.get("color", "#ff000080"),
        position=options.get("position", "top"),
        align=options.get("align", "left"),
        outline=options.get("outline", False),
    )["images"]


@pytest.mark.parametrize("channels", (3, 4))
def test_text_overlay_translucent_color_composites_each_frame_without_source_mutation(
    channels: int,
) -> None:
    images = _image(channels)
    before = images.clone()
    result = _render(images)
    alpha = float(torch.tensor(128 / 255, dtype=torch.float32))
    expected = before.clone()
    if channels == 3:
        expected[..., :3] = before[..., :3] * (1 - alpha)
        expected[..., 0] += alpha
    else:
        expected[..., 3] = alpha + (1 - alpha) * before[..., 3]
        expected[..., :3] = before[..., :3] * before[..., 3:] * (1 - alpha)
        expected[..., 0] += alpha
        expected[..., :3] /= expected[..., 3:]
    # Interior pixels have full glyph coverage, unlike antialiased edges.
    full = (result[0] - expected[0]).abs().amax(dim=-1) < 1e-12
    assert full.sum() > 100
    torch.testing.assert_close(result[:, full], expected[:, full], rtol=0, atol=1e-12)
    assert result.dtype == images.dtype and result.device == images.device
    assert torch.equal(images, before)
    assert torch.equal(result[:, -1, -1], before[:, -1, -1])


def test_text_overlay_empty_identity_and_transparent_destination() -> None:
    images = _image()
    assert _render(images, text=" \n\t ") is images
    transparent = images.clone()
    transparent[..., 3] = 0
    result = _render(transparent)
    assert torch.equal(result[:, -1, -1], torch.zeros((2, 4), dtype=images.dtype))
    alpha = float(torch.tensor(128 / 255, dtype=torch.float32))
    assert result[..., 3].max().item() == alpha
    full = result[0, ..., 3] == alpha
    assert full.sum() > 100
    assert torch.equal(
        result[:, full, :3], torch.tensor([1.0, 0.0, 0.0]).double().expand(2, int(full.sum()), 3)
    )


def test_text_overlay_alignment_escapes_wrapping_and_outlines() -> None:
    images = torch.zeros((1, 128, 240, 3))
    bounds = []
    for align, position in (("left", "top"), ("center", "top"), ("right", "bottom")):
        result = _render(
            images, text="Hi", color="#ffffff", align=align, position=position, font_size=25
        )
        ys, xs = result[0].amax(dim=-1).nonzero(as_tuple=True)
        bounds.append((xs.min().item(), xs.max().item(), ys.min().item(), ys.max().item()))
    assert bounds[0][0] < bounds[1][0] < bounds[2][0]
    assert bounds[0][3] < bounds[2][2]
    escaped = _render(images, text="A\\nB\\tC", color="#ffffff")
    literal = _render(images, text="A\nB\tC", color="#ffffff")
    assert torch.equal(escaped, literal)
    narrow = torch.zeros((1, 64, 31, 3))
    wrapped = _render(narrow, text="abcdefghijklmno\n\nlong words wrap", font_size=50)
    assert wrapped.shape == narrow.shape and torch.isfinite(wrapped).all()
    white = torch.ones_like(images)
    outlined = _render(white, text="Hi", color="#ffffff", outline=True)
    assert (outlined < 1).any()
    assert torch.equal(_render(white, text="Hi", color="#ffffff", outline=False), white)


class OverlaySource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "overlay-fixture.source", outputs=(OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),)
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(image=_image().numpy())


class OverlayConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "overlay-fixture.consume",
            inputs=(InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, image: Any) -> dict[str, object]:
        image = materialize_image(image)
        torch.testing.assert_close(image, _render(_image().float()), rtol=0, atol=0)
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_text_overlay_ordinary_native_hosts_consume_image_and_cache(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        checkout = package.parents[1]
        native = package / "dinkster-pack.toml"
        (tmp_path / "overlay_fixture.py").write_text(
            "from test_text_overlay_torch import OverlaySource, OverlayConsumer\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (OverlaySource, OverlayConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="overlay-fixture"\npublisher="dinkster"\n'
            'namespaces=["overlay-fixture"]\n[pack.entry]\nnodes="overlay_fixture:NODES"\n'
            'types="overlay_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        paths = (
            tmp_path,
            Path(__file__).parent,
            checkout / "src",
            *sorted(checkout.glob("packages/*/src")),
            *(Path(path) for path in sys.path if path and Path(path).is_dir()),
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(map(str, paths)),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest, worker_group="overlay", group_manifests=(native, fixture)
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("overlay-fixture.source"),
                    "overlay": GraphNode(
                        "TextOverlay",
                        {
                            "images": Link("source", "image"),
                            "text": "MMMM",
                            "font_size": 40.0,
                            "color": "#ff000080",
                            "outline": False,
                        },
                    ),
                    "consume": GraphNode(
                        "overlay-fixture.consume", {"image": Link("overlay", "images")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda event: None)
            for run in range(2):
                result = await engine.run(graph, ["consume"])
                assert result.outputs["consume"]["valid"].resolve() is True
                if run == 1:
                    assert set(result.cached) == set(graph.nodes)
        finally:
            await composer.close()

    asyncio.run(scenario())
