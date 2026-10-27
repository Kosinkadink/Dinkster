"""Text rendering, straight-alpha composition, and ordinary-host transport."""

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
from dinkster_nodes_comfy.text_overlay import TextOverlay
from PIL import Image, ImageDraw, ImageFont

from dinkster.compose import PackSpec, ServingComposer


def _images(channels: int = 4, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return (
        torch.tensor([[0.1, 0.3, 0.7, 0.25], [0.8, 0.2, 0.4, 0.0]], dtype=dtype)[:, :channels]
        .reshape(2, 1, 1, channels)
        .expand(2, 80, 120, channels)
        .clone()
    )


def _expected(images: torch.Tensor) -> torch.Tensor:
    layer = Image.new("RGBA", (120, 80))
    font = ImageFont.load_default(size=20)
    top = font.getbbox("Ab")[1]
    ImageDraw.Draw(layer).text((1, 1 - top), "Ab", font=font, fill=(128, 64, 32, 128))
    rgba = torch.from_numpy(np.array(layer).astype(np.float32) / 255).to(images.dtype)
    color, alpha = rgba[..., :3], rgba[..., 3:]
    if images.shape[-1] == 3:
        return images * (1 - alpha) + color * alpha
    expected_alpha = alpha + (1 - alpha) * images[..., 3:]
    numerator = color * alpha + (1 - alpha) * images[..., :3] * images[..., 3:]
    expected_rgb = torch.where(
        expected_alpha > 1e-5, numerator / expected_alpha, torch.zeros_like(numerator)
    )
    return torch.cat((expected_rgb.clamp(0, 1), expected_alpha), dim=-1)


@pytest.mark.parametrize(
    "channels,dtype", [(3, torch.float32), (4, torch.float32), (4, torch.float16)]
)
def test_text_overlay_matches_independent_pillow_raster_and_source_over(
    channels: int, dtype: torch.dtype
) -> None:
    images = _images(channels, dtype)
    original = images.clone()
    actual = TextOverlay.execute(
        images=images,
        text="Ab",
        font_size=25.0,
        color="#80402080",
        position="top",
        align="left",
        outline=False,
    )["images"]
    assert torch.equal(actual, _expected(images))
    assert actual.dtype == dtype and actual.device.type == "cpu"
    assert torch.equal(images, original)


def test_text_overlay_empty_text_preserves_input_without_parsing_other_controls() -> None:
    images = _images()
    assert (
        TextOverlay.execute(
            images=images,
            text=" \t\n",
            font_size=25.0,
            color="not a color",
            position="unused",
            align="unused",
            outline=False,
        )["images"]
        is images
    )


def test_text_overlay_escaped_lines_match_newlines_and_invalid_color_fails() -> None:
    images = _images(3)
    kwargs = dict(
        images=images,
        font_size=25.0,
        color="#cc8855",
        position="bottom",
        align="right",
        outline=True,
    )
    escaped = TextOverlay.execute(text="Ab\\ncd\\tef", **kwargs)["images"]
    literal = TextOverlay.execute(text="Ab\ncd\tef", **kwargs)["images"]
    assert torch.equal(escaped, literal)
    assert not torch.equal(escaped, images)
    with pytest.raises(ValueError, match="unknown color"):
        TextOverlay.execute(
            images=images,
            text="Ab",
            font_size=25.0,
            color="not a color",
            position="top",
            align="left",
            outline=False,
        )


class OverlaySource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "overlay-fixture.source", outputs=(OutputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),)
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(images=_images())


class OverlayInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "overlay-fixture.inspect",
            inputs=(InputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, images: Any) -> Any:
        actual = materialize_image(images)
        assert torch.equal(actual, _expected(_images()))
        assert actual.dtype == torch.float32 and actual.device.type == "cpu"
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_text_overlay_crosses_ordinary_manifest_hosts_and_caches(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "overlay_fixture.py").write_text(
            "from test_text_overlay_torch import OverlaySource, OverlayInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (OverlaySource, OverlayInspect)\n",
            encoding="ascii",
        )
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "overlay-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["overlay-fixture"]\n\n[pack.entry]\n'
            'nodes = "overlay_fixture:NODES"\n'
            'types = "overlay_fixture:register_inference_types"\n',
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
            for pack in (native_manifest, manifest):
                await composer.add_pack(PackSpec(pack, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("overlay-fixture.source"),
                    "overlay": GraphNode(
                        "TextOverlay",
                        {
                            "images": Link("source", "images"),
                            "text": "Ab",
                            "font_size": 25.0,
                            "color": "#80402080",
                            "position": "top",
                            "align": "left",
                            "outline": False,
                        },
                    ),
                    "inspect": GraphNode(
                        "overlay-fixture.inspect", {"images": Link("overlay", "images")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            first = await engine.run(graph, ["inspect"])
            assert first.outputs["inspect"]["valid"].resolve() is True
            second = await engine.run(graph, ["inspect"])
            assert second.outputs["inspect"]["valid"].resolve() is True
            assert set(second.cached) == {"source", "overlay", "inspect"}
        finally:
            await composer.close()

    asyncio.run(scenario())
