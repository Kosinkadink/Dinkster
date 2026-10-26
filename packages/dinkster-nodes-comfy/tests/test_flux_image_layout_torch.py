"""CPU tensor and ordinary-pack execution behavior for Flux image layout nodes."""

from __future__ import annotations

import asyncio
import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    ARRAY_TYPE,
    CORE_FLOAT,
    CORE_STRING,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.flux_image_layout import Flux3LayoutPrompt, FluxKontextImageScale
from PIL import Image

from dinkster.compose import PackSpec, ServingComposer

_ARRAY = TypeExpr.concrete(ARRAY_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)


@pytest.mark.parametrize(
    ("height", "width", "target_height", "target_width", "crop_left", "crop_right"),
    ((11, 17, 832, 1248, 0, 17), (50, 83, 800, 1328, 0, 83), (8, 32, 672, 1568, 7, 25)),
)
def test_flux_image_scale_preserves_preferred_geometry_and_lanczos_pixels(
    height: int, width: int, target_height: int, target_width: int, crop_left: int, crop_right: int
) -> None:
    image = torch.linspace(-0.2, 1.2, height * width * 3).reshape(1, height, width, 3)
    source = image.clone()
    scaled: Any = FluxKontextImageScale.execute(image=image.numpy())["IMAGE"]
    assert torch.equal(image, source)
    assert scaled.shape == (1, target_height, target_width, 3)
    pixels = np.clip(255 * source[0, :, crop_left:crop_right].numpy(), 0, 255).astype(np.uint8)
    expected = (
        np.asarray(
            Image.fromarray(pixels).resize((target_width, target_height), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        / 255
    )
    torch.testing.assert_close(scaled[0], torch.from_numpy(expected), rtol=0, atol=0)


def test_flux_layout_preserves_order_quoting_and_original_prompt() -> None:
    prompt = "A cafe  \n"
    elements = [
        {"bbox": [1, 2, 3, 4], "desc": "sign", "type": "text", "text": "Cafe"},
        {"bbox": [5, 6, 7, 8], "desc": "table", "type": "image"},
    ]
    original = copy.deepcopy(elements)
    result = Flux3LayoutPrompt.execute(prompt=prompt, element=elements)["prompt"]
    assert result == (
        'A cafe [{"id": "obj_1", "bbox": [1, 2, 3, 4], "desc": "sign \\"Cafe\\""}, '
        '{"id": "obj_2", "bbox": [5, 6, 7, 8], "desc": "table"}]'
    )
    assert elements == original
    assert Flux3LayoutPrompt.execute(prompt=prompt, element={})["prompt"] == prompt
    assert Flux3LayoutPrompt.execute(prompt=prompt, element=[])["prompt"] == prompt


class FluxSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="flux-image-layout-fixture.source",
            outputs=(
                OutputSpec("image", _IMAGE),
                OutputSpec("prompt", _STRING),
                OutputSpec("elements", _ARRAY),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            image=np.arange(11 * 17 * 3, dtype=np.float32).reshape(1, 11, 17, 3) / 511,
            prompt="A sign ",
            elements=[{"bbox": [1, 2, 3, 4], "desc": "shop", "type": "text", "text": "Caf\u00e9"}],
        )


class FluxInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="flux-image-layout-fixture.inspect",
            inputs=(InputSpec("image", _IMAGE), InputSpec("prompt", _STRING)),
            outputs=(OutputSpec("height", _FLOAT), OutputSpec("prompt", _STRING)),
        )

    @classmethod
    def execute(cls, *, image: Any, prompt: str) -> dict[str, object]:
        return cls.outputs(height=float(materialize_image(image).shape[1]), prompt=prompt)


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_flux_image_layout_nodes(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        str(source_root),
                        os.environ.get("PYTHONPATH", ""),
                    )
                ),
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "flux_image_layout_fixture.py").write_text(
                "from test_flux_image_layout_torch import FluxInspect, FluxSource\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (FluxSource, FluxInspect)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                '[pack]\nname = "flux-image-layout-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["flux-image-layout-fixture"]\n\n[pack.entry]\n'
                'nodes = "flux_image_layout_fixture:NODES"\n'
                'types = "flux_image_layout_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            for path in (manifest, source_manifest):
                await composer.add_pack(
                    PackSpec(
                        manifest=path,
                        in_process=in_process,
                        worker_group=None if in_process else "flux-image-layout-fixture",
                        group_manifests=() if in_process else (manifest, source_manifest),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("flux-image-layout-fixture.source"),
                    "scale": GraphNode("FluxKontextImageScale", {"image": Link("source", "image")}),
                    "layout": GraphNode(
                        "Flux3LayoutPrompt",
                        {
                            "prompt": Link("source", "prompt"),
                            "element": Link("source", "elements"),
                        },
                    ),
                    "inspect": GraphNode(
                        "flux-image-layout-fixture.inspect",
                        {"image": Link("scale", "IMAGE"), "prompt": Link("layout", "prompt")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                output = result.outputs["inspect"]
                assert output["height"].resolve() == 832.0
                assert output["prompt"].resolve().endswith('"shop \\"Caf\u00e9\\""}]')
        finally:
            await composer.close()

    asyncio.run(scenario())
