"""CPU tensor and ordinary-pack execution behavior for Flux image layout nodes."""

from __future__ import annotations

import asyncio
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

from dinkster.compose import PackSpec, ServingComposer

_ARRAY = TypeExpr.concrete(ARRAY_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)


def test_flux_image_layout_preserves_resolution_tiebreak_and_prompt_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_nodes_comfy.flux_image_layout as module

    monkeypatch.setattr(module, "initialize_inference_process", lambda: None)
    image = torch.arange(1 * 11 * 17 * 3, dtype=torch.float32).reshape(1, 11, 17, 3) / 511
    source = image.clone()
    scaled: Any = FluxKontextImageScale.execute(image=image.numpy())["IMAGE"]
    assert torch.equal(image, source)
    assert scaled.shape == (1, 832, 1248, 3)
    prompt = "A cafe  \n"
    elements = [
        {"bbox": [1, 2, 3, 4], "desc": "sign", "type": "text", "text": "Cafe"},
        {"bbox": [5, 6, 7, 8], "desc": "table", "type": "image"},
    ]
    result = Flux3LayoutPrompt.execute(prompt=prompt, element=elements)["prompt"]
    assert result == (
        'A cafe [{"id": "obj_1", "bbox": [1, 2, 3, 4], "desc": "sign \\"Cafe\\""}, '
        '{"id": "obj_2", "bbox": [5, 6, 7, 8], "desc": "table"}]'
    )
    assert Flux3LayoutPrompt.execute(prompt=prompt, element={})["prompt"] == prompt


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
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(source_root)))
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "flux_image_layout_fixture.py").write_text(
                "from test_flux_image_layout_torch import FluxInspect, FluxSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, FluxSource, FluxInspect)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["flux-image-layout-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "flux_image_layout_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "flux_image_layout_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            result = await composer.composition.make_engine(lambda _event: None).run(
                Graph(
                    nodes={
                        "source": GraphNode("flux-image-layout-fixture.source"),
                        "scale": GraphNode(
                            "FluxKontextImageScale", {"image": Link("source", "image")}
                        ),
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
                ),
                ["inspect"],
            )
            output = result.outputs["inspect"]
            assert output["height"].resolve() == 832.0
            assert output["prompt"].resolve().endswith('"shop \\"Caf\u00e9\\""}]')
        finally:
            await composer.close()

    asyncio.run(scenario())
