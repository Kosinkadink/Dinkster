"""CPU tensor and hosted execution behavior for ComfyUI compositing nodes."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
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
from dinkster_nodes_comfy.compositing import (
    JoinImageWithAlpha,
    PorterDuffImageComposite,
    SplitImageWithAlpha,
)

from dinkster.compose import PackSpec, ServingComposer

_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)


def test_compositing_materializes_decoded_masks() -> None:
    source = torch.full((1, 2, 3, 3), 0.8)
    destination = torch.full((1, 2, 3, 3), 0.2)
    source_alpha = np.full((1, 2, 3), 0.75, dtype=np.float32)
    destination_alpha = np.full((1, 2, 3), 0.25, dtype=np.float32)
    result = PorterDuffImageComposite.execute(
        source=source,
        source_alpha=source_alpha,
        destination=destination,
        destination_alpha=destination_alpha,
        mode="SRC_OVER",
    )
    torch.testing.assert_close(result["IMAGE"], torch.full((1, 2, 3, 3), 5.0 / 13.0))
    torch.testing.assert_close(result["MASK"], torch.full((1, 2, 3), 3.0 / 16.0))

    split = SplitImageWithAlpha.execute(
        image=torch.cat((source, torch.full((1, 2, 3, 1), 0.25)), -1)
    )
    joined = JoinImageWithAlpha.execute(image=split["IMAGE"], alpha=source_alpha)["IMAGE"]
    torch.testing.assert_close(joined[..., 3], torch.full((1, 2, 3), 0.25))


class CompositingSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="compositing-fixture.source",
            outputs=(
                OutputSpec("source", _IMAGE),
                OutputSpec("source_alpha", _MASK),
                OutputSpec("destination", _IMAGE),
                OutputSpec("destination_alpha", _MASK),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            source=np.full((1, 2, 3, 3), 0.8, dtype=np.float32),
            source_alpha=np.full((1, 2, 3), 0.75, dtype=np.float32),
            destination=np.full((1, 2, 3, 3), 0.2, dtype=np.float32),
            destination_alpha=np.full((1, 2, 3), 0.25, dtype=np.float32),
        )


class CompositingInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="compositing-fixture.inspect",
            inputs=(InputSpec("image", _IMAGE), InputSpec("mask", _MASK)),
            outputs=(OutputSpec("pixel", _FLOAT), OutputSpec("mask", _FLOAT)),
        )

    @classmethod
    def execute(cls, *, image: Any, mask: Any) -> dict[str, object]:
        image, mask = materialize_image(image), materialize_image(mask)
        return cls.outputs(pixel=float(image[0, 0, 0, 0]), mask=float(mask[0, 0, 0]))


@pytest.mark.parametrize("in_process", (True, False))
def test_manifest_links_decoded_masks_to_compositing_nodes(
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
            (tmp_path / "compositing_fixture_sources.py").write_text(
                "from test_compositing_torch import CompositingInspect, CompositingSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_comfy.device_routing import device_choices\n"
                "NODES = (*COMFY_NODES, CompositingSource, CompositingInspect)\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            source_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["compositing-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "compositing_fixture_sources:NODES")
                .replace(
                    "dinkster_nodes_comfy.device_routing:device_choices",
                    "compositing_fixture_sources:device_choices",
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "compositing_fixture_sources:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            result = await composer.composition.make_engine(lambda _event: None).run(
                Graph(
                    nodes={
                        "source": GraphNode("compositing-fixture.source"),
                        "composite": GraphNode(
                            "PorterDuffImageComposite",
                            {
                                "source": Link("source", "source"),
                                "source_alpha": Link("source", "source_alpha"),
                                "destination": Link("source", "destination"),
                                "destination_alpha": Link("source", "destination_alpha"),
                                "mode": "SRC_OVER",
                            },
                        ),
                        "inspect": GraphNode(
                            "compositing-fixture.inspect",
                            {
                                "image": Link("composite", "IMAGE"),
                                "mask": Link("composite", "MASK"),
                            },
                        ),
                    }
                ),
                ["inspect"],
            )
            output = result.outputs["inspect"]
            assert output["pixel"].resolve() == pytest.approx(5.0 / 13.0)
            assert output["mask"].resolve() == pytest.approx(3.0 / 16.0)
        finally:
            await composer.close()

    asyncio.run(scenario())
