"""CPU pixels and ordinary-manifest transport for native Canny edges."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.image_filters import Canny

from dinkster.compose import PackSpec, ServingComposer


def image() -> np.ndarray[Any, Any]:
    pixels = np.zeros((2, 13, 15, 4), dtype=np.float64)
    pixels[0, :, 7:, :3] = 1.0
    pixels[1, 5:, :, :3] = 1.0
    pixels[..., 3] = np.arange(13 * 15).reshape(13, 15) % 2
    return pixels


def expected() -> torch.Tensor:
    pixels = torch.zeros((2, 13, 15, 3), dtype=torch.float32)
    pixels[0, :, 6, :] = 1.0
    pixels[1, 4, :, :] = 1.0
    return pixels


def test_canny_discards_alpha_converts_dtype_and_preserves_source() -> None:
    pixels = torch.from_numpy(image())
    before = pixels.clone()
    output = Canny.execute(image=pixels, low_threshold=0.2, high_threshold=0.6)["IMAGE"]
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert torch.equal(output, expected())
    assert torch.equal(pixels, before)
    constant_rgb = pixels.clone()
    constant_rgb[..., :3] = 0
    assert (
        torch.count_nonzero(
            Canny.execute(image=constant_rgb, low_threshold=0.1, high_threshold=0.9)["IMAGE"]
        )
        == 0
    )


class ImageSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="canny-fixture.image",
            inputs=(),
            outputs=(OutputSpec("IMAGE", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(IMAGE=image())


class Result(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="canny-fixture.result",
            inputs=(InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("pixels", TypeExpr.concrete(CORE_INT)),),
        )

    @classmethod
    def execute(cls, *, image: Any) -> Mapping[str, object]:
        pixels = materialize_image(image)
        assert pixels.dtype == torch.float32
        assert torch.equal(pixels, expected())
        return cls.outputs(pixels=int(torch.count_nonzero(pixels)))


FIXTURE_NODES = (ImageSource, Result)


@pytest.mark.parametrize("in_process", (True,))
def test_canny_executes_twice_with_numpy_input_in_both_manifest_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            (tmp_path / "canny_fixture.py").write_text(
                "from test_canny_torch import FIXTURE_NODES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, *FIXTURE_NODES)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                (Path(__file__).parents[1] / "dinkster-pack.toml")
                .read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["canny-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "canny_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "canny_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode("canny-fixture.image", {}),
                    "edges": GraphNode(
                        "Canny",
                        {
                            "image": Link("source", "IMAGE"),
                            "low_threshold": 0.2,
                            "high_threshold": 0.6,
                        },
                    ),
                    "result": GraphNode("canny-fixture.result", {"image": Link("edges", "IMAGE")}),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["result"])
                assert result.outputs["result"]["pixels"].resolve() == 84
        finally:
            await composer.close()

    asyncio.run(scenario())
