"""First-frame histogram contracts through direct and ordinary native hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    HISTOGRAM_TYPE,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.image_histogram import ImageHistogram

from dinkster.compose import PackSpec, ServingComposer


def source_image() -> torch.Tensor:
    pixels = torch.tensor(
        [
            [-1.0, 0.0, 2.0, 0.125],
            [1.0, 0.0, 0.0, 0.25],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.75],
            [0.5, 0.5, 0.5, 1.0],
            [2.0, -1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.375],
        ]
    ).reshape(1, 1, 7, 4)
    return torch.cat((pixels, torch.ones_like(pixels)), dim=0)


def expected_histograms() -> dict[str, list[int]]:
    # Hand-derived from the seven first-frame pixels above. Luminance bins use
    # raw RGB before clipping; the RGB composite floors each bin independently.
    counts = {
        "rgb": {0: 3, 127: 1, 255: 2},
        "luminance": {0: 2, 18: 1, 54: 1, 127: 1, 182: 1, 255: 1},
        "red": {0: 3, 127: 1, 255: 3},
        "green": {0: 4, 127: 1, 255: 2},
        "blue": {0: 3, 127: 1, 255: 3},
    }
    return {name: [bins.get(index, 0) for index in range(256)] for name, bins in counts.items()}


@pytest.mark.parametrize("channels", (3, 4))
def test_histogram_uses_first_frame_and_raw_luminance_without_mutation(channels: int) -> None:
    source = source_image()[..., :channels]
    before = source.clone()
    result = ImageHistogram.execute(image=source)
    assert result == expected_histograms()
    assert tuple(result) == ("rgb", "luminance", "red", "green", "blue")
    torch.testing.assert_close(source, before, rtol=0, atol=0)


class HistogramSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="histogram-fixture.source",
            outputs=(OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(image=source_image())


class HistogramConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="histogram-fixture.consumer",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(HISTOGRAM_TYPE))
                for name in ("rgb", "luminance", "red", "green", "blue")
            )
            + (InputSpec("original", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, original: object, **histograms: object) -> dict[str, object]:
        assert histograms == expected_histograms()
        assert all(
            type(value) is list and all(type(count) is int for count in value)
            for value in histograms.values()
        )
        torch.testing.assert_close(materialize_image(original), source_image(), rtol=0, atol=0)
        return cls.outputs(summary="histogram")


@pytest.mark.parametrize("in_process", (True, False))
def test_histogram_outputs_execute_and_cache_in_ordinary_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "histogram_fixture.py").write_text(
            "from test_image_histogram_torch import HistogramSource, HistogramConsumer\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (HistogramSource, HistogramConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname = "histogram-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["histogram-fixture"]\n[pack.entry]\n'
            'nodes = "histogram_fixture:NODES"\n'
            'types = "histogram_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "DINKSTER_ACCELERATOR": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(native.parent / "src"))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest, worker_group="histogram", group_manifests=(native, fixture)
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("histogram-fixture.source"),
                    "histogram": GraphNode("ImageHistogram", {"image": Link("source", "image")}),
                    "consumer": GraphNode(
                        "histogram-fixture.consumer",
                        {
                            **{name: Link("histogram", name) for name in expected_histograms()},
                            "original": Link("source", "image"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for run in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "histogram"
                if run == 1:
                    assert set(result.cached) == {"source", "histogram", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
