"""Pinned autoregressive sampler construction through ordinary native hosts."""

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from dinkster_api.v1 import (
    CORE_STRING,
    SAMPLER_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.ar_video import SamplerARVideo

from dinkster.compose import PackSpec, ServingComposer


def _assert_sampler(sampler: Any, frames: int) -> None:
    initialize_inference_process()
    from dinkster_inference.k_diffusion.sampling import sample_ar_video
    from dinkster_inference.samplers import KSAMPLER

    assert type(sampler) is KSAMPLER
    assert sampler.sampler_function is sample_ar_video
    assert sampler.extra_options == {"num_frame_per_block": frames}
    assert sampler.inpaint_options == {}


def test_ar_video_sampler_retains_independent_frame_options() -> None:
    samplers = [SamplerARVideo.execute(num_frame_per_block=n)["SAMPLER"] for n in (1, 3, 64)]
    for sampler, frames in zip(samplers, (1, 3, 64), strict=True):
        _assert_sampler(sampler, frames)
    assert samplers[0] is not samplers[1]


class ARSamplerConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ar-sampler-fixture.consumer",
            inputs=(InputSpec("sampler", TypeExpr.concrete(SAMPLER_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, sampler: object) -> dict[str, object]:
        _assert_sampler(sampler, 3)
        return cls.outputs(summary="ar-video-block3")


@pytest.mark.parametrize("in_process", (True,))
def test_ar_video_sampler_runs_twice_through_ordinary_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(source_root))
                ),
            }
        )
        try:
            native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "ar_sampler_fixture.py").write_text(
                "from test_ar_video_sampler_torch import ARSamplerConsumer\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (ARSamplerConsumer,)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                '[pack]\nname = "ar-sampler-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["ar-sampler-fixture"]\n\n[pack.entry]\n'
                'nodes = "ar_sampler_fixture:NODES"\n'
                'types = "ar_sampler_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "ar-sampler-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "sampler": GraphNode("SamplerARVideo", {"num_frame_per_block": 3}),
                    "consumer": GraphNode(
                        "ar-sampler-fixture.consumer", {"sampler": Link("sampler", "SAMPLER")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "ar-video-block3"
        finally:
            await composer.close()

    asyncio.run(scenario())
