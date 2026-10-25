"""MiniMax H3 AV frame boundaries and ordinary pack execution."""

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    InputSpec,
    MultiStreamLatent,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.minimax_h3_latent import EmptyMiniMaxH3LatentAV

from dinkster.compose import PackSpec, ServingComposer


def _assert_av(latent: Any, video_frames: int, audio_frames: int) -> None:
    assert set(latent) == {"samples"}
    samples = latent["samples"]
    assert type(samples) is MultiStreamLatent
    video = samples.by_role("video")
    audio = samples.by_role("audio")
    assert video.shape == (1, 24, video_frames, 4, 6)
    assert audio.shape == (1, 32, 2, audio_frames)
    for tensor in (video, audio):
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert torch.count_nonzero(tensor).item() == 0


@pytest.mark.parametrize(
    "length,video_frames,audio_frames",
    (
        (4, 2, 8),
        (5, 2, 8),
        (6, 7, 37),
        (21, 7, 37),
        (22, 7, 37),
        (23, 12, 65),
        (38, 12, 65),
        (39, 12, 65),
        (40, 17, 93),
        (124, 37, 207),
    ),
)
def test_empty_av_snaps_both_streams_at_frame_boundaries(
    length: int, video_frames: int, audio_frames: int
) -> None:
    latent = EmptyMiniMaxH3LatentAV.execute(width=96, height=64, length=length)["LATENT"]
    _assert_av(latent, video_frames, audio_frames)


class AVConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="minimax-empty-fixture.consumer",
            inputs=(InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, latent: object) -> dict[str, object]:
        _assert_av(materialize_latent(latent), 12, 65)
        return cls.outputs(summary="video12-audio65-zero")


@pytest.mark.parametrize("in_process", (True, False))
def test_empty_av_runs_twice_through_ordinary_pack_hosts(tmp_path: Path, in_process: bool) -> None:
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
            (tmp_path / "minimax_empty_fixture.py").write_text(
                "from test_minimax_h3_latent_torch import AVConsumer\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (AVConsumer,)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                '[pack]\nname = "minimax-empty-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["minimax-empty-fixture"]\n\n[pack.entry]\n'
                'nodes = "minimax_empty_fixture:NODES"\n'
                'types = "minimax_empty_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "minimax-empty-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "empty": GraphNode(
                        "EmptyMiniMaxH3LatentAV", {"width": 96, "height": 64, "length": 23}
                    ),
                    "consumer": GraphNode(
                        "minimax-empty-fixture.consumer", {"latent": Link("empty", "LATENT")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "video12-audio65-zero"
        finally:
            await composer.close()

    asyncio.run(scenario())
