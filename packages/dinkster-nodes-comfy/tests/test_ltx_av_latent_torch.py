"""LTXV AV latent role, mask, and audio-fitting behavior."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_api.v1 import (
    InputSpec,
    MultiStreamLatent,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.ltx_av_latent import LTXVConcatAVLatent, LTXVSeparateAVLatent

from dinkster.compose import PackSpec, ServingComposer


def _latent(samples: object, **metadata: object) -> dict[str, object]:
    return {"samples": samples, **metadata}


def test_concat_and_separate_preserve_explicit_roles_metadata_and_values(monkeypatch) -> None:
    monkeypatch.setattr(
        "dinkster_nodes_comfy.ltx_av_latent.initialize_inference_process", lambda: None
    )
    video = torch.arange(12, dtype=torch.float64).reshape(1, 2, 2, 3)
    audio = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    video_mask = torch.zeros_like(video)
    output = LTXVConcatAVLatent.execute(
        video_latent=_latent(video, noise_mask=video_mask, owner="video", shared="video"),
        audio_latent=_latent(audio, owner="audio", shared="audio"),
    )["LATENT"]
    streams = output["samples"]
    assert type(streams) is MultiStreamLatent
    assert streams.roles == ("video", "audio")
    assert streams.by_role("video") is video and streams.by_role("audio") is audio
    assert output["owner"] == "audio" and output["shared"] == "audio"
    masks = output["noise_mask"]
    assert masks.roles == ("video", "audio")
    assert torch.equal(masks.by_role("video"), video_mask)
    assert torch.equal(masks.by_role("audio"), torch.ones_like(audio))

    split = LTXVSeparateAVLatent.execute(av_latent=output)
    assert split["VIDEO_LATENT"]["samples"] is video
    assert split["AUDIO_LATENT"]["samples"] is audio
    assert split["VIDEO_LATENT"]["noise_mask"] is masks.by_role("video")
    assert split["AUDIO_LATENT"]["noise_mask"] is masks.by_role("audio")


def test_concat_replaces_audio_by_role_and_pads_mask_tail(monkeypatch) -> None:
    monkeypatch.setattr(
        "dinkster_nodes_comfy.ltx_av_latent.initialize_inference_process", lambda: None
    )
    monkeypatch.setattr("dinkster_inference.utils.reshape_mask", lambda mask, _shape: mask)
    video = torch.full((1, 2, 2, 3), 9.0, dtype=torch.float64)
    previous_audio = torch.full((1, 2, 6), 4.0, dtype=torch.float32)
    new_audio = torch.tensor(
        [[[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0]]], dtype=torch.float32
    )
    audio_mask = torch.zeros_like(new_audio)
    av = _latent(MultiStreamLatent.from_pairs((("audio", previous_audio), ("video", video))))
    output = LTXVConcatAVLatent.execute(
        video_latent=av, audio_latent=_latent(new_audio, noise_mask=audio_mask)
    )["LATENT"]
    streams = output["samples"]
    assert streams.roles == ("video", "audio")
    assert streams.by_role("video") is video
    assert torch.equal(
        streams.by_role("audio"),
        torch.tensor([[[7.0, 8.0, 9.0, 10.0, 0.0, 0.0], [11.0, 12.0, 13.0, 14.0, 0.0, 0.0]]]),
    )
    assert torch.equal(
        output["noise_mask"].by_role("audio"),
        torch.tensor([[[0.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]]),
    )


class _AVLatentSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-av-fixture.source",
            outputs=(
                OutputSpec("video", TypeExpr.concrete("LATENT")),
                OutputSpec("audio", TypeExpr.concrete("LATENT")),
            ),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(
            video={"samples": torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)},
            audio={"samples": torch.arange(8, dtype=torch.float32).reshape(1, 2, 4) + 20},
        )


class _AVLatentConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-av-fixture.consumer",
            inputs=(
                InputSpec("video", TypeExpr.concrete("LATENT")),
                InputSpec("audio", TypeExpr.concrete("LATENT")),
            ),
            outputs=(OutputSpec("values", TypeExpr.concrete("STRING")),),
        )

    @classmethod
    def execute(cls, *, video: Mapping[str, Any], audio: Mapping[str, Any]) -> Mapping[str, object]:
        video = cast("Mapping[str, Any]", materialize_latent(video))
        audio = cast("Mapping[str, Any]", materialize_latent(audio))
        return cls.outputs(
            values=(f"{video['samples'].flatten().tolist()}|{audio['samples'].flatten().tolist()}")
        )


@pytest.mark.parametrize("in_process", (True, False))
def test_manifest_round_trips_av_latent_roles_and_values_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        manifest = Path(__file__).parents[1] / "dinkster-pack.toml"
        (tmp_path / "ltx_av_fixture.py").write_text(
            "from test_ltx_av_latent_torch import _AVLatentConsumer, _AVLatentSource\n"
            "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
            "NODES = (*COMFY_NODES, _AVLatentSource, _AVLatentConsumer)\n",
            encoding="ascii",
        )
        fixture_manifest = tmp_path / "dinkster-pack.toml"
        fixture_manifest.write_text(
            manifest.read_text(encoding="ascii")
            .replace("namespaces = [", 'namespaces = ["ltx-av-fixture",')
            .replace("dinkster_nodes_comfy:COMFY_NODES", "ltx_av_fixture:NODES")
            .replace(
                "dinkster_nodes_comfy:register_inference_types",
                "ltx_av_fixture:register_inference_types",
            ),
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), *sys.path))}
        )
        try:
            await composer.add_pack(PackSpec(fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("ltx-av-fixture.source"),
                    "concat": GraphNode(
                        "LTXVConcatAVLatent",
                        {
                            "video_latent": Link("source", "video"),
                            "audio_latent": Link("source", "audio"),
                        },
                    ),
                    "separate": GraphNode(
                        "LTXVSeparateAVLatent", {"av_latent": Link("concat", "LATENT")}
                    ),
                    "consumer": GraphNode(
                        "ltx-av-fixture.consumer",
                        {
                            "video": Link("separate", "VIDEO_LATENT"),
                            "audio": Link("separate", "AUDIO_LATENT"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            expected = (
                "[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]"
                "|[20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0]"
            )
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert cast(str, result.outputs["consumer"]["values"].resolve()) == expected
        finally:
            await composer.close()

    asyncio.run(scenario())
