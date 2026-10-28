"""Audio latent duration boundaries and ordinary manifest execution on CPU."""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import (
    CORE_INT,
    LATENT_TYPE,
    InputSpec,
    MultiStreamLatent,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_engine import ExecutionError
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.audio_latent import (
    EmptyAceStep15LatentAudio,
    EmptyAceStepLatentAudio,
    EmptyLatentAudio,
    EmptyYuE2LatentAudio,
    GetAudioLatentSize,
)

from dinkster.compose import PackSpec, ServingComposer


@pytest.mark.parametrize(
    ("node", "seconds", "shape", "ratio"),
    (
        (EmptyLatentAudio, 1.068, (2, 64, 22), 2048),
        (EmptyLatentAudio, 1.07, (2, 64, 24), 2048),
        (EmptyAceStepLatentAudio, 1.02, (2, 8, 16, 10), None),
        (EmptyAceStepLatentAudio, 1.03, (2, 8, 16, 11), None),
        (EmptyAceStep15LatentAudio, 1.02, (2, 64, 26), 1764),
        (EmptyAceStep15LatentAudio, 1.06, (2, 64, 26), 1764),
        (EmptyYuE2LatentAudio, 1.02, (2, 64, 26), 1920),
        (EmptyYuE2LatentAudio, 1.06, (2, 64, 26), 1920),
        (EmptyYuE2LatentAudio, 0.0, (2, 64, 1), 1920),
    ),
)
def test_audio_duration_rounding_and_metadata(node, seconds, shape, ratio) -> None:
    latent = node.execute(seconds=seconds, batch_size=2)["LATENT"]
    samples = latent["samples"]
    assert samples.shape == shape
    assert samples.device.type == "cpu"
    assert samples.dtype == torch.float32
    assert not samples.count_nonzero()
    expected = {"type": "audio"}
    if ratio is not None:
        expected["downscale_ratio_temporal"] = ratio
    assert {key: value for key, value in latent.items() if key != "samples"} == expected


@pytest.mark.parametrize("in_process", (True,))
def test_ordinary_manifest_creates_audio_latents_twice(in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={"CUDA_VISIBLE_DEVICES": "", "DINKSTER_ACCELERATOR": "cpu"}
        )
        cases = (
            ("EmptyLatentAudio", (3, 64, 22), 2048),
            ("EmptyAceStepLatentAudio", (3, 8, 16, 10), None),
            ("EmptyAceStep1.5LatentAudio", (3, 64, 26), 1764),
            ("EmptyYuE2LatentAudio", (3, 64, 26), 1920),
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    name: GraphNode(name, {"seconds": 1.02, "batch_size": 3})
                    for name, _, _ in cases
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, list(graph.nodes))
                for name, shape, ratio in cases:
                    latent = materialize_latent(result.outputs[name]["LATENT"].resolve())
                    samples = latent["samples"]
                    assert samples.shape == shape
                    assert samples.device.type == "cpu"
                    assert samples.dtype == torch.float32
                    assert not samples.count_nonzero()
                    assert latent["type"] == "audio"
                    assert latent.get("downscale_ratio_temporal") == ratio
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("shape", "expected"), (((3, 7, 11), (11, 1, 3)), ((2, 8, 5, 13), (13, 5, 2)))
)
def test_audio_latent_size_preserves_values(shape, expected) -> None:
    samples = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    before = samples.clone()
    latent = {"samples": samples, "type": "audio", "extra": "retained"}
    result = GetAudioLatentSize.execute(latent=latent)
    assert result == dict(zip(("length", "audio_channels", "batch_size"), expected, strict=True))
    assert all(type(value) is int for value in result.values())
    assert torch.equal(samples, before)
    assert latent["extra"] == "retained"


def test_audio_latent_size_uses_last_nested_stream() -> None:
    from dinkster_inference.nested_tensor import NestedTensor

    video = torch.zeros((1, 16, 4, 2, 3))
    audio = torch.arange(2 * 8 * 5 * 13, dtype=torch.float64).reshape(2, 8, 5, 13)
    nested = NestedTensor((video, audio))
    result = GetAudioLatentSize.execute(latent={"samples": nested})
    assert result == {"length": 13, "audio_channels": 5, "batch_size": 2}
    assert torch.equal(nested.unbind()[-1], audio)


@pytest.mark.parametrize("audio_last", (True, False))
def test_audio_size_canonical_streams_preserve_last_stream_semantics(audio_last: bool) -> None:
    video = torch.arange(1 * 16 * 4 * 2 * 3, dtype=torch.float64).reshape(1, 16, 4, 2, 3)
    audio = torch.arange(2 * 8 * 5 * 13, dtype=torch.float32).reshape(2, 8, 5, 13)
    pairs = (("video", video), ("audio", audio))
    samples = MultiStreamLatent.from_pairs(pairs if audio_last else pairs[::-1])
    if audio_last:
        assert GetAudioLatentSize.execute(latent={"samples": samples}) == {
            "length": 13,
            "audio_channels": 5,
            "batch_size": 2,
        }
    else:
        with pytest.raises(ValueError, match="too many values to unpack"):
            GetAudioLatentSize.execute(latent={"samples": samples})
    assert samples.by_role("video") is video
    assert samples.by_role("audio") is audio
    assert torch.equal(video.flatten(), torch.arange(video.numel(), dtype=torch.float64))
    assert torch.equal(audio.flatten(), torch.arange(audio.numel(), dtype=torch.float32))


class SizeSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "audio-size-fixture.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(LATENT_TYPE))
                for name in ("plain", "channel", "joint", "reverse")
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        audio = torch.arange(2 * 8 * 5 * 13, dtype=torch.float64).reshape(2, 8, 5, 13)
        video = torch.zeros((1, 16, 4, 2, 3))
        return cls.outputs(
            plain={"samples": torch.zeros((3, 7, 11))},
            channel={"samples": audio},
            joint={"samples": MultiStreamLatent.from_pairs((("video", video), ("audio", audio)))},
            reverse={"samples": MultiStreamLatent.from_pairs((("audio", audio), ("video", video)))},
        )


class SizeInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "audio-size-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(CORE_INT))
                for name in ("length", "audio_channels", "batch_size")
            ),
            outputs=(OutputSpec("length", TypeExpr.concrete(CORE_INT)),),
        )

    @classmethod
    def execute(cls, *, length: int, audio_channels: int, batch_size: int) -> dict[str, object]:
        assert all(type(value) is int for value in (length, audio_channels, batch_size))
        assert (length, audio_channels, batch_size) in ((11, 1, 3), (13, 5, 2))
        return cls.outputs(length=length)


@pytest.mark.parametrize("in_process", (True, False))
def test_audio_size_production_manifest_both_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(package / "src"))),
            }
        )
        (tmp_path / "audio_size_fixture.py").write_text(
            "from test_audio_latent_torch import SizeSource, SizeInspect\n"
            "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
            "NODES = (*COMFY_NODES, SizeSource, SizeInspect)\n",
            encoding="ascii",
        )
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            (package / "dinkster-pack.toml")
            .read_text()
            .replace("namespaces = [", 'namespaces = ["audio-size-fixture",')
            .replace("dinkster_nodes_comfy:COMFY_NODES", "audio_size_fixture:NODES")
            .replace(
                "dinkster_nodes_comfy:register_inference_types",
                "audio_size_fixture:register_inference_types",
            ),
            encoding="ascii",
        )
        try:
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            nodes = {"source": GraphNode("audio-size-fixture.source")}
            for name in ("plain", "channel", "joint"):
                nodes[name] = GraphNode("GetAudioLatentSize", {"latent": Link("source", name)})
                nodes[name + "_inspect"] = GraphNode(
                    "audio-size-fixture.inspect",
                    {
                        field: Link(name, field)
                        for field in ("length", "audio_channels", "batch_size")
                    },
                )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(
                    Graph(nodes=nodes),
                    [name + "_inspect" for name in ("plain", "channel", "joint")],
                )
                for name, expected in (("plain", 11), ("channel", 13), ("joint", 13)):
                    assert result.outputs[name + "_inspect"]["length"].resolve() == expected
                reverse = Graph(
                    nodes={
                        "source": nodes["source"],
                        "reverse": GraphNode(
                            "GetAudioLatentSize", {"latent": Link("source", "reverse")}
                        ),
                    }
                )
                with pytest.raises(ExecutionError, match="too many values to unpack"):
                    await engine.run(reverse, ["reverse"])
        finally:
            await composer.close()

    asyncio.run(scenario())
