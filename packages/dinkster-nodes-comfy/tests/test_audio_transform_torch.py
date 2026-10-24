"""CPU audio channel and volume behavior through native pack hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_TYPE,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_audio,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.audio_transform import AudioAdjustVolume, SplitAudioChannels

from dinkster.compose import PackSpec, ServingComposer


def _audio() -> dict[str, object]:
    return {
        "waveform": torch.tensor(
            [
                [[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]],
                [[-9.0, 10.0, -11.0, 12.0], [13.0, -14.0, 15.0, -16.0]],
            ]
        ),
        "sample_rate": 44100,
        "extra": "not-output",
    }


def test_split_and_volume_preserve_audio_contracts() -> None:
    audio = _audio()
    waveform = audio["waveform"].to(torch.float64)
    audio["waveform"] = waveform
    original = waveform.clone()
    left, right = (SplitAudioChannels.execute(audio=audio)[name] for name in ("left", "right"))
    assert set(left) == set(right) == {"waveform", "sample_rate"}
    assert left["sample_rate"] == right["sample_rate"] == 44100
    assert torch.equal(left["waveform"], waveform[..., 0:1, :])
    assert torch.equal(right["waveform"], waveform[..., 1:2, :])
    assert left["waveform"].dtype == waveform.dtype
    assert left["waveform"].device == waveform.device
    assert SplitAudioChannels.execute(audio=None) == {"left": None, "right": None}
    for channels in (1, 3):
        invalid = {"waveform": torch.zeros((2, channels, 4)), "sample_rate": 44100}
        with pytest.raises(ValueError, match=rf"got {channels} channel\(s\)"):
            SplitAudioChannels.execute(audio=invalid)

    assert AudioAdjustVolume.execute(audio=None, volume=6) == {"AUDIO": None}
    assert AudioAdjustVolume.execute(audio=audio, volume=0)["AUDIO"] is audio
    opaque = object()
    assert AudioAdjustVolume.execute(audio=opaque, volume=0)["AUDIO"] is opaque
    for decibels in (6, -6):
        output = AudioAdjustVolume.execute(audio=audio, volume=decibels)["AUDIO"]
        assert set(output) == {"waveform", "sample_rate"}
        assert output["sample_rate"] == 44100
        assert output["waveform"].dtype == waveform.dtype
        assert output["waveform"].device == waveform.device
        assert torch.allclose(output["waveform"], waveform * 10 ** (decibels / 20))
    assert audio["waveform"] is waveform
    assert torch.equal(waveform, original)
    assert audio["extra"] == "not-output"


class AudioSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="audio-transform-fixture.source",
            outputs=(OutputSpec("audio", TypeExpr.concrete(AUDIO_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(audio=_audio())


class AudioInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        audio = TypeExpr.concrete(AUDIO_TYPE)
        return NodeSchema(
            node_type="audio-transform-fixture.inspect",
            inputs=(
                InputSpec("left", audio),
                InputSpec("right", audio),
                InputSpec("adjusted", audio),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, left: Any, right: Any, adjusted: Any) -> dict[str, object]:
        left = materialize_audio(left)
        right = materialize_audio(right)
        adjusted = materialize_audio(adjusted)
        assert all(set(audio) == {"waveform", "sample_rate"} for audio in (left, right, adjusted))
        assert left["sample_rate"] == right["sample_rate"] == adjusted["sample_rate"] == 44100
        assert tuple(left["waveform"].shape) == tuple(right["waveform"].shape) == (2, 1, 4)
        assert tuple(adjusted["waveform"].shape) == (2, 2, 4)
        assert all(audio["waveform"].dtype == torch.float32 for audio in (left, right, adjusted))
        assert all(audio["waveform"].device.type == "cpu" for audio in (left, right, adjusted))
        torch.testing.assert_close(
            left["waveform"],
            torch.tensor([[[1.0, -2.0, 3.0, -4.0]], [[-9.0, 10.0, -11.0, 12.0]]]),
        )
        torch.testing.assert_close(
            right["waveform"],
            torch.tensor([[[5.0, -6.0, 7.0, -8.0]], [[13.0, -14.0, 15.0, -16.0]]]),
        )
        torch.testing.assert_close(
            adjusted["waveform"],
            torch.tensor(
                [
                    [[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]],
                    [[-9.0, 10.0, -11.0, 12.0], [13.0, -14.0, 15.0, -16.0]],
                ]
            )
            * 10 ** (6 / 20),
        )
        return cls.outputs(summary="ok")


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_audio_transforms_twice(
    in_process: bool, tmp_path: Path
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(source_root))),
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "audio_transform_fixture.py").write_text(
                "from test_audio_transform_torch import AudioInspect, AudioSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_media_io import register_media_types\n"
                "NODES = (*COMFY_NODES, AudioSource, AudioInspect)\n"
                "\ndef register_types(registry):\n"
                "    register_inference_types(registry)\n"
                "    register_media_types(registry)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["audio-transform-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "audio_transform_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "audio_transform_fixture:register_types",
                ),
                encoding="ascii",
            )
            media_manifest = manifest.parent.parent / "dinkster-nodes-media-io/dinkster-pack.toml"
            await composer.add_pack(
                PackSpec(manifest=media_manifest, in_process=in_process, trust_reserved=True)
            )
            await composer.add_pack(PackSpec(manifest=fixture, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("audio-transform-fixture.source"),
                    "split": GraphNode("SplitAudioChannels", {"audio": Link("source", "audio")}),
                    "volume": GraphNode(
                        "AudioAdjustVolume", {"audio": Link("source", "audio"), "volume": 6}
                    ),
                    "inspect": GraphNode(
                        "audio-transform-fixture.inspect",
                        {
                            "left": Link("split", "left"),
                            "right": Link("split", "right"),
                            "adjusted": Link("volume", "AUDIO"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["summary"].resolve() == "ok"
        finally:
            await composer.close()

    asyncio.run(scenario())
