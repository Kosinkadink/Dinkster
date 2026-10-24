"""CPU coverage for the public AUDIO tensor materializer."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
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
from dinkster_values.audio_codec import (
    AudioSourceUnavailableError,
    audio_from_source,
    audio_window,
    coerce_audio,
    render_audio_wav,
)
from dinkster_values.audio_lazy import AUDIO_WINDOW_LIMIT

from dinkster.compose import PackSpec, ServingComposer

torch = pytest.importorskip("torch")


def test_materialize_audio_keeps_live_float_tensor_mapping_by_identity() -> None:
    waveform = torch.arange(12, dtype=torch.float64).reshape(1, 2, 6)
    metadata = {"kept": [1, 2]}
    audio = {"waveform": waveform, "sample_rate": 48000, "metadata": metadata}
    materialized = materialize_audio(audio)
    materialized_waveform = cast(Any, materialized["waveform"])
    assert materialized is audio
    assert materialized_waveform is waveform
    assert materialized_waveform.dtype == torch.float64
    assert materialized_waveform.device == waveform.device
    assert materialized["metadata"] is metadata

    pcm = torch.tensor([[[-32768, 32767]]], dtype=torch.int16)
    normalized = materialize_audio({"waveform": pcm, "sample_rate": 8000})
    normalized_waveform = cast(Any, normalized["waveform"])
    assert normalized_waveform.dtype == torch.float32
    assert normalized_waveform.device == pcm.device
    torch.testing.assert_close(
        normalized_waveform, torch.tensor([[[-1.0, 32767 / 32768]]], dtype=torch.float32)
    )


def test_materialize_audio_converts_portable_pcm_and_preserves_metadata() -> None:
    floats = np.array([[[0.25, -0.5]], [[0.75, 1.0]]], dtype=np.float32)
    portable = {"waveform": floats, "sample_rate": 44100, "metadata": {"name": "portable"}}
    materialized = materialize_audio(portable)
    assert materialized["sample_rate"] == 44100
    assert materialized["metadata"] == {"name": "portable"}
    waveform = cast(Any, materialized["waveform"])
    assert waveform.dtype == torch.float32
    torch.testing.assert_close(waveform, torch.from_numpy(floats))

    pcm = {"waveform": np.array([[[-32768, 32767]]], dtype=np.int16), "sample_rate": 8000}
    normalized = materialize_audio(pcm)
    torch.testing.assert_close(
        normalized["waveform"], torch.tensor([[[-1.0, 32767 / 32768]]], dtype=torch.float32)
    )


def test_materialize_audio_reads_known_wav_extent_and_refuses_unbounded_whole_audio() -> None:
    samples = np.linspace(-0.5, 0.5, 16, dtype=np.float32).reshape(1, 1, 16)
    lazy = audio_from_source(render_audio_wav({"waveform": samples, "sample_rate": 8000}))
    materialized = materialize_audio(lazy)
    torch.testing.assert_close(
        materialized["waveform"], torch.from_numpy(samples), atol=1 / 32768, rtol=0
    )
    assert materialized["sample_rate"] == 8000
    assert audio_window(lazy, 0, 16)["waveform"].shape == (1, 1, 16)

    probe = cast(dict[str, object], lazy["probe"])
    unknown = {**lazy, "probe": {**probe, "frames": None, "duration": None}}
    with pytest.raises(ValueError, match="unknown extent"):
        materialize_audio(unknown)

    batch, channels = 2, 3
    frames = AUDIO_WINDOW_LIMIT // (batch * channels * 4) + 1
    over_budget = {
        **lazy,
        "probe": {
            **probe,
            "batch": batch,
            "channels": channels,
            "frames": frames,
            "duration": frames / 8000,
        },
    }
    with pytest.raises(ValueError, match="32 MiB"):
        materialize_audio(over_budget)


def test_materialize_audio_refuses_unavailable_lazy_source() -> None:
    samples = np.zeros((1, 1, 8), dtype=np.float32)
    lazy = audio_from_source(render_audio_wav({"waveform": samples, "sample_rate": 8000}))
    portable = {
        "source": {"digest": f"blake3:{'0' * 64}", "size": 1},
        "probe": lazy["probe"],
        "edits": [],
    }
    with pytest.raises(AudioSourceUnavailableError, match="no local asset binding"):
        materialize_audio(portable)


def test_materialize_audio_accepts_exact_lazy_window_budget() -> None:
    batch, channels = 2, 2
    frames = AUDIO_WINDOW_LIMIT // (batch * channels * 4)
    samples = np.zeros((batch, channels, frames), dtype=np.float32)
    materialized = materialize_audio(coerce_audio({"waveform": samples, "sample_rate": 8000}))
    waveform = cast(Any, materialized["waveform"])
    assert tuple(waveform.shape) == (batch, channels, frames)
    assert waveform.dtype == torch.float32
    assert waveform.device.type == "cpu"
    assert not waveform.count_nonzero()


class AudioSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="audio-materializer-fixture.source",
            outputs=(OutputSpec("audio", TypeExpr.concrete(AUDIO_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        waveform = np.array([[[0.25, -0.5, 0.75], [-0.25, 0.5, -0.75]]], dtype=np.float32)
        return cls.outputs(audio={"waveform": waveform, "sample_rate": 12000})


class AudioConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="audio-materializer-fixture.consumer",
            inputs=(InputSpec("audio", TypeExpr.concrete(AUDIO_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, audio: object) -> Mapping[str, object]:
        materialized = materialize_audio(audio)
        waveform = cast(Any, materialized["waveform"])
        assert materialized["sample_rate"] == 12000
        assert tuple(waveform.shape) == (1, 2, 3)
        assert waveform.dtype == torch.float32
        assert waveform.device.type == "cpu"
        torch.testing.assert_close(
            waveform,
            torch.tensor([[[0.25, -0.5, 0.75], [-0.25, 0.5, -0.75]]]),
        )
        return cls.outputs(summary=f"{materialized['sample_rate']}:{tuple(waveform.shape)}")


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_materializes_audio_twice_in_both_hosts(
    in_process: bool, tmp_path: Path
) -> None:
    async def scenario() -> None:
        source_root = (
            Path(__file__).resolve().parents[1] / "packages" / "dinkster-nodes-media-io" / "src"
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(source_root))),
            }
        )
        try:
            manifest = (
                Path(__file__).resolve().parents[1]
                / "packages"
                / "dinkster-nodes-media-io"
                / "dinkster-pack.toml"
            )
            (tmp_path / "audio_materializer_fixture.py").write_text(
                "from test_audio_materializer import AudioConsumer, AudioSource\n"
                "from dinkster_nodes_media_io import (\n"
                "    MEDIA_IO_NODES, capture_device_choices, register_media_types,\n"
                ")\n"
                "NODES = (*MEDIA_IO_NODES, AudioSource, AudioConsumer)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["audio-materializer-fixture",')
                .replace(
                    "dinkster_nodes_media_io:MEDIA_IO_NODES", "audio_materializer_fixture:NODES"
                )
                .replace(
                    "dinkster_nodes_media_io:register_media_types",
                    "audio_materializer_fixture:register_media_types",
                )
                .replace(
                    "dinkster_nodes_media_io:capture_device_choices",
                    "audio_materializer_fixture:capture_device_choices",
                ),
                encoding="ascii",
            )
            await composer.add_pack(
                PackSpec(manifest=fixture_manifest, in_process=in_process, trust_reserved=True)
            )
            graph = Graph(
                nodes={
                    "source": GraphNode("audio-materializer-fixture.source"),
                    "consumer": GraphNode(
                        "audio-materializer-fixture.consumer", {"audio": Link("source", "audio")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "12000:(1, 2, 3)"
        finally:
            await composer.close()

    asyncio.run(scenario())
