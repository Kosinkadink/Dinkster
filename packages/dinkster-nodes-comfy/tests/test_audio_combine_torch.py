"""CPU behavior for native ComfyUI audio combine nodes."""

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
    register_audio_value_type,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.audio_combine import (
    AudioConcat,
    AudioMerge,
    JoinAudioChannels,
)

from dinkster.compose import PackSpec, ServingComposer

_AUDIO = TypeExpr.concrete(AUDIO_TYPE)


def _audio(values: list[list[list[float]]], rate: int = 8_000) -> dict[str, object]:
    return {"waveform": torch.tensor(values, dtype=torch.float32), "sample_rate": rate}


def test_null_audio_returns_the_original_object_without_materializing() -> None:
    opaque = object()
    assert JoinAudioChannels.execute(audio_left=opaque, audio_right=None)["audio"] is opaque
    assert AudioConcat.execute(audio1=None, audio2=opaque, direction="after")["AUDIO"] is opaque
    assert AudioMerge.execute(audio1=opaque, audio2=None, merge_method="add")["AUDIO"] is opaque


def test_join_rejects_non_mono_and_trims_with_left_right_order() -> None:
    left = _audio([[[1, 2, 3, 4]]])
    right = _audio([[[5, 6]]])
    joined = JoinAudioChannels.execute(audio_left=left, audio_right=right)["audio"]
    assert torch.equal(joined["waveform"], torch.tensor([[[1, 2], [5, 6]]], dtype=torch.float32))
    with pytest.raises(ValueError, match="Both input audios must be mono"):
        JoinAudioChannels.execute(audio_left=_audio([[[1], [2]]]), audio_right=right)


def test_concat_duplicates_mono_and_preserves_direction() -> None:
    first = _audio([[[1, 2]]])
    second = _audio([[[3]]])
    after = AudioConcat.execute(audio1=first, audio2=second, direction="after")["AUDIO"]
    before = AudioConcat.execute(audio1=first, audio2=second, direction="before")["AUDIO"]
    assert torch.equal(
        after["waveform"], torch.tensor([[[1, 2, 3], [1, 2, 3]]], dtype=torch.float32)
    )
    assert torch.equal(
        before["waveform"], torch.tensor([[[3, 1, 2], [3, 1, 2]]], dtype=torch.float32)
    )


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        ("add", [0.75, 0.5, 0.5]),
        ("mean", [0.375, 0.25, 0.25]),
        ("subtract", [0.25, 0.5, 0.5]),
        ("multiply", [0.125, 0.0, 0.0]),
    ),
)
def test_merge_methods_pad_to_first_input_extent(method: str, expected: list[float]) -> None:
    first = _audio([[[0.5, 0.5, 0.5]]])
    second = _audio([[[0.25]]])
    result = AudioMerge.execute(audio1=first, audio2=second, merge_method=method)["AUDIO"]
    assert torch.equal(result["waveform"], torch.tensor([[expected]], dtype=torch.float32))


def test_merge_normalizes_only_above_one_and_retains_dtype_device() -> None:
    first = _audio([[[0.75, 0.75]]])
    result = AudioMerge.execute(audio1=first, audio2=first, merge_method="add")["AUDIO"]
    assert torch.equal(result["waveform"], torch.ones((1, 1, 2)))
    assert result["waveform"].dtype is torch.float32
    assert result["waveform"].device.type == "cpu"


class AudioSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="audio-combine-fixture.source",
            outputs=tuple(
                OutputSpec(name, _AUDIO) for name in ("left", "right", "first", "second")
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            left=_audio([[[1, 2, 3]]]),
            right=_audio([[[4, 5]]]),
            first=_audio([[[0.75, 0.5]]]),
            second=_audio([[[0.75]]]),
        )


class AudioInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="audio-combine-fixture.inspect",
            inputs=(
                InputSpec("joined", _AUDIO),
                InputSpec("concatenated", _AUDIO),
                InputSpec("merged", _AUDIO),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, joined: Any, concatenated: Any, merged: Any) -> dict[str, object]:
        joined_waveform = materialize_audio(joined)["waveform"]
        concatenated_waveform = materialize_audio(concatenated)["waveform"]
        merged_waveform = materialize_audio(merged)["waveform"]
        assert torch.equal(joined_waveform, torch.tensor([[[1, 2], [4, 5]]], dtype=torch.float32))
        assert torch.equal(
            concatenated_waveform,
            torch.tensor([[[0.75, 0.5, 0.75], [0.75, 0.5, 0.75]]], dtype=torch.float32),
        )
        assert torch.equal(merged_waveform, torch.tensor([[[1.0, 1.0 / 3.0]]], dtype=torch.float32))
        return cls.outputs(summary="audio-combine")


def register_audio_combine_types(registry: Any) -> None:
    register_audio_value_type(registry, AUDIO_TYPE)


@pytest.mark.parametrize("in_process", (True, False))
def test_audio_combine_nodes_execute_twice_in_both_manifest_hosts(
    tmp_path: Path, in_process: bool
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
            (tmp_path / "audio_combine_fixture.py").write_text(
                "from test_audio_combine_torch import AudioInspect, AudioSource\n"
                "from test_audio_combine_torch import register_audio_combine_types\n"
                "from dinkster_nodes_comfy import COMFY_NODES\n"
                "NODES = (*COMFY_NODES, AudioSource, AudioInspect)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["audio-combine-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "audio_combine_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "audio_combine_fixture:register_audio_combine_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("audio-combine-fixture.source"),
                    "join": GraphNode(
                        "JoinAudioChannels",
                        {
                            "audio_left": Link("source", "left"),
                            "audio_right": Link("source", "right"),
                        },
                    ),
                    "concat": GraphNode(
                        "AudioConcat",
                        {
                            "audio1": Link("source", "first"),
                            "audio2": Link("source", "second"),
                            "direction": "after",
                        },
                    ),
                    "merge": GraphNode(
                        "AudioMerge",
                        {
                            "audio1": Link("source", "first"),
                            "audio2": Link("source", "second"),
                            "merge_method": "add",
                        },
                    ),
                    "inspect": GraphNode(
                        "audio-combine-fixture.inspect",
                        {
                            "joined": Link("join", "audio"),
                            "concatenated": Link("concat", "AUDIO"),
                            "merged": Link("merge", "AUDIO"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                assert (await engine.run(graph, ["inspect"])).outputs["inspect"][
                    "summary"
                ].resolve() == "audio-combine"
        finally:
            await composer.close()

    asyncio.run(scenario())
