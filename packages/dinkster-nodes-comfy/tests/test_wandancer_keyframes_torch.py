"""WanDancer frame placement, audio slicing and ordinary-host transport."""

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
    IMAGE_TYPE,
    MASK_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_audio,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.wandancer_keyframes import (
    WanDancerPadKeyframes,
    WanDancerPadKeyframesList,
)

from dinkster.compose import PackSpec, ServingComposer


def images() -> torch.Tensor:
    return torch.arange(7 * 2 * 3 * 3, dtype=torch.float32).reshape(7, 2, 3, 3) / 128


def audio(length: int = 17) -> dict[str, Any]:
    return {
        "waveform": torch.arange(2 * length, dtype=torch.float32).reshape(1, 2, length) / 64,
        "sample_rate": 30,
    }


def check(result: dict[str, Any], index: int) -> None:
    frames = materialize_image(result["keyframes_sequence"])
    mask = materialize_image(result["keyframes_mask"])
    sound = materialize_audio(result["audio_segment"])
    expected = torch.zeros((5, 2, 3, 3))
    expected[0], expected[4] = images()[index], images()[index + 1]
    expected_mask = torch.zeros((5, 2, 3))
    expected_mask[0] = expected_mask[4] = 1
    torch.testing.assert_close(frames, expected, rtol=0, atol=0)
    torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)
    assert frames.dtype == mask.dtype == torch.float32
    assert frames.device.type == mask.device.type == "cpu"
    assert sound["sample_rate"] == 30
    torch.testing.assert_close(
        sound["waveform"],
        audio()["waveform"][:, :, index * 5 : min(index * 5 + 5, 17)],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("index", (0, 1, 2, 3))
def test_keyframe_positions_audio_slice_and_sources(index: int) -> None:
    pixels, sound = images(), audio()
    before_image, before_sound = pixels.clone(), sound["waveform"].clone()
    result = WanDancerPadKeyframes.execute(
        images=pixels, segment_length=5, segment_index=index, audio=sound
    )
    check(dict(result), index)
    assert torch.equal(pixels, before_image) and torch.equal(sound["waveform"], before_sound)


def test_list_outputs_and_audio_buffer_boundary() -> None:
    outputs = WanDancerPadKeyframesList.execute(
        images=images(), segment_length=5, num_segments=4, audio=audio()
    )
    assert all(len(value) == 4 for value in outputs.values())
    for index in range(4):
        check({key: value[index] for key, value in outputs.items()}, index)
    for count in (0, 7):
        result = WanDancerPadKeyframes.execute(
            images=images()[:count], segment_length=5, segment_index=0, audio=audio(6)
        )
        assert not result["keyframes_sequence"].count_nonzero()
        assert not result["keyframes_mask"].count_nonzero()
        assert result["audio_segment"]["waveform"].shape == (1, 2, 5)
    result = WanDancerPadKeyframes.execute(
        images=images()[:1], segment_length=5, segment_index=0, audio=audio(7)
    )
    assert torch.equal(result["keyframes_sequence"][4], images()[0])
    assert result["keyframes_mask"].sum() == 6


class KeyframeSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wandancer-keyframe-fixture.source",
            outputs=(
                OutputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),
                OutputSpec("audio", TypeExpr.concrete(AUDIO_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(images=images(), audio=audio())


class KeyframeConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wandancer-keyframe-fixture.consumer",
            inputs=tuple(
                InputSpec(name, TypeExpr.list_of(TypeExpr.concrete(kind)))
                for name, kind in (
                    ("keyframes_sequence", IMAGE_TYPE),
                    ("keyframes_mask", MASK_TYPE),
                    ("audio_segment", AUDIO_TYPE),
                )
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, **inputs: Any) -> dict[str, object]:
        assert all(len(value) == 4 for value in inputs.values())
        for index in range(4):
            check({key: value[index] for key, value in inputs.items()}, index)
        return cls.outputs(summary="wandancer-keyframes")


class KeyframeScalarConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wandancer-keyframe-fixture.scalar-consumer",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("keyframes_sequence", IMAGE_TYPE),
                    ("keyframes_mask", MASK_TYPE),
                    ("audio_segment", AUDIO_TYPE),
                )
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, **inputs: Any) -> dict[str, object]:
        check(inputs, 1)
        return cls.outputs(summary="wandancer-keyframes")


@pytest.mark.parametrize("in_process", (True, False))
def test_keyframes_run_twice_through_ordinary_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        module = "wandancer_keyframe_fixture"
        (tmp_path / f"{module}.py").write_text(
            "from test_wandancer_keyframes_torch import (\n"
            "    KeyframeSource, KeyframeConsumer, KeyframeScalarConsumer)\n"
            "from dinkster_api.v1 import (AUDIO_TYPE, register_inference_types,\n"
            "    register_audio_value_type, resolver_from_env)\n"
            "NODES=(KeyframeSource,KeyframeConsumer,KeyframeScalarConsumer)\n"
            "def register_types(registry):\n"
            "    register_inference_types(registry)\n"
            "    if AUDIO_TYPE not in registry:\n"
            "        register_audio_value_type(registry, AUDIO_TYPE, resolver_from_env())\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="wandancer-keyframe-fixture"\npublisher="dinkster"\nnamespaces=["wandancer-keyframe-fixture"]\n[pack.entry]\n'
            + f'nodes="{module}:NODES"\ntypes="{module}:register_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(package / "src"))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "wandancer-keyframe-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("wandancer-keyframe-fixture.source", {}),
                    "pad": GraphNode(
                        "WanDancerPadKeyframesList",
                        {
                            "images": Link("source", "images"),
                            "audio": Link("source", "audio"),
                            "segment_length": 5,
                            "num_segments": 4,
                        },
                    ),
                    "consumer": GraphNode(
                        "wandancer-keyframe-fixture.consumer",
                        {
                            key: Link("pad", key)
                            for key in ("keyframes_sequence", "keyframes_mask", "audio_segment")
                        },
                    ),
                    "scalar": GraphNode(
                        "WanDancerPadKeyframes",
                        {
                            "images": Link("source", "images"),
                            "audio": Link("source", "audio"),
                            "segment_length": 5,
                            "segment_index": 1,
                        },
                    ),
                    "scalar-consumer": GraphNode(
                        "wandancer-keyframe-fixture.scalar-consumer",
                        {
                            key: Link("scalar", key)
                            for key in ("keyframes_sequence", "keyframes_mask", "audio_segment")
                        },
                    ),
                }
            )
            for _ in range(2):
                engine = composer.composition.make_engine(lambda _event: None)
                result = await engine.run(graph, ["consumer", "scalar-consumer"])
                for consumer in ("consumer", "scalar-consumer"):
                    assert result.outputs[consumer]["summary"].resolve() == "wandancer-keyframes"
        finally:
            await composer.close()

    asyncio.run(scenario())
