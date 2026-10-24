"""Audio latent duration boundaries and ordinary manifest execution on CPU."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import materialize_latent
from dinkster_graph import Graph, GraphNode
from dinkster_nodes_comfy.audio_latent import (
    EmptyAceStep15LatentAudio,
    EmptyAceStepLatentAudio,
    EmptyLatentAudio,
    EmptyYuE2LatentAudio,
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


@pytest.mark.parametrize("in_process", (True, False))
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
