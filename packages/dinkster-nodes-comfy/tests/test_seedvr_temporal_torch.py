"""SeedVR2 temporal chunking and merging."""

import asyncio
import math
import os
from pathlib import Path

import dinkster_nodes_comfy.seedvr_temporal as seedvr_temporal
import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.seedvr_temporal import SeedVR2TemporalChunk, SeedVR2TemporalMerge

from dinkster.compose import PackSpec, ServingComposer


def _latent(frames: int = 7) -> dict[str, object]:
    return {
        "samples": torch.arange(16 * frames * 2 * 2, dtype=torch.float64).reshape(
            1, 16, frames, 2, 2
        ),
        "metadata": {"source": "seedvr"},
        "noise_mask": torch.zeros(1),
    }


def test_manual_chunks_and_merges_with_runt_final_chunk(monkeypatch) -> None:
    monkeypatch.setattr(seedvr_temporal, "initialize_inference_process", lambda: None)
    source = _latent()
    chunks = SeedVR2TemporalChunk.execute(
        latent=source,
        temporal_overlap=2,
        chunking_mode={"chunking_mode": "manual", "frames_per_chunk": 13},
    )
    assert [chunk["samples"].shape[2] for chunk in chunks["latents"]] == [4, 4, 3]
    assert chunks["temporal_overlap"] == 2
    merged = SeedVR2TemporalMerge.execute(
        latents=chunks["latents"], temporal_overlap=chunks["temporal_overlap"]
    )["latent"]
    assert merged["samples"].shape == source["samples"].shape
    assert merged["samples"].dtype == source["samples"].dtype
    assert merged["metadata"] == source["metadata"]
    assert "noise_mask" not in merged


def test_hann_crossfade_weights_overlap(monkeypatch) -> None:
    monkeypatch.setattr(seedvr_temporal, "initialize_inference_process", lambda: None)
    overlap = 6
    first = {"samples": torch.zeros((1, 16, 7, 1, 1), dtype=torch.float64)}
    second = {"samples": torch.ones((1, 16, 7, 1, 1), dtype=torch.float64)}
    merged = SeedVR2TemporalMerge.execute(latents=[first, second], temporal_overlap=overlap)[
        "latent"
    ]["samples"]
    assert torch.equal(merged[:, :, 0], first["samples"][:, :, 0])
    assert torch.equal(merged[:, :, -1], second["samples"][:, :, -1])
    expected = []
    for index in range(overlap):
        ramp = min(1.0, max(0.0, (index / (overlap - 1) - 1.0 / 3.0) / (1.0 / 3.0)))
        previous = 0.5 + 0.5 * math.cos(math.pi * ramp)
        expected.append(1.0 - previous)
    actual = merged[0, 0, 1 : 1 + overlap, 0, 0].tolist()
    assert actual == pytest.approx(expected)
    assert actual[3] != pytest.approx(3 / (overlap - 1))


def test_auto_chunking_uses_free_memory_boundary(monkeypatch) -> None:
    monkeypatch.setattr(seedvr_temporal, "initialize_inference_process", lambda: None)
    source = _latent(8)

    class Management:
        @staticmethod
        def get_torch_device() -> str:
            return "cpu"

        @staticmethod
        def get_free_memory(device: str) -> int:
            assert device == "cpu"
            return int((8.5 + 4 * 0.55 + 0.55 * (2 * 2 * 64 / 1e6) * 2.1) * 1024**3)

    original = seedvr_temporal.importlib.import_module

    def import_module(name: str):
        return Management if name == "dinkster_inference.model_management" else original(name)

    monkeypatch.setattr(seedvr_temporal.importlib, "import_module", import_module)
    chunks = SeedVR2TemporalChunk.execute(
        latent=source, temporal_overlap=0, chunking_mode={"chunking_mode": "auto"}
    )
    assert [chunk["samples"].shape[2] for chunk in chunks["latents"]] == [2, 2, 2, 2]


class TemporalSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="seedvr-temporal-fixture.source",
            outputs=(OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(latent=_latent())


class TemporalConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="seedvr-temporal-fixture.consumer",
            inputs=(InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, latent: object) -> dict[str, object]:
        value = seedvr_temporal.materialize_latent(latent)
        assert value["metadata"] == {"source": "seedvr"}
        assert "noise_mask" not in value
        samples = value["samples"]
        assert samples.shape == (1, 16, 7, 2, 2)
        assert samples.dtype is torch.float64 and samples.device.type == "cpu"
        expected = _latent()["samples"]
        assert torch.equal(samples, expected)
        return cls.outputs(summary="seedvr-temporal")


@pytest.mark.parametrize("in_process", (True,))
@pytest.mark.parametrize(
    ("chunking_mode", "inputs"),
    (
        ("manual", {"frames_per_chunk": 13, "temporal_overlap": 2}),
        ("auto", {"temporal_overlap": 0}),
    ),
)
def test_temporal_nodes_cross_ordinary_manifest_hosts_twice(
    tmp_path: Path, in_process: bool, chunking_mode: str, inputs: dict[str, object]
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
            module = f"seedvr_temporal_fixture_{chunking_mode}_{int(in_process)}"
            (tmp_path / f"{module}.py").write_text(
                "from test_seedvr_temporal_torch import TemporalConsumer, TemporalSource\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (TemporalSource, TemporalConsumer)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname="seedvr-temporal-fixture"\npublisher="dinkster"\n'
                'namespaces=["seedvr-temporal-fixture"]\n[pack.entry]\n'
                f'nodes="{module}:NODES"\n'
                f'types="{module}:register_inference_types"\n',
                encoding="ascii",
            )
            native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            specs = (
                (
                    PackSpec(manifest=native_manifest, in_process=True),
                    PackSpec(manifest=fixture_manifest, in_process=True),
                )
                if in_process
                else (
                    PackSpec(
                        manifest=native_manifest,
                        worker_group="seedvr-temporal",
                        group_manifests=(native_manifest, fixture_manifest),
                    ),
                    PackSpec(
                        manifest=fixture_manifest,
                        worker_group="seedvr-temporal",
                        group_manifests=(native_manifest, fixture_manifest),
                    ),
                )
            )
            for spec in specs:
                await composer.add_pack(spec)
            chunk_inputs = {
                "latent": Link("source", "latent"),
                "temporal_overlap": inputs["temporal_overlap"],
            }
            if chunking_mode == "manual":
                chunk_inputs["chunking_mode.frames_per_chunk"] = inputs["frames_per_chunk"]
            graph = Graph(
                nodes={
                    "source": GraphNode("seedvr-temporal-fixture.source"),
                    "chunk": GraphNode(
                        "SeedVR2TemporalChunk",
                        chunk_inputs,
                        slot_variants={"chunking_mode": chunking_mode},
                    ),
                    "merge": GraphNode(
                        "SeedVR2TemporalMerge",
                        {
                            "latents": Link("chunk", "latents"),
                            "temporal_overlap": Link("chunk", "temporal_overlap"),
                        },
                    ),
                    "consumer": GraphNode(
                        "seedvr-temporal-fixture.consumer", {"latent": Link("merge", "latent")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "seedvr-temporal"
        finally:
            await composer.close()

    asyncio.run(scenario())
