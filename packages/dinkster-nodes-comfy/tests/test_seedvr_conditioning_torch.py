"""CPU contracts for native SeedVR2 conditioning."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    LATENT_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.seedvr_conditioning import SeedVR2Conditioning

from dinkster.compose import PackSpec, ServingComposer

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_MODEL = TypeExpr.concrete(MODEL_TYPE)


class _SeedVR2Module(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.positive_conditioning = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        self.negative_conditioning = torch.nn.Parameter(torch.tensor([-1.0, -2.0, -3.0]))


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diffusion_model = _SeedVR2Module()


def _model() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    return ModelPatcher(_Model(), torch.device("cpu"), torch.device("cpu"))


def _latent() -> dict[str, object]:
    return {
        "samples": torch.arange(2 * 16 * 3 * 2 * 4, dtype=torch.float64).reshape(2, 16, 3, 2, 4),
        "metadata": "preserved",
    }


def _assert_conditioning(positive: object, negative: object, source: dict[str, object]) -> None:
    source_samples = source["samples"]
    assert isinstance(source_samples, torch.Tensor)
    for conditioning, expected_vector in (
        (positive, [1.0, 2.0, 3.0]),
        (negative, [-1.0, -2.0, -3.0]),
    ):
        vector, metadata = conditioning[0]
        condition = metadata["condition"]
        assert torch.equal(vector, torch.tensor([expected_vector], dtype=torch.float32))
        assert condition.dtype == source_samples.dtype and condition.device == source_samples.device
        assert condition.shape == (2, 17, 3, 2, 4)
        assert torch.equal(condition[:, :16], source_samples)
        assert torch.equal(condition[:, 16], torch.ones_like(condition[:, 16]))


def test_seedvr_conditioning_preserves_source_and_builds_asymmetric_outputs() -> None:
    source = _latent()
    before = source["samples"].clone()
    output = SeedVR2Conditioning.execute(model=_model(), vae_conditioning=source)
    assert output["positive"][0][1]["condition"] is output["negative"][0][1]["condition"]
    _assert_conditioning(output["positive"], output["negative"], source)
    assert source["metadata"] == "preserved"
    assert torch.equal(source["samples"], before)


@pytest.mark.parametrize(
    "latent, message",
    (
        ({"samples": torch.zeros((2, 16, 3, 4))}, "5-D VAE latent"),
        ({"samples": torch.zeros((2, 3, 2, 4, 16))}, "channel-first layout"),
        ({"samples": torch.zeros((2, 15, 3, 2, 4))}, "16 channels"),
    ),
)
def test_seedvr_conditioning_rejects_invalid_latent_shapes(
    latent: dict[str, torch.Tensor], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SeedVR2Conditioning.execute(model=_model(), vae_conditioning=latent)


@pytest.mark.parametrize(
    "model, message",
    (
        (object(), "no 'model' attribute"),
        (type("Wrapper", (), {"model": None})(), "input.model is None"),
        (type("Wrapper", (), {"model": object()})(), "no 'diffusion_model' attribute"),
        (
            type("Wrapper", (), {"model": type("Inner", (), {"diffusion_model": None})()})(),
            "is None",
        ),
    ),
)
def test_seedvr_conditioning_rejects_invalid_models(model: object, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        SeedVR2Conditioning.execute(model=model, vae_conditioning=_latent())


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="seedvr-conditioning-fixture.source",
            outputs=(OutputSpec("model", _MODEL), OutputSpec("latent", _LATENT)),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(model=_model(), latent=_latent())


class Consumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="seedvr-conditioning-fixture.consumer",
            inputs=(InputSpec("positive", _CONDITIONING), InputSpec("negative", _CONDITIONING)),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, positive: object, negative: object) -> dict[str, object]:
        source = materialize_latent({"samples": _latent()["samples"]})
        _assert_conditioning(positive, negative, source)
        return cls.outputs(summary="seedvr-conditioning")


@pytest.mark.parametrize("in_process", (True,))
def test_seedvr_conditioning_executes_twice_through_both_production_hosts(
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
            native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "seedvr_conditioning_fixture.py").write_text(
                "from test_seedvr_conditioning_torch import Consumer, Source\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (Source, Consumer)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "seedvr-conditioning-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["seedvr-conditioning-fixture"]\n\n[pack.entry]\n'
                'nodes = "seedvr_conditioning_fixture:NODES"\n'
                'types = "seedvr_conditioning_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            for manifest in (native_manifest, fixture_manifest):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "seedvr-conditioning-fixture",
                        group_manifests=() if in_process else (native_manifest, fixture_manifest),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("seedvr-conditioning-fixture.source"),
                    "conditioning": GraphNode(
                        "SeedVR2Conditioning",
                        {
                            "model": Link("source", "model"),
                            "vae_conditioning": Link("source", "latent"),
                        },
                    ),
                    "consumer": GraphNode(
                        "seedvr-conditioning-fixture.consumer",
                        {
                            "positive": Link("conditioning", "positive"),
                            "negative": Link("conditioning", "negative"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "seedvr-conditioning"
        finally:
            await composer.close()

    asyncio.run(scenario())
