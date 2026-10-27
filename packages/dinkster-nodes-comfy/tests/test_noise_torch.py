"""CPU behavior for native ComfyUI noise nodes."""

import asyncio
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    LATENT_TYPE,
    MODEL_TYPE,
    SIGMAS_TYPE,
    InputSpec,
    MultiStreamLatent,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.noise import AddNoise, DisableNoise, RandomNoise

from dinkster.compose import PackSpec, ServingComposer

# Arithmetic and identity expectations below are hand-derived from the supplied
# tensors and fixture transforms, except the explicitly attributed random draw.


def test_random_noise_uses_seeded_batch_indices_and_disable_noise_preserves_layout() -> None:
    latent = {
        "samples": torch.zeros((2, 1, 2, 2), dtype=torch.float32),
        "batch_index": [7, 3],
    }
    noise = RandomNoise.execute(noise_seed=42)["NOISE"].generate_noise(latent)
    # ComfyUI Noise_Random, F bc5f880d: seed=42, float32 zeros (2,1,2,2),
    # batch_index=[7,3]; generate_noise delegates to comfy.sample.prepare_noise.
    assert noise.flatten().tolist() == [
        -0.049903348088264465,
        0.5263369679450989,
        -0.008498823270201683,
        0.7290605902671814,
        1.110290288925171,
        -1.6897989511489868,
        -0.9889599084854126,
        0.9579718112945557,
    ]
    disabled = DisableNoise.execute()["NOISE"].generate_noise(latent)
    assert disabled.dtype == latent["samples"].dtype
    assert disabled.device == latent["samples"].device
    assert torch.equal(disabled, torch.zeros_like(latent["samples"]))


def test_disable_noise_uses_pinned_cpu_nested_noise_preparation() -> None:
    from dinkster_api.v1 import initialize_inference_process

    initialize_inference_process()
    from dinkster_inference.nested_tensor import NestedTensor

    samples = NestedTensor([torch.ones((1, 1, 2, 2), dtype=torch.float64)])
    disabled = DisableNoise.execute()["NOISE"].generate_noise({"samples": samples})
    assert disabled.is_nested
    tensor = disabled.unbind()[0]
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float64
    assert torch.equal(tensor, torch.zeros_like(tensor))


@pytest.mark.parametrize("noise_node", (RandomNoise, DisableNoise))
def test_public_noise_preserves_asymmetric_av_roles_shapes_and_dtypes(
    noise_node: type[Any],
) -> None:
    video = torch.zeros((1, 3, 2, 2, 3), dtype=torch.float64)
    audio = torch.zeros((1, 2, 4), dtype=torch.float32)
    samples = MultiStreamLatent.from_pairs((("video", video), ("audio", audio)))
    noise = (
        noise_node.execute(noise_seed=42)["NOISE"]
        if noise_node is RandomNoise
        else noise_node.execute()["NOISE"]
    ).generate_noise({"samples": samples, "batch_index": [3]})
    assert type(noise) is MultiStreamLatent
    assert noise.roles == ("video", "audio")
    assert noise.by_role("video").shape == video.shape
    assert noise.by_role("video").dtype == video.dtype
    assert noise.by_role("audio").shape == audio.shape
    assert noise.by_role("audio").dtype == audio.dtype


class _Sampling:
    def noise_scaling(
        self, scale: torch.Tensor, noise: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        return latent + noise * scale


class _Model:
    def get_model_object(self, name: str) -> object:
        return {
            "model_sampling": _Sampling(),
            "process_latent_in": lambda value: value + 10,
            "process_latent_out": lambda value: torch.nan_to_num(value * 2),
        }[name]


def test_add_noise_preserves_metadata_skips_empty_input_and_uses_sigma_endpoints() -> None:
    latent = {"samples": torch.zeros((1, 1, 1, 2)), "note": "preserved"}
    value = AddNoise.execute(
        model=_Model(),
        noise=DisableNoise.execute()["NOISE"],
        sigmas=torch.tensor([4.0, 1.5]),
        latent_image=latent,
    )["LATENT"]
    assert value["note"] == "preserved"
    assert torch.equal(value["samples"], torch.zeros((1, 1, 1, 2)))
    assert (
        AddNoise.execute(
            model=_Model(),
            noise=DisableNoise.execute()["NOISE"],
            sigmas=torch.tensor([]),
            latent_image=latent,
        )["LATENT"]
        is latent
    )


def test_add_noise_uses_absolute_endpoint_delta_and_transforms_nonzero_latents() -> None:
    latent = {"samples": torch.tensor([[[[1.0, -2.0]]]]), "note": "kept"}

    class Noise:
        def generate_noise(self, _: object) -> torch.Tensor:
            return torch.tensor([[[[2.0, -3.0]]]])

    value = AddNoise.execute(
        model=_Model(), noise=Noise(), sigmas=torch.tensor([1.0, 4.0]), latent_image=latent
    )["LATENT"]
    assert value["note"] == "kept"
    assert torch.equal(value["samples"], torch.tensor([[[[34.0, -2.0]]]]))
    one_sigma = AddNoise.execute(
        model=_Model(), noise=Noise(), sigmas=torch.tensor([3.0]), latent_image=latent
    )["LATENT"]
    assert torch.equal(one_sigma["samples"], torch.tensor([[[[34.0, -2.0]]]]))


@pytest.mark.parametrize("empty", (False, True))
def test_add_noise_handles_native_av_arithmetic_and_preserves_source(empty: bool) -> None:
    class Model:
        def get_model_object(self, name: str) -> object:
            return {
                "model_sampling": _Sampling(),
                "process_latent_in": lambda value: value + 10,
                "process_latent_out": lambda value: value * 2,
            }[name]

    video = torch.zeros((1, 1, 2, 2, 3), dtype=torch.float64)
    audio = torch.zeros((1, 1, 4), dtype=torch.float32)
    if not empty:
        audio[0, 0] = torch.tensor([3.0, float("nan"), float("inf"), -float("inf")])
    original_audio = audio.clone()
    samples = MultiStreamLatent.from_pairs((("video", video), ("audio", audio)))
    source = {"samples": samples, "note": "kept"}
    result = AddNoise.execute(
        model=Model(),
        noise=DisableNoise.execute()["NOISE"],
        sigmas=torch.tensor([4.0, 1.0]),
        latent_image=source,
    )["LATENT"]
    assert result["note"] == "kept"
    assert result["samples"].roles == ("video", "audio")
    # Hand-derived: zero noise, shift both streams iff any stream is nonzero,
    # then multiply by two. Mixed empty/nonempty streams detect per-stream shifts.
    shift = 0 if empty else 10
    torch.testing.assert_close(result["samples"].by_role("video"), (video + shift) * 2)
    expected_audio = torch.zeros_like(audio) if empty else torch.tensor([[[26.0, 0.0, 0.0, 0.0]]])
    torch.testing.assert_close(result["samples"].by_role("audio"), expected_audio)
    torch.testing.assert_close(audio, original_audio, equal_nan=True)
    assert source["samples"] is samples


class _PackSampling:
    def noise_scaling(
        self, scale: torch.Tensor, noise: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        return latent + noise * scale


class _PackModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.model_sampling = _PackSampling()
        self.process_latent_in = lambda value: value + 1
        self.process_latent_out = lambda value: value * 2


def _pack_model() -> Any:
    from dinkster_api.v1 import initialize_inference_process

    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    return ModelPatcher(_PackModule(), torch.device("cpu"), torch.device("cpu"))


class _Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="noise-fixture.source",
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("sigmas", TypeExpr.concrete(SIGMAS_TYPE)),
                OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(
            model=_pack_model(),
            sigmas=torch.tensor([4.0, 1.0]),
            latent={
                "samples": torch.tensor([[[[1.0, -2.0]]], [[[3.0, 4.0]]]]),
                "noise_mask": torch.tensor([[[0.0, 0.25]], [[0.5, 1.0]]]),
                "batch_index": [7, 3],
                "note": "preserved",
            },
        )


class _Consumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="noise-fixture.consumer",
            inputs=(
                InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("original", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("random", TypeExpr.concrete(CORE_BOOLEAN)),
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, latent: Any, original: Any, random: bool) -> Mapping[str, object]:
        materialized = materialize_latent(latent)
        original = materialize_latent(original)
        assert materialized["note"] == "preserved"
        assert materialized["batch_index"] == [7, 3]
        source = torch.tensor([[[[1.0, -2.0]]], [[[3.0, 4.0]]]])
        mask = torch.tensor([[[0.0, 0.25]], [[0.5, 1.0]]])
        assert torch.equal(original["samples"], source)
        assert torch.equal(original["noise_mask"], mask)
        assert torch.equal(materialized["noise_mask"], mask)
        assert original["batch_index"] == [7, 3]
        assert original["note"] == "preserved"
        if random:
            generator = torch.Generator(device="cpu").manual_seed(42)
            draws = [torch.randn((1, 1, 1, 2), generator=generator) for _ in range(8)]
            noise = torch.cat((draws[7], draws[3]))
            expected = 2 * (source + 1 + noise * 3)
        else:
            expected = 2 * (source + 1)
        assert torch.equal(materialized["samples"], expected)
        return cls.outputs(value=float(materialized["samples"].sum()))


def test_noise_nodes_cross_an_ordinary_complete_pack_twice_in_both_hosts(tmp_path: Path) -> None:
    async def scenario(in_process: bool, noise_node: str) -> None:
        manifest = Path(__file__).parents[1] / "dinkster-pack.toml"
        fixture_manifest = Path(__file__).with_name("noise-fixture.toml")
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), *sys.path)),
                "DINKSTER_ACCELERATOR": "cpu",
            }
        )
        try:
            manifests = (manifest, fixture_manifest)
            if in_process:
                await composer.add_pack(PackSpec(manifest, in_process=True))
                await composer.add_pack(PackSpec(fixture_manifest, in_process=True))
            else:
                await composer.add_pack(
                    PackSpec(manifest, worker_group="noise", group_manifests=manifests)
                )
                await composer.add_pack(
                    PackSpec(fixture_manifest, worker_group="noise", group_manifests=manifests)
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("noise-fixture.source"),
                    "noise": GraphNode(noise_node, {"noise_seed": 42})
                    if noise_node == "RandomNoise"
                    else GraphNode(noise_node),
                    "add": GraphNode(
                        "AddNoise",
                        {
                            "model": Link("source", "model"),
                            "noise": Link("noise", "NOISE"),
                            "sigmas": Link("source", "sigmas"),
                            "latent_image": Link("source", "latent"),
                        },
                    ),
                    "consumer": GraphNode(
                        "noise-fixture.consumer",
                        {
                            "latent": Link("add", "LATENT"),
                            "original": Link("source", "latent"),
                            "random": noise_node == "RandomNoise",
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert isinstance(result.outputs["consumer"]["value"].resolve(), float)
                if _:
                    assert set(result.cached) == set(graph.nodes)
        finally:
            await composer.close()

    for in_process in (True, False):
        for noise_node in ("RandomNoise", "DisableNoise"):
            asyncio.run(scenario(in_process, noise_node))
