"""CPU coverage for native SIGMAS transport through ordinary pack hosting."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    NOISE_TYPE,
    SAMPLER_TYPE,
    SIGMAS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_inference_wire import register_inference_types
from dinkster_values import TypeRegistry

from dinkster.compose import PackSpec, ServingComposer


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _sigma_case(name: str) -> torch.Tensor:
    if name == "float64":
        return torch.tensor([0.125, -3.5, 19.25], dtype=torch.float64)
    if name == "noncontiguous-float32":
        return torch.arange(12, dtype=torch.float32).reshape(3, 4).t()[:, ::2]
    if name == "bfloat16":
        return torch.tensor([0.5, -1.5, 7.0], dtype=torch.bfloat16)
    if name == "empty":
        return torch.empty((0, 3), dtype=torch.float32)
    raise ValueError(f"unknown SIGMAS test case: {name}")


@pytest.mark.parametrize(
    "tensor",
    (
        _sigma_case("float64"),
        _sigma_case("noncontiguous-float32"),
        _sigma_case("bfloat16"),
        _sigma_case("empty"),
    ),
)
def test_sigmas_tensor_codec_preserves_cpu_tensor_bytes(tensor: torch.Tensor) -> None:
    source_pointer = tensor.data_ptr()
    before = tensor.clone()
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(SIGMAS_TYPE)

    encoded = spec.encode(tensor)
    decoded = spec.decode(encoded)
    restored = materialize_latent(decoded)

    assert tensor.data_ptr() == source_pointer
    assert tensor.dtype == before.dtype
    assert tensor.shape == before.shape
    assert torch.equal(tensor, before)
    assert decoded.dtype == str(tensor.dtype).removeprefix("torch.")
    assert decoded.shape == tuple(tensor.shape)
    assert decoded.data == _raw_bytes(tensor)
    assert restored.dtype == tensor.dtype
    assert restored.shape == tensor.shape
    assert _raw_bytes(restored) == _raw_bytes(tensor)
    assert spec.fingerprint(tensor) == spec.fingerprint(decoded)
    assert materialize_latent(tensor) is tensor


SIGMAS = TypeExpr.concrete(SIGMAS_TYPE)
STRING = TypeExpr.concrete(CORE_STRING)


class SigmaSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sigma-source.source",
            inputs=(InputSpec("case", STRING),),
            outputs=(OutputSpec("sigmas", SIGMAS),),
        )

    @classmethod
    def execute(cls, *, case: str) -> dict[str, object]:
        return cls.outputs(sigmas=_sigma_case(case))


class SigmaConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sigma-consumer.consumer",
            inputs=(InputSpec("sigmas", SIGMAS),),
            outputs=(OutputSpec("summary", STRING),),
        )

    @classmethod
    def execute(cls, *, sigmas: Any) -> dict[str, object]:
        tensor = materialize_latent(sigmas)
        return cls.outputs(
            summary=(
                f"{tensor.dtype};{tuple(tensor.shape)};"
                f"{hashlib.sha256(_raw_bytes(tensor)).hexdigest()}"
            )
        )


@pytest.mark.parametrize("case", ("float64", "noncontiguous-float32", "bfloat16", "empty"))
@pytest.mark.parametrize("in_process", (True,))
def test_ordinary_manifest_sigma_producer_consumer_repeats_on_both_hosts(
    tmp_path: Path, case: str, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            module = "sigma_fixture_" + case.replace("-", "_") + str(in_process)
            (tmp_path / f"{module}.py").write_text(
                "from test_sampling_wire_torch import SigmaConsumer, SigmaSource\n"
                "from dinkster_inference_wire import register_inference_types\n"
                "SOURCE_NODES = [SigmaSource]\n"
                "CONSUMER_NODES = [SigmaConsumer]\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            consumer_manifest = tmp_path / "consumer-dinkster-pack.toml"
            source_manifest.write_text(
                '[pack]\nname = "sigma-source"\npublisher = "dinkster"\n'
                'namespaces = ["sigma-source"]\n'
                f'[pack.entry]\nnodes = "{module}:SOURCE_NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            consumer_manifest.write_text(
                '[pack]\nname = "sigma-consumer"\npublisher = "dinkster"\n'
                'namespaces = ["sigma-consumer"]\n'
                f'[pack.entry]\nnodes = "{module}:CONSUMER_NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            await composer.add_pack(PackSpec(manifest=consumer_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("sigma-source.source", {"case": case}),
                    "consumer": GraphNode(
                        "sigma-consumer.consumer", {"sigmas": Link("source", "sigmas")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            expected_tensor = _sigma_case(case)
            expected = (
                f"{expected_tensor.dtype};{tuple(expected_tensor.shape)};"
                f"{hashlib.sha256(_raw_bytes(expected_tensor)).hexdigest()}"
            )
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == expected
                if attempt:
                    assert set(result.cached) == {"source", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())


class RuntimeNoise:
    def __init__(self, seed: int, enabled: bool) -> None:
        self.seed = seed
        self.enabled = enabled

    def generate_noise(self, latent: dict[str, Any]) -> torch.Tensor:
        initialize_inference_process()
        from dinkster_inference import sample

        if not self.enabled:
            return sample.prepare_empty_noise(latent["samples"])
        return sample.prepare_noise(latent["samples"], self.seed, latent.get("batch_index"))


class RuntimeSamplingSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="runtime-sampling.source",
            inputs=(InputSpec("case", STRING),),
            outputs=(
                OutputSpec("sampler", TypeExpr.concrete(SAMPLER_TYPE)),
                OutputSpec("noise", TypeExpr.concrete(NOISE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, *, case: str) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference import samplers
        from dinkster_inference.k_diffusion import sampling

        custom, enabled, seed = case.split(":")
        if custom == "custom":
            # A closure must stay in the holding worker, not be serialized.
            delegate = sampling.sample_euler_ancestral

            def node_local_sampler(*args: Any, **kwargs: Any) -> Any:
                assert kwargs["eta"] == 0.375
                assert kwargs["s_noise"] == 1.25
                return delegate(*args, **kwargs)

            sampler = samplers.KSAMPLER(
                node_local_sampler, {"eta": 0.375, "s_noise": 1.25}, {"random": True}
            )
        else:
            sampler = samplers.ksampler("euler")
        return cls.outputs(sampler=sampler, noise=RuntimeNoise(int(seed), enabled == "enabled"))


class SamplingModel:
    def __init__(self) -> None:
        self.inner_model = SimpleNamespace(inner_model=SimpleNamespace(model_sampling=object()))

    def __call__(self, x: torch.Tensor, _sigma: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        return torch.full_like(x, 0.25)


class RuntimeSamplingConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="runtime-sampling.consume",
            inputs=(
                InputSpec("sampler", TypeExpr.concrete(SAMPLER_TYPE)),
                InputSpec("noise", TypeExpr.concrete(NOISE_TYPE)),
            ),
            outputs=(OutputSpec("summary", STRING),),
        )

    @classmethod
    def execute(cls, *, sampler: Any, noise: Any) -> dict[str, object]:
        latent = {"samples": torch.zeros((1, 1, 2, 3), dtype=torch.float32)}
        generated = noise.generate_noise(latent)
        before = generated.clone()
        options = sampler.extra_options.copy()
        inpaint_options = sampler.inpaint_options.copy()
        if options:
            assert sampler.sampler_function.__name__ == "node_local_sampler"
            assert options == {"eta": 0.375, "s_noise": 1.25}
            assert inpaint_options == {"random": True}
        else:
            assert sampler.sampler_function.__name__ == "sample_euler"
        sampled = sampler.sampler_function(
            SamplingModel(),
            generated.clone(),
            torch.tensor([1.5, 0.75]),
            extra_args={"seed": noise.seed},
            disable=True,
            **options,
        )
        assert sampler.extra_options == options
        assert sampler.inpaint_options == inpaint_options
        assert torch.equal(generated, before)
        assert torch.count_nonzero(latent["samples"]) == 0
        return cls.outputs(
            summary=json.dumps(
                {
                    "seed": noise.seed,
                    "noise": generated.flatten().tolist(),
                    "sampled": sampled.flatten().tolist(),
                    "custom": bool(options),
                }
            )
        )


@pytest.mark.parametrize("in_process", (True, False))
def test_runtime_sampler_and_noise_repeat_in_ordinary_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={"PYTHONPATH": str(Path(__file__).parent), "CUDA_VISIBLE_DEVICES": ""}
        )
        try:
            module = f"runtime_sampling_fixture_{in_process}"
            (tmp_path / f"{module}.py").write_text(
                "from test_sampling_wire_torch import (\n"
                "    RuntimeSamplingSource, RuntimeSamplingConsumer)\n"
                "from dinkster_inference_wire import register_inference_types\n"
                "NODES = [RuntimeSamplingSource, RuntimeSamplingConsumer]\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                '[pack]\nname = "runtime-sampling"\npublisher = "dinkster"\n'
                'namespaces = ["runtime-sampling"]\n'
                f'[pack.entry]\nnodes = "{module}:NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            for kind in ("builtin", "custom"):
                for enabled in ("enabled", "disabled"):
                    for seed in (739, 740):
                        graph = Graph(
                            nodes={
                                "source": GraphNode(
                                    "runtime-sampling.source", {"case": f"{kind}:{enabled}:{seed}"}
                                ),
                                "consumer": GraphNode(
                                    "runtime-sampling.consume",
                                    {
                                        "sampler": Link("source", "sampler"),
                                        "noise": Link("source", "noise"),
                                    },
                                ),
                            }
                        )
                        for attempt in range(2):
                            result = await engine.run(graph, ["consumer"])
                            summary = json.loads(result.outputs["consumer"]["summary"].resolve())
                            expected = torch.zeros(6)
                            if enabled == "enabled":
                                expected = torch.randn(
                                    6, generator=torch.Generator().manual_seed(seed)
                                )
                            assert summary["noise"] == expected.tolist()
                            assert summary["seed"] == seed
                            assert summary["custom"] == (kind == "custom")
                            if kind == "builtin":
                                torch.testing.assert_close(
                                    torch.tensor(summary["sampled"]), expected / 2 + 0.125
                                )
                            else:
                                sigma_up = 0.375 * math.sqrt(0.75**2 * (1.5**2 - 0.75**2) / 1.5**2)
                                sigma_down = math.sqrt(0.75**2 - sigma_up**2)
                                ancestral_noise = torch.randn(
                                    6, generator=torch.Generator().manual_seed(seed + 1)
                                )
                                torch.testing.assert_close(
                                    torch.tensor(summary["sampled"]),
                                    expected * (sigma_down / 1.5)
                                    + 0.25 * (1 - sigma_down / 1.5)
                                    + ancestral_noise * 1.25 * sigma_up,
                                )
                            if attempt:
                                assert set(result.cached) == {"source", "consumer"}
                            else:
                                assert not result.cached
        finally:
            await composer.close()

    asyncio.run(scenario())
