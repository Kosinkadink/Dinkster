"""CPU coverage for native SIGMAS transport through ordinary pack hosting."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    SIGMAS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
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
def test_ordinary_manifest_sigma_producer_consumer_repeats_on_both_hosts(
    tmp_path: Path, case: str
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        repository_root = Path(__file__).resolve().parents[3]
        package_sources = tuple((repository_root / "packages").glob("*/src"))
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(Path(__file__).parent),
                        *(str(path) for path in package_sources),
                        str(source_root),
                    )
                ),
            }
        )
        try:
            (tmp_path / "sigma_fixture.py").write_text(
                "from test_sampling_wire_torch import SigmaConsumer, SigmaSource\n"
                "from dinkster_inference_wire import register_inference_types\n"
                "SOURCE_NODES = [SigmaSource]\n"
                "CONSUMER_NODES = [SigmaConsumer]\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            consumer_manifest = tmp_path / "consumer-dinkster-pack.toml"
            source_manifest.write_text(
                '[pack]\nname = "sigma-source"\nnamespaces = ["sigma-source"]\n'
                '[pack.entry]\nnodes = "sigma_fixture:SOURCE_NODES"\n'
                'types = "sigma_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            consumer_manifest.write_text(
                '[pack]\nname = "sigma-consumer"\nnamespaces = ["sigma-consumer"]\n'
                '[pack.entry]\nnodes = "sigma_fixture:CONSUMER_NODES"\n'
                'types = "sigma_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest))
            await composer.add_pack(PackSpec(manifest=consumer_manifest))
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
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == expected
        finally:
            await composer.close()

    asyncio.run(scenario())
