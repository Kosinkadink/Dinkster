"""CPU transport checks for structural hook strengths."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    FLOATS_TYPE,
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


class FloatStrengthSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-values-source.source",
            outputs=(OutputSpec("strengths", TypeExpr.concrete(FLOATS_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(strengths=torch.tensor([0.5, -3.25], dtype=torch.bfloat16))


class FloatStrengthConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="hook-values-consumer.consumer",
            inputs=(InputSpec("strengths", TypeExpr.concrete(FLOATS_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, strengths: Any) -> dict[str, object]:
        tensor = materialize_latent(strengths)
        return cls.outputs(
            summary=f"{tensor.dtype};{tuple(tensor.shape)};{hashlib.sha256(_raw_bytes(tensor)).hexdigest()}"
        )


def test_floats_tensor_codec_preserves_dtype_and_bytes() -> None:
    source = torch.tensor([0.5, -3.25], dtype=torch.bfloat16)
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(FLOATS_TYPE)
    decoded = spec.decode(spec.encode(source))
    restored = materialize_latent(decoded)

    assert restored.dtype == source.dtype
    assert restored.shape == source.shape
    assert _raw_bytes(restored) == _raw_bytes(source)


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_transports_float_strengths_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            module = "hook_values_fixture_" + str(in_process)
            (tmp_path / f"{module}.py").write_text(
                "from test_hook_values_torch import FloatStrengthConsumer, FloatStrengthSource\n"
                "from dinkster_inference_wire import register_inference_types\n"
                "SOURCE_NODES = [FloatStrengthSource]\n"
                "CONSUMER_NODES = [FloatStrengthConsumer]\n",
                encoding="ascii",
            )
            source_manifest = tmp_path / "dinkster-pack.toml"
            consumer_manifest = tmp_path / "consumer-dinkster-pack.toml"
            source_manifest.write_text(
                '[pack]\nname = "hook-values-source"\npublisher = "dinkster"\n'
                'namespaces = ["hook-values-source"]\n'
                f'[pack.entry]\nnodes = "{module}:SOURCE_NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            consumer_manifest.write_text(
                '[pack]\nname = "hook-values-consumer"\npublisher = "dinkster"\n'
                'namespaces = ["hook-values-consumer"]\n'
                f'[pack.entry]\nnodes = "{module}:CONSUMER_NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=source_manifest, in_process=in_process))
            await composer.add_pack(PackSpec(manifest=consumer_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("hook-values-source.source", {}),
                    "consumer": GraphNode(
                        "hook-values-consumer.consumer", {"strengths": Link("source", "strengths")}
                    ),
                }
            )
            expected_tensor = torch.tensor([0.5, -3.25], dtype=torch.bfloat16)
            expected = (
                f"{expected_tensor.dtype};{tuple(expected_tensor.shape)};"
                f"{hashlib.sha256(_raw_bytes(expected_tensor)).hexdigest()}"
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == expected
                if attempt:
                    assert set(result.cached) == {"source", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
