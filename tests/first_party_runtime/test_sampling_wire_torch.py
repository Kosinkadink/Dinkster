"""CPU coverage for native SIGMAS transport through ordinary pack hosting."""

from __future__ import annotations

import hashlib
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
from dinkster_inference_wire import register_inference_types
from dinkster_values import TypeRegistry


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
