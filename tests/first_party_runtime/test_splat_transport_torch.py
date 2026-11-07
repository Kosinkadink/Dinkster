"""Portable SPLAT values through the public API and ordinary CPU hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    SPLAT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    materialize_splat,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link

from dinkster.compose import PackSpec, ServingComposer

_CASES = ("uniform", "nonuniform", "empty", "zero-counts", "float16", "bfloat16", "float64")


def _source(case: str) -> SimpleNamespace:
    dtype = getattr(torch, case) if case.endswith("16") or case == "float64" else torch.float32
    count = 0 if case == "empty" else 3
    fields = {
        "positions": torch.arange(18, dtype=dtype).reshape(2, 3, 3)[:, :count],
        "scales": torch.arange(101, 119, dtype=dtype).reshape(2, 3, 3)[:, :count],
        "rotations": torch.arange(201, 225, dtype=dtype).reshape(2, 3, 4)[:, :count],
        "opacities": torch.tensor([0.125, 0.25, 0.5, 0.75, 0.875, 1], dtype=dtype).reshape(2, 3, 1)[
            :, :count
        ],
        "sh": torch.arange(51, 123, dtype=dtype).reshape(2, 3, 4, 3)[:, :count],
        "counts": (
            torch.tensor([1, 3], dtype=torch.int32)
            if case == "nonuniform"
            else torch.zeros(2, dtype=torch.int64)
            if case == "zero-counts"
            else None
        ),
    }
    if case == "float64":
        fields["positions"][0, 0, 0] = 1 + 2**-40
    return SimpleNamespace(**fields)


def _check(value: Any, case: str) -> None:
    dtype = getattr(torch, case) if case.endswith("16") or case == "float64" else torch.float32
    count = 0 if case == "empty" else 3
    assert value.positions.shape == (2, count, 3)
    expected = {
        "positions": list(range(18)),
        "scales": list(range(101, 119)),
        "rotations": list(range(201, 225)),
        "opacities": [0.125, 0.25, 0.5, 0.75, 0.875, 1],
        "sh": list(range(51, 123)),
    }
    if case == "float64":
        expected["positions"][0] = 1 + 2**-40
    for key, numbers in expected.items():
        tensor = getattr(value, key)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == dtype and tensor.device.type == "cpu"
        assert tensor.flatten().tolist() == ([] if count == 0 else numbers)
    if case == "nonuniform":
        assert value.counts.dtype == torch.int32
        assert value.counts.tolist() == [1, 3]
    elif case == "zero-counts":
        assert value.counts.dtype == torch.int64
        assert value.counts.tolist() == [0, 0]
    else:
        assert value.counts is None


@pytest.mark.parametrize("case", _CASES)
def test_splat_materializer_preserves_live_storage_and_portable_values(case: str) -> None:
    source = _source(case)
    live = materialize_splat(source)
    for key, tensor in vars(source).items():
        assert getattr(live, key) is tensor
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(SPLAT_TYPE)
    portable = spec.decode(spec.encode(source))
    restored = materialize_splat(portable)
    _check(restored, case)
    _check(source, case)
    if case == "nonuniform":
        numpy_mapping = {
            key: tensor.numpy() if tensor is not None else None
            for key, tensor in vars(source).items()
        }
        _check(materialize_splat(numpy_mapping), case)


class SplatSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "splat-source.source",
            inputs=(InputSpec("case", TypeExpr.concrete(CORE_STRING)),),
            outputs=(OutputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
        )

    @classmethod
    def execute(cls, *, case: str) -> dict[str, object]:
        return cls.outputs(splat=_source(case))


class SplatConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "splat-consumer.consumer",
            inputs=(
                InputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),
                InputSpec("case", TypeExpr.concrete(CORE_STRING)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, splat: object, case: str) -> dict[str, object]:
        _check(materialize_splat(splat), case)
        return cls.outputs(summary=f"SPLAT:{case}:checked")


@pytest.mark.parametrize("in_process", [True, False])
def test_splat_crosses_ordinary_hosts_without_model_or_reference_execution(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        module = f"splat_transport_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_splat_transport_torch import SplatSource, SplatConsumer\n"
            "SOURCE_NODES = (SplatSource,)\nCONSUMER_NODES = (SplatConsumer,)\n",
            encoding="ascii",
        )
        manifests = [Path(__file__).resolve().parents[1] / "dinkster-pack.toml"]
        for name, nodes in (("splat-source", "SOURCE_NODES"), ("splat-consumer", "CONSUMER_NODES")):
            manifest = tmp_path / f"{name}.toml"
            manifest.write_text(
                f'[pack]\nname = "{name}"\npublisher = "dinkster"\nnamespaces = ["{name}"]\n'
                f'[pack.entry]\nnodes = "{module}:{nodes}"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            manifests.append(manifest)
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            }
        )
        try:
            for index, manifest in enumerate(manifests):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group=f"splat-transport-{index}",
                        group_manifests=(manifest,),
                    )
                )
            engine = composer.composition.make_engine(lambda _event: None)
            for case in _CASES:
                graph = Graph(
                    {
                        "source": GraphNode("splat-source.source", {"case": case}),
                        "consumer": GraphNode(
                            "splat-consumer.consumer",
                            {"splat": Link("source", "splat"), "case": case},
                        ),
                    }
                )
                for _attempt in range(2):
                    result = await engine.run(graph, ["consumer"])
                    assert (
                        result.outputs["consumer"]["summary"].resolve() == f"SPLAT:{case}:checked"
                    )
        finally:
            await composer.close()

    asyncio.run(scenario())
