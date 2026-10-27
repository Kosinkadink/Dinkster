"""CPU transport coverage for public portable LORA_MODEL values."""

from __future__ import annotations

import asyncio
import os
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LORA_MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link

from dinkster.compose import PackSpec, ServingComposer

LORA_MODEL = TypeExpr.concrete(LORA_MODEL_TYPE)
STRING = TypeExpr.concrete(CORE_STRING)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return bytes(value.contiguous().reshape(-1).view(torch.uint8).tolist())


def _lora_model() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (
            ("lora_te.alpha", torch.tensor(1.5, dtype=torch.float64)),
            (
                "lora_unet.down.weight",
                torch.arange(24, dtype=torch.float16).reshape(2, 3, 4).transpose(0, 2),
            ),
            ("lora_unet.up.weight", torch.empty((2, 0, 3), dtype=torch.bfloat16)),
        )
    )


def _assert_lora_model(value: object) -> None:
    expected = _lora_model()
    assert isinstance(value, Mapping)
    assert tuple(value) == tuple(expected)
    for key, source in expected.items():
        received = value[key]
        assert type(received) is torch.Tensor
        assert received.dtype is source.dtype
        assert received.shape == source.shape
        assert received.device.type == "cpu"
        assert _tensor_bytes(received) == _tensor_bytes(source)


class LoraModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="lora-model-public.source",
            outputs=(OutputSpec("lora", LORA_MODEL),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(lora=_lora_model())


class LoraModelConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="lora-model-public.consumer",
            inputs=(InputSpec("lora", LORA_MODEL),),
            outputs=(OutputSpec("keys", STRING),),
        )

    @classmethod
    def execute(cls, *, lora: object) -> dict[str, object]:
        restored = materialize_latent(lora)
        _assert_lora_model(restored)
        return cls.outputs(keys=",".join(restored))


class IsolatedLoraModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="lora-model-source.source",
            outputs=(OutputSpec("lora", LORA_MODEL),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(lora=_lora_model())


class IsolatedLoraModelConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="lora-model-consumer.consumer",
            inputs=(InputSpec("lora", LORA_MODEL),),
            outputs=(OutputSpec("keys", STRING),),
        )

    @classmethod
    def execute(cls, *, lora: object) -> dict[str, object]:
        restored = materialize_latent(lora)
        assert type(restored) is dict
        _assert_lora_model(restored)
        return cls.outputs(keys=",".join(restored))


@pytest.mark.parametrize("in_process", (True, False))
def test_lora_model_crosses_native_and_public_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        native_manifest = root / "dinkster-pack.toml"
        native_manifest_bytes = native_manifest.read_bytes()
        module = "lora_model_public_fixture"
        (tmp_path / f"{module}.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_lora_model_type_torch import LoraModelConsumer, LoraModelSource\n"
            "NODES = (LoraModelSource, LoraModelConsumer)\n",
            encoding="ascii",
        )
        fixture_manifest = tmp_path / "lora-model-public-dinkster-pack.toml"
        fixture_manifest.write_text(
            "[pack]\n"
            'name = "lora-model-public-fixture"\n'
            'publisher = "dinkster"\n'
            'namespaces = ["lora-model-public"]\n\n'
            "[pack.entry]\n"
            f'nodes = "{module}:NODES"\n'
            f'types = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        manifests = (native_manifest, fixture_manifest)
        composer = ServingComposer(
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            },
        )
        try:
            specs = (
                tuple(PackSpec(manifest=manifest, in_process=True) for manifest in manifests)
                if in_process
                else tuple(
                    PackSpec(
                        manifest=manifest,
                        worker_group="lora-model-public",
                        group_manifests=manifests,
                    )
                    for manifest in manifests
                )
            )
            for spec in specs:
                await composer.add_pack(spec)
            graph = Graph(
                nodes={
                    "source": GraphNode("lora-model-public.source"),
                    "consumer": GraphNode(
                        "lora-model-public.consumer", {"lora": Link("source", "lora")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            expected = ",".join(_lora_model())
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["keys"].resolve() == expected
                assert set(result.cached) == ({"source", "consumer"} if attempt else set())
        finally:
            await composer.close()
        assert native_manifest.read_bytes() == native_manifest_bytes

    asyncio.run(scenario())


def test_lora_model_crosses_isolated_producer_and_consumer_hosts_twice(tmp_path: Path) -> None:
    async def scenario() -> None:
        source_module = "lora_model_isolated_source"
        consumer_module = "lora_model_isolated_consumer"
        (tmp_path / f"{source_module}.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_lora_model_type_torch import IsolatedLoraModelSource\n"
            "NODES = (IsolatedLoraModelSource,)\n",
            encoding="ascii",
        )
        (tmp_path / f"{consumer_module}.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_lora_model_type_torch import IsolatedLoraModelConsumer\n"
            "NODES = (IsolatedLoraModelConsumer,)\n",
            encoding="ascii",
        )

        def write_manifest(name: str, module: str, namespace: str) -> Path:
            manifest = tmp_path / f"{name}-dinkster-pack.toml"
            manifest.write_text(
                "[pack]\n"
                f'name = "{name}"\n'
                'publisher = "dinkster"\n'
                f'namespaces = ["{namespace}"]\n\n'
                "[pack.entry]\n"
                f'nodes = "{module}:NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            return manifest

        producer = write_manifest(
            "lora-model-isolated-producer", source_module, "lora-model-source"
        )
        consumer = write_manifest(
            "lora-model-isolated-consumer", consumer_module, "lora-model-consumer"
        )
        composer = ServingComposer(
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            },
        )
        try:
            await composer.add_pack(
                PackSpec(
                    manifest=producer,
                    worker_group="lora-model-producer",
                    group_manifests=(producer,),
                )
            )
            await composer.add_pack(
                PackSpec(
                    manifest=consumer,
                    worker_group="lora-model-consumer",
                    group_manifests=(consumer,),
                )
            )
            graph = Graph(
                nodes={
                    "source": GraphNode("lora-model-source.source"),
                    "consumer": GraphNode(
                        "lora-model-consumer.consumer", {"lora": Link("source", "lora")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            expected = ",".join(_lora_model())
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["keys"].resolve() == expected
                assert set(result.cached) == ({"source", "consumer"} if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
