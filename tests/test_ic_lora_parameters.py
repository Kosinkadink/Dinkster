"""Portable IC-LoRA parameter mapping contracts."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from dinkster_api.v1 import (
    CORE_STRING,
    IC_LORA_PARAMETERS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_values import EncodedLatentTensor, encode_latent

from dinkster.compose import PackSpec, ServingComposer


def test_ic_lora_parameters_roundtrip_preserves_mapping_order_and_extras() -> None:
    value = {
        "reference_downscale_factor": 2,
        "nested": {"keep": ("source", 3)},
        "tensor": EncodedLatentTensor("float16", (1, 2), b"\x00\x3c\x00\xc0"),
    }
    registry = TypeRegistry()
    registered = register_inference_types(registry)
    spec = registry.spec(IC_LORA_PARAMETERS_TYPE)

    assert spec in registered
    assert register_inference_types(registry) == registered
    encoded = spec.encode(value)
    assert spec.decode(encoded) == value
    assert list(cast("Mapping[str, object]", spec.decode(encoded))) == list(value)
    assert spec.fingerprint is not None
    assert spec.fingerprint(value) == spec.fingerprint(spec.decode(encoded))
    assert spec.fingerprint(value) != spec.fingerprint({**value, "extra": True})


def test_ic_lora_parameters_rejects_non_mapping_and_invalid_frames() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(IC_LORA_PARAMETERS_TYPE)
    assert spec.validate_encoded is not None

    with pytest.raises(ValueError):
        spec.encode(("reference_downscale_factor", 2))
    payload = encode_latent({"samples": {"reference_downscale_factor": 1}, "extra": 1})
    with pytest.raises(ValueError):
        spec.decode(payload)
    with pytest.raises(ValueError):
        spec.validate_encoded(payload, {})


@pytest.mark.parametrize("entry", ("ab", 12, None, {"a": 1}, ("a",), ("a", 1, 2)))
def test_ic_lora_parameters_rejects_malformed_mapping_entries(entry: object) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(IC_LORA_PARAMETERS_TYPE)
    assert spec.validate_encoded is not None
    payload = encode_latent({"samples": [entry]})
    with pytest.raises(ValueError, match="string-keyed pairs"):
        spec.decode(payload)
    with pytest.raises(ValueError, match="string-keyed pairs"):
        spec.validate_encoded(payload, {})


def test_ic_lora_parameters_public_registration_is_cold_and_idempotent() -> None:
    script = "\n".join(
        (
            "import sys",
            "from dinkster_api.v1 import (",
            "    IC_LORA_PARAMETERS_TYPE, TypeRegistry, register_inference_types,",
            ")",
            "registry = TypeRegistry()",
            "first = register_inference_types(registry)",
            "second = register_inference_types(registry)",
            "assert first == second",
            "spec = registry.spec(IC_LORA_PARAMETERS_TYPE)",
            "value = {'reference_downscale_factor': 2, 'extra': {'preserve': True}}",
            "assert spec.decode(spec.encode(value)) == value",
            "blocked = ('torch', 'dinkster_inference')",
            "assert not any(name.split('.')[0] in blocked for name in sys.modules)",
        )
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        check=True,
    )


class ICLoRAParametersSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ic-lora-source.node",
            outputs=(OutputSpec("parameters", TypeExpr.concrete(IC_LORA_PARAMETERS_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return dict(
            cls.outputs(
                parameters={
                    "reference_downscale_factor": 3,
                    "source_only": {"ordered": ("first", 2), "enabled": True},
                    "extra": ["preserve", {"nested": None}],
                }
            )
        )


class ICLoRAParametersConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ic-lora-consumer.node",
            inputs=(InputSpec("parameters", TypeExpr.concrete(IC_LORA_PARAMETERS_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, parameters: object) -> dict[str, object]:
        mapping = cast("Mapping[str, object]", parameters)
        assert parameters == {
            "reference_downscale_factor": 3,
            "source_only": {"ordered": ("first", 2), "enabled": True},
            "extra": ["preserve", {"nested": None}],
        }
        assert list(mapping) == ["reference_downscale_factor", "source_only", "extra"]
        return dict(cls.outputs(summary="IC-LoRA parameters preserved"))


@pytest.mark.parametrize("in_process", (True, False))
def test_ic_lora_parameters_cross_hosts_cache_preserves_mapping(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        module = "ic_lora_parameters_fixture"
        source_root = Path(__file__).resolve().parents[1] / "packages"
        (tmp_path / f"{module}.py").write_text(
            "from test_ic_lora_parameters import ICLoRAParametersConsumer, ICLoRAParametersSource\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "SOURCE_NODES = (ICLoRAParametersSource,)\n"
            "CONSUMER_NODES = (ICLoRAParametersConsumer,)\n",
            encoding="ascii",
        )
        source_manifest = tmp_path / "source-dinkster-pack.toml"
        consumer_manifest = tmp_path / "consumer-dinkster-pack.toml"
        for manifest, name, namespace, nodes in (
            (
                source_manifest,
                "ic-lora-parameters-source",
                "ic-lora-source",
                "SOURCE_NODES",
            ),
            (
                consumer_manifest,
                "ic-lora-parameters-consumer",
                "ic-lora-consumer",
                "CONSUMER_NODES",
            ),
        ):
            manifest.write_text(
                f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
                f'namespaces = ["{namespace}"]\n\n[pack.entry]\n'
                f'nodes = "{module}:{nodes}"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
        composer = ServingComposer(
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(source_root))
                ),
            },
        )
        try:
            if in_process:
                specs = (
                    PackSpec(manifest=source_manifest, in_process=True),
                    PackSpec(manifest=consumer_manifest, in_process=True),
                )
            else:
                specs = (
                    PackSpec(
                        manifest=source_manifest,
                        worker_group="ic-lora-producer",
                        group_manifests=(source_manifest,),
                    ),
                    PackSpec(
                        manifest=consumer_manifest,
                        worker_group="ic-lora-consumer",
                        group_manifests=(consumer_manifest,),
                    ),
                )
            for spec in specs:
                await composer.add_pack(spec)
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode("ic-lora-source.node", {}),
                    "consumer": GraphNode(
                        "ic-lora-consumer.node",
                        {"parameters": Link("source", "parameters")},
                    ),
                }
            )
            for repeat in range(2):
                result = await engine.run(graph, ["consumer"])
                assert (
                    result.outputs["consumer"]["summary"].resolve()
                    == "IC-LoRA parameters preserved"
                )
                assert set(result.cached if repeat else result.executed) == {"source", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
