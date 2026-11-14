from __future__ import annotations

import asyncio
import gc
import os
from pathlib import Path

from dinkster_graph import Graph, GraphNode, Link, TypedLiteral, validate
from dinkster_protocol import Invocation
from dinkster_schema import (
    InputSpec,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_values import (
    EncodedPayload,
    TypeRegistry,
    Value,
    ValueMeta,
    decode_image_array,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    prepare_image_array_encoding,
    register_core_types,
)
from dinkster_workers import BoundaryDiagnostic, IsolatedWorker

from tests.async_scenarios import run_scenario

TESTS_DIR = Path(__file__).parent
IMAGE = "IMAGE"
MASK = "MASK"


def _register_image_type(registry: TypeRegistry, type_id: str) -> None:
    registry.register(
        type_id,
        encode=encode_image_array,
        decode=decode_image_array,
        prepare_buffer_encoding=prepare_image_array_encoding,
        fingerprint=image_array_fingerprint(type_id),
        meta=image_array_meta,
    )


def test_restamped_encoded_buffer_keeps_shared_memory_alive() -> None:
    registry = TypeRegistry()
    registry.register(
        "compat.value", encode=lambda value: str(value).encode(), decode=lambda data: data.decode()
    )
    registry.register(
        "native.value", encode=lambda value: str(value).encode(), decode=lambda data: data.decode()
    )
    registry.register_type_equivalence(
        "compat.value", "native.value", provider_id="test.value-wire@1"
    )
    released: list[bool] = []
    source_payload = EncodedPayload.from_buffer(
        "compat.value",
        memoryview(b"shared bytes"),
        lambda data: data.decode(),
        "shm",
        lambda: released.append(True),
    )
    source = Value(
        type_id="compat.value",
        fingerprint="source-fingerprint",
        meta=ValueMeta(),
        payload=source_payload,
    )
    bridged = registry.bridge_equivalent(source, "native.value")

    del source, source_payload
    gc.collect()
    assert released == []
    assert isinstance(bridged.payload, EncodedPayload)
    with bridged.payload.borrow_data() as view:
        assert bytes(view) == b"shared bytes"
    assert released == []

    del bridged
    gc.collect()
    assert released == [True]


def test_graph_validation_uses_registered_generic_equivalence_for_links_and_literals() -> None:
    registry = TypeRegistry()
    registry.register("compat.value")
    registry.register("native.value")
    registry.register_type_equivalence(
        "compat.value", "native.value", provider_id="test.compat-wire@1"
    )
    source = NodeSchema("source", outputs=(OutputSpec("value", TypeExpr.concrete("compat.value")),))
    sink = NodeSchema("sink", inputs=(InputSpec("value", TypeExpr.concrete("native.value")),))
    schemas = {source.node_type: source, sink.node_type: sink}
    linked = Graph(
        {
            "source": GraphNode("source", {}),
            "sink": GraphNode("sink", {"value": Link("source", "value")}),
        }
    )
    stamped = Graph({"sink": GraphNode("sink", {"value": TypedLiteral("compat.value", "value")})})
    assert validate(linked, schemas, ["sink"], known_types=registry) == []
    assert validate(stamped, schemas, ["sink"], known_types=registry) == []


def _write_manifest(root: Path) -> Path:
    root.mkdir()
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "native-equivalence-test"\npublisher = "dinkster"\n'
        'type-namespaces = []\nnamespaces = ["dinkster"]\n\n'
        '[pack.entry]\nnodes = "type_equivalence_native_nodes:NODES"\n'
        'types = "type_equivalence_native_nodes:register_types"\n',
        encoding="utf-8",
    )
    return manifest


def test_exact_image_and_mask_atoms_cross_isolated_workers(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        _register_image_type(registry, IMAGE)
        _register_image_type(registry, MASK)
        diagnostics: list[BoundaryDiagnostic] = []
        extra_env = {
            "PYTHONPATH": os.pathsep.join((str(TESTS_DIR), os.environ.get("PYTHONPATH", "")))
        }
        first = IsolatedWorker(
            _write_manifest(tmp_path / "first"),
            registry,
            extra_env=extra_env,
            shm_threshold=1,
            on_diagnostic=diagnostics.append,
        )
        second = IsolatedWorker(
            _write_manifest(tmp_path / "second"),
            registry,
            extra_env=extra_env,
            shm_threshold=1,
            on_diagnostic=diagnostics.append,
        )
        await asyncio.gather(first.start(), second.start())
        try:
            for source, destination in ((first, second), (second, first)):
                for port, type_id, shape in (
                    ("image", IMAGE, "1x2x3x3"),
                    ("mask", MASK, "1x2x3"),
                ):
                    producer = f"dinkster.test.{port}_producer"
                    produced = await source.invoke(
                        Invocation(
                            invocation_id=f"produce-{port}",
                            node_id=f"produce-{port}",
                            node_type=producer,
                            inputs={},
                            effective_schema=source.schemas[producer],
                        )
                    )
                    assert produced.error is None
                    assert produced.outputs is not None
                    value = produced.outputs[port]
                    assert value.type_id == type_id
                    consumer = f"dinkster.test.{port}_consumer"
                    consumed = await destination.invoke(
                        Invocation(
                            invocation_id=f"consume-{port}",
                            node_id=f"consume-{port}",
                            node_type=consumer,
                            inputs={port: value},
                            effective_schema=destination.schemas[consumer],
                        )
                    )
                    assert consumed.error is None
                    assert consumed.outputs is not None
                    assert consumed.outputs["shape"].resolve() == shape
            crossings = [edge for diagnostic in diagnostics for edge in diagnostic.inputs]
            assert [edge.type_id for edge in crossings] == [IMAGE, MASK, IMAGE, MASK]
            assert all(edge.transport == "shm" and edge.reused for edge in crossings)
        finally:
            await asyncio.gather(first.close(), second.close())

    run_scenario(scenario())
