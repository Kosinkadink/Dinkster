from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral, validate
from dinkster_nodes_foundation import AddInts, EntryFanOut, MathExpressions
from dinkster_protocol import Invocation
from dinkster_schema import (
    ElaborationError,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputDescriptorsSpec,
    OutputInterface,
    OutputProbeSpec,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    elaborate,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)
from dinkster_server import create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker, diagnose, load_manifest
from dinkster_workers.catalog import read_catalog

from dinkster.compose import PackSpec, ServingComposer
from dinkster.lazy_worker import LazyWorker

INT = TypeExpr.concrete("core.int")
STRING = TypeExpr.concrete("core.string")


def document(*entries: Mapping[str, object]) -> str:
    return json.dumps({"entries": entries})


@pytest.fixture
def runtime() -> tuple[Engine, InProcessWorker, TypeRegistry]:
    registry = TypeRegistry()
    register_core_types(registry)
    nodes = [MathExpressions, EntryFanOut, AddInts]
    worker = InProcessWorker(build_node_types(nodes), registry)
    engine = Engine(
        schemas=build_schemas(nodes), registry=registry, worker=worker, cache=MemoryLRUCache()
    )
    return engine, worker, registry


def test_wire_and_effective_interface_roundtrip() -> None:
    schema = MathExpressions.schema()
    assert schema_from_wire(schema_to_wire(schema)) == schema
    value = document(
        {"id": "stable", "name": "Answer", "type": "int", "expression": "2+3"},
        {"id": "other", "name": "Ready", "type": "boolean", "expression": "1"},
    )
    effective = elaborate(schema, {"entries": value})
    assert [(out.id, out.display_name, out.type) for out in effective.outputs] == [
        ("stable", "Answer", INT),
        ("other", "Ready", TypeExpr.concrete("core.boolean")),
    ]
    assert schema_from_wire(schema_to_wire(effective)) == effective
    assert effective.is_static
    assert schema_signature(schema)
    with pytest.raises(ValueError, match="39"):
        schema_to_wire(schema, wire_version=38)
    old = schema_to_wire(schema)
    old["schemaVersion"] = 38
    with pytest.raises(ValueError, match="39"):
        schema_from_wire(old)


@pytest.mark.parametrize(
    "value",
    [
        "[]",
        "null",
        "not json",
        '{"entries":{}}',
        '{"entries":[1]}',
        '{"entries":[],"value":NaN}',
        '{"entries":[],"value":Infinity}',
        pytest.param(" " * 1_048_577, id="oversized-whitespace"),
        document({"id": "x", "name": "x" * 257, "type": "int"}),
        document({"id": "bad.id", "name": "Result", "type": "int"}),
        document({"id": "x", "name": " ", "type": "int"}),
        document({"id": "x", "name": "Result", "type": "unknown"}),
        document({"id": "x", "name": "A", "type": "int"}, {"id": "x", "name": "B", "type": "int"}),
        document({"id": "x", "name": "A", "type": "int"}, {"id": "y", "name": "A", "type": "int"}),
        Link("producer", "text"),
        TypedLiteral("core.string", '{"entries":[]}'),
    ],
)
def test_malformed_descriptors_fail_before_execution(value: object) -> None:
    with pytest.raises(ElaborationError):
        elaborate(MathExpressions.schema(), {"entries": value})


def test_descriptor_bounds_collisions_and_fixed_semantics() -> None:
    source = InputSpec("entries", STRING)
    schema = NodeSchema(
        "test.fixed",
        inputs=(source,),
        output_descriptors=OutputDescriptorsSpec(
            "entries", (OutputSpec("model", INT),), 1, 1, True
        ),
    )
    with pytest.raises(ElaborationError, match="bounds"):
        elaborate(schema, {"entries": document()})
    with pytest.raises(ElaborationError, match="semantic"):
        elaborate(schema, {"entries": document({"id": "clip", "name": "Clip", "type": "model"})})
    collision = replace(schema, outputs=(OutputSpec("model", INT),))
    with pytest.raises(ElaborationError, match="collides"):
        elaborate(
            collision, {"entries": document({"id": "model", "name": "Model", "type": "model"})}
        )


def test_descriptor_schema_constraints_and_presentation_identity() -> None:
    schema = MathExpressions.schema()
    spec = schema.output_descriptors
    assert spec is not None
    with pytest.raises(ValueError, match="concrete"):
        replace(spec, choices=(OutputSpec("any", TypeExpr.variable("T")),))
    with pytest.raises(ValueError, match="concrete"):
        replace(spec, choices=(OutputSpec("list", TypeExpr.list_of(INT)),))
    with pytest.raises(ValueError, match="unique"):
        replace(spec, choices=(spec.choices[0], spec.choices[0]))
    with pytest.raises(ValueError, match="1-512"):
        replace(spec, choices=())
    with pytest.raises(ValueError, match="bounds"):
        replace(spec, max_entries=513)
    with pytest.raises(ValueError, match="required top-level"):
        replace(schema, inputs=(InputSpec("entries", STRING, required=False),))
    with pytest.raises(ValueError, match="fixed semantic"):
        replace(spec, probe=OutputProbeSpec("asset", "model", "1"))
    presentation = replace(
        schema,
        output_descriptors=replace(
            spec, choices=tuple(replace(choice, display_name="Label") for choice in spec.choices)
        ),
    )
    assert schema_signature(presentation) == schema_signature(schema)
    assert schema_from_wire(schema_to_wire(presentation)) == presentation


def test_expression_executes_links_and_cache_identity(
    runtime: tuple[Engine, InProcessWorker, TypeRegistry],
) -> None:
    engine, _, _ = runtime

    async def scenario() -> None:
        entry = {"id": "stable", "name": "Answer", "type": "int", "expression": "2+3"}

        def graph() -> Graph:
            return Graph(
                nodes={
                    "expression": GraphNode(
                        "dinkster.math.expressions", {"entries": document(entry)}
                    ),
                    "sum": GraphNode(
                        "std.math.add_ints", {"a": Link("expression", "stable"), "b": 7}
                    ),
                }
            )

        first = await engine.run(graph(), ["sum"])
        assert first.outputs["sum"]["sum"].resolve() == 12
        second = await engine.run(graph(), ["sum"])
        assert "expression" in second.cached
        entry["name"] = "Renamed"
        renamed = await engine.run(graph(), ["sum"])
        assert "expression" in renamed.executed
        assert renamed.outputs["sum"]["sum"].resolve() == 12
        entry["id"] = "new_identity"
        diagnostics = validate(graph(), build_schemas([MathExpressions, AddInts]), ["sum"])
        assert any(item.severity == "error" for item in diagnostics)

    asyncio.run(scenario())


def test_entry_list_fan_out(runtime: tuple[Engine, InProcessWorker, TypeRegistry]) -> None:
    engine, _, _ = runtime

    async def scenario() -> None:
        graph = Graph(
            nodes={
                "fan": GraphNode(
                    "dinkster.list.fan_out",
                    {
                        "entries": document(
                            {"id": "a", "name": "Count", "type": "int", "value": 3},
                            {"id": "b", "name": "Label", "type": "string", "value": "ready"},
                        )
                    },
                )
            }
        )
        result = await engine.run(graph, ["fan"])
        assert {key: value.resolve() for key, value in result.outputs["fan"].items()} == {
            "a": 3,
            "b": "ready",
        }

    asyncio.run(scenario())


def test_entry_list_fan_out_interface_depends_only_on_stored_descriptors(
    runtime: tuple[Engine, InProcessWorker, TypeRegistry],
) -> None:
    engine, _, _ = runtime
    selected = {"id": "selected", "name": "Selected", "type": "int", "index": 1}

    def graph(entries: str, items: list[int]) -> Graph:
        return Graph(
            nodes={
                "fan": GraphNode(
                    "dinkster.list.fan_out",
                    {"entries": entries, "items": TypedLiteral("list<core.int>", items)},
                ),
                "sum": GraphNode(
                    "std.math.add_ints",
                    {"a": Link("fan", "selected"), "b": 7},
                ),
            }
        )

    async def scenario() -> None:
        one_output = document(selected)
        short = await engine.run(graph(one_output, [10, 20]), ["fan", "sum"])
        long = await engine.run(graph(one_output, [0, 20, 30, 40]), ["fan", "sum"])
        assert {key: value.resolve() for key, value in short.outputs["fan"].items()} == {
            "selected": 20
        }
        assert {key: value.resolve() for key, value in long.outputs["fan"].items()} == {
            "selected": 20
        }
        assert short.outputs["sum"]["sum"].resolve() == 27
        assert long.outputs["sum"]["sum"].resolve() == 27

        two_outputs = document(
            selected,
            {"id": "first", "name": "First", "type": "int", "index": 0},
        )
        expanded = await engine.run(graph(two_outputs, [5, 20, 30]), ["fan", "sum"])
        assert {key: value.resolve() for key, value in expanded.outputs["fan"].items()} == {
            "selected": 20,
            "first": 5,
        }

        removed = graph(
            document({"id": "first", "name": "First", "type": "int", "index": 0}),
            [5, 20, 30],
        )
        diagnostics = validate(removed, build_schemas([EntryFanOut, AddInts]), ["sum"])
        assert "dangling-output" in {item.code for item in diagnostics}

    asyncio.run(scenario())


@pytest.mark.parametrize("changed", ["order", "name", "type", "id"])
def test_worker_reprojects_descriptors(
    runtime: tuple[Engine, InProcessWorker, TypeRegistry], changed: str
) -> None:
    _, worker, registry = runtime
    value = document(
        {"id": "a", "name": "A", "type": "int", "value": 1},
        {"id": "b", "name": "B", "type": "int", "value": 2},
    )
    effective = elaborate(EntryFanOut.schema(), {"entries": value})
    outputs = list(effective.outputs)
    if changed == "order":
        outputs.reverse()
    elif changed == "name":
        outputs[0] = replace(outputs[0], display_name="Different")
    elif changed == "type":
        outputs[0] = replace(outputs[0], type=STRING)
    else:
        outputs[0] = replace(outputs[0], id="different")
    result = asyncio.run(
        worker.invoke(
            Invocation(
                invocation_id="test",
                node_id="fan",
                node_type="dinkster.list.fan_out",
                inputs={"entries": registry.wrap("core.string", value)},
                effective_schema=replace(effective, outputs=tuple(outputs)),
            )
        )
    )
    assert result.error is not None
    assert "worker projection" in result.error.message


@pytest.mark.parametrize("in_process", [True, False])
def test_persisted_descriptors_start_only_on_execution(tmp_path: Path, in_process: bool) -> None:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "catalogdescriptors"\n[pack.sandbox]\n'
        '[pack.entry]\nnodes = "descriptor_nodes:NODES"\n'
    )
    (tmp_path / "descriptor_nodes.py").write_text(
        "from dinkster_api.v1 import (Node, NodeSchema, InputSpec, OutputSpec, "
        "OutputDescriptorsSpec, TypeExpr, output_descriptor_entries)\n"
        "class Expressions(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema('catalogdescriptors.math',\n"
        "            inputs=(InputSpec('entries', TypeExpr.concrete('core.string')),),\n"
        "            output_descriptors=OutputDescriptorsSpec('entries',\n"
        "                (OutputSpec('int', TypeExpr.concrete('core.int')),), 32))\n"
        "    @classmethod\n"
        "    def execute(cls, *, entries, output_spec):\n"
        "        return {entry['id']: entry['value'] "
        "for entry in output_descriptor_entries(entries)}\n"
        "NODES = [Expressions]\n"
    )
    report = diagnose(manifest)
    assert report.ok, report
    assert read_catalog(load_manifest(manifest)) is not None

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(
                PackSpec(
                    manifest,
                    in_process=in_process,
                    env={"PYTHONPATH": str(tmp_path)},
                    require_catalog=True,
                )
            )
            worker = next(iter(composer._records.values())).worker
            assert isinstance(worker, LazyWorker) and worker.cold and not worker.alive
            composition = composer.composition
            app = create_app(composition.make_engine, composition.schemas)
            async with TestClient(TestServer(app)) as client:
                response = await client.get("/api/nodes?wire=39")
                assert response.status == 200
                assert '"outputDescriptors"' in await response.text()
                assert worker.cold and not worker.alive
            engine = composition.make_engine(lambda event: None)
            result = await engine.run(
                Graph(
                    nodes={
                        "math": GraphNode(
                            "catalogdescriptors.math",
                            {
                                "entries": document(
                                    {
                                        "id": "answer",
                                        "name": "Answer",
                                        "type": "int",
                                        "value": 42,
                                    }
                                )
                            },
                        )
                    }
                ),
                ["math"],
            )
            assert result.outputs["math"]["answer"].resolve() == 42
            assert worker.alive
        finally:
            await composer.close()
            sys.modules.pop("descriptor_nodes", None)

    asyncio.run(scenario())


def test_descriptor_projection_retains_undemanded_lazy_family_members() -> None:
    class LazyDescriptors(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                "test.lazy_descriptors",
                inputs=(InputSpec("entries", STRING),),
                input_families=(
                    InputFamilySpec("values", (InputSpec("value", INT, lazy=True),), min_members=1),
                ),
                output_descriptors=OutputDescriptorsSpec("entries", (OutputSpec("int", INT),), 1),
            )

        @classmethod
        def check_lazy_status(cls, **kwargs: object) -> list[str]:
            return []

        @classmethod
        def execute(
            cls, *, entries: str, values: Mapping[str, object], output_spec: OutputInterface
        ) -> Mapping[str, object]:
            return {output_spec.outputs[0].id: 42}

    registry = TypeRegistry()
    register_core_types(registry)
    nodes = [LazyDescriptors, AddInts]
    worker = InProcessWorker(build_node_types(nodes), registry)
    engine = Engine(
        schemas=build_schemas(nodes), registry=registry, worker=worker, cache=MemoryLRUCache()
    )
    graph = Graph(
        nodes={
            "producer": GraphNode("std.math.add_ints", {"a": 1, "b": 2}),
            "lazy": GraphNode(
                "test.lazy_descriptors",
                {
                    "entries": document({"id": "answer", "name": "Answer", "type": "int"}),
                    "values.a": Link("producer", "sum"),
                },
            ),
        }
    )
    result = asyncio.run(engine.run(graph, ["lazy"]))
    assert result.outputs["lazy"]["answer"].resolve() == 42
    assert "producer" not in result.executed
