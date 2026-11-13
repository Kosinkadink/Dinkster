from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral, validate
from dinkster_schema import (
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputDescriptorsSpec,
    OutputInterface,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
    output_descriptor_entries,
)
from dinkster_server import create_app
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker, diagnose, load_manifest
from dinkster_workers.catalog import read_catalog

from dinkster.compose import PackSpec, ServingComposer
from dinkster.lazy_worker import LazyWorker
from tests.async_scenarios import run_scenario

INT = TypeExpr.concrete("INT")
STRING = TypeExpr.concrete("STRING")


class DeclaredValues(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "test.declared_values",
            inputs=(InputSpec("entries", STRING),),
            output_descriptors=OutputDescriptorsSpec(
                "entries",
                (OutputSpec("int", INT), OutputSpec("boolean", TypeExpr.concrete("BOOLEAN"))),
                32,
            ),
        )

    @classmethod
    def execute(cls, *, entries: str, output_spec: OutputInterface) -> Mapping[str, object]:
        return {
            output.id: entry["value"]
            for output, entry in zip(
                output_spec.outputs, output_descriptor_entries(entries), strict=True
            )
        }


class EntryFanOut(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "test.entry_fan_out",
            inputs=(
                InputSpec("entries", STRING),
                InputSpec("items", TypeExpr.list_of(TypeExpr.variable("T")), required=False),
            ),
            output_descriptors=OutputDescriptorsSpec(
                "entries", (OutputSpec("int", INT), OutputSpec("string", STRING)), 512
            ),
        )

    @classmethod
    def execute(
        cls, *, entries: str, output_spec: OutputInterface, items: Sequence[object] = ()
    ) -> Mapping[str, object]:
        return {
            output.id: items[cast(int, entry["index"])] if "index" in entry else entry["value"]
            for output, entry in zip(
                output_spec.outputs, output_descriptor_entries(entries), strict=True
            )
        }


class AddInts(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "test.add_ints",
            inputs=(InputSpec("a", INT), InputSpec("b", INT)),
            outputs=(OutputSpec("sum", INT),),
        )

    @classmethod
    def execute(cls, *, a: int, b: int) -> Mapping[str, object]:
        return {"sum": a + b}


def document(*entries: Mapping[str, object]) -> str:
    return json.dumps({"entries": entries})


@pytest.fixture
def runtime() -> tuple[Engine, InProcessWorker, TypeRegistry]:
    registry = TypeRegistry()
    register_core_types(registry)
    nodes = [DeclaredValues, EntryFanOut, AddInts]
    worker = InProcessWorker(build_node_types(nodes), registry)
    engine = Engine(
        schemas=build_schemas(nodes), registry=registry, worker=worker, cache=MemoryLRUCache()
    )
    return engine, worker, registry


def test_declared_values_execute_links_and_cache_identity(
    runtime: tuple[Engine, InProcessWorker, TypeRegistry],
) -> None:
    engine, _, _ = runtime

    async def scenario() -> None:
        entry = {"id": "stable", "name": "Answer", "type": "int", "value": 5}

        def graph() -> Graph:
            return Graph(
                nodes={
                    "expression": GraphNode("test.declared_values", {"entries": document(entry)}),
                    "sum": GraphNode("test.add_ints", {"a": Link("expression", "stable"), "b": 7}),
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
        diagnostics = validate(graph(), build_schemas([DeclaredValues, AddInts]), ["sum"])
        assert any(item.severity == "error" for item in diagnostics)

    run_scenario(scenario())


def test_entry_list_fan_out_interface_depends_only_on_stored_descriptors(
    runtime: tuple[Engine, InProcessWorker, TypeRegistry],
) -> None:
    engine, _, _ = runtime
    selected = {"id": "selected", "name": "Selected", "type": "int", "index": 1}

    def graph(entries: str, items: list[int]) -> Graph:
        return Graph(
            nodes={
                "fan": GraphNode(
                    "test.entry_fan_out",
                    {"entries": entries, "items": TypedLiteral("list<INT>", items)},
                ),
                "sum": GraphNode(
                    "test.add_ints",
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

    run_scenario(scenario())


@pytest.mark.parametrize("in_process", [True, False])
def test_persisted_descriptors_start_only_on_execution(tmp_path: Path, in_process: bool) -> None:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "catalogdescriptors"\n[pack.sandbox]\n'
        '[pack.entry]\nnodes = "catalogdescriptors:NODES"\n'
    )
    (tmp_path / "catalogdescriptors").mkdir()
    (tmp_path / "catalogdescriptors/__init__.py").write_text(
        "from dinkster_api.v1 import (Node, NodeSchema, InputSpec, OutputSpec, "
        "OutputDescriptorsSpec, TypeExpr, output_descriptor_entries)\n"
        "class Expressions(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema('catalogdescriptors.math',\n"
        "            inputs=(InputSpec('entries', TypeExpr.concrete('STRING')),),\n"
        "            output_descriptors=OutputDescriptorsSpec('entries',\n"
        "                (OutputSpec('int', TypeExpr.concrete('INT')),), 32))\n"
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
                response = await client.get("/api/nodes")
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

    run_scenario(scenario())


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
            "producer": GraphNode("test.add_ints", {"a": 1, "b": 2}),
            "lazy": GraphNode(
                "test.lazy_descriptors",
                {
                    "entries": document({"id": "answer", "name": "Answer", "type": "int"}),
                    "values.a": Link("producer", "sum"),
                },
            ),
        }
    )
    result = run_scenario(engine.run(graph, ["lazy"]))
    assert result.outputs["lazy"]["answer"].resolve() == 42
    assert "producer" not in result.executed
