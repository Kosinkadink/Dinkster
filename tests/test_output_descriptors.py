from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, validate
from dinkster_schema import (
    Described,
    Group,
    Input,
    Item,
    Members,
    Node,
    NodeSchema,
    Numbered,
    Output,
    TypeExpr,
    build_node_types,
    build_schemas,
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
            body=(
                Input("base", INT, default=5),
                Described(
                    "entries",
                    (Output("int", INT), Output("boolean", TypeExpr.concrete("BOOLEAN"))),
                    32,
                ),
            ),
        )

    @classmethod
    def execute(cls, *, base: int, entries: Mapping[str, object]) -> Mapping[str, object]:
        return {"entries": dict.fromkeys(entries, base)}


class AddInts(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "test.add_ints",
            body=(Input("a", INT), Input("b", INT), Output("sum", INT)),
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
    nodes = [DeclaredValues, AddInts]
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
        entry = {"id": "stable", "label": "Answer", "kind": "int"}

        def graph() -> Graph:
            return Graph(
                nodes={
                    "expression": GraphNode("test.declared_values", {"entries": document(entry)}),
                    "sum": GraphNode(
                        "test.add_ints", {"a": Link("expression", "entries.stable"), "b": 7}
                    ),
                }
            )

        first = await engine.run(graph(), ["sum"])
        assert first.outputs["sum"]["sum"].resolve() == 12
        second = await engine.run(graph(), ["sum"])
        assert "expression" in second.cached
        entry["label"] = "Renamed"
        renamed = await engine.run(graph(), ["sum"])
        assert "expression" in renamed.executed
        assert renamed.outputs["sum"]["sum"].resolve() == 12
        entry["id"] = "new_identity"
        diagnostics = validate(graph(), build_schemas([DeclaredValues, AddInts]), ["sum"])
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
        "from dinkster_api.v1 import Described, Node, NodeSchema, Output, TypeExpr\n"
        "class Expressions(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema('catalogdescriptors.math',\n"
        "            body=(Described('entries',\n"
        "                (Output('int', TypeExpr.concrete('INT')),), 32),))\n"
        "    @classmethod\n"
        "    def execute(cls, *, entries):\n"
        "        return {'entries': dict.fromkeys(entries, 42)}\n"
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
                assert '"described"' in await response.text()
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
                                        "label": "Answer",
                                        "kind": "int",
                                    }
                                )
                            },
                        )
                    }
                ),
                ["math"],
            )
            assert result.outputs["math"]["entries.answer"].resolve() == 42
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
                body=(
                    Group(
                        "values",
                        value=Input("value", INT, lazy=True),
                        keys=Numbered(""),
                        min_members=1,
                    ),
                    Described("entries", (Output("int", INT),), 1),
                ),
            )

        @classmethod
        def check_lazy_status(cls, **kwargs: object) -> list[str]:
            return []

        @classmethod
        def execute(
            cls, *, entries: Mapping[str, object], values: Members[Item]
        ) -> Mapping[str, object]:
            return {"entries": {"answer": 42}}

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
                    "entries": document({"id": "answer", "label": "Answer", "kind": "int"}),
                    "values.1": Link("producer", "sum"),
                },
            ),
        }
    )
    result = run_scenario(engine.run(graph, ["lazy"]))
    assert result.outputs["lazy"]["entries.answer"].resolve() == 42
    assert "producer" not in result.executed
