"""Declared records have directional named-field connections."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_api.v1 import Input, Node, NodeSchema, Output, RecordField, TypeExpr
from dinkster_caches import DiskCacheStore, MemoryLRUCache, encode_entry
from dinkster_engine import Engine, ExecutionError
from dinkster_graph import PORTS_NODE_ID, Graph, GraphNode, Link, RegionNode, RegionOutput, validate
from dinkster_schema import build_node_types, build_schemas, schema_from_wire, schema_to_wire
from dinkster_values import (
    RESOURCE_HANDLE_TYPE,
    RecordPayload,
    ResourceHandle,
    TypeRegistry,
    make_record_value,
    register_core_types,
    register_resource_handle_type,
    value_resource_ids,
)
from dinkster_workers import InProcessWorker
from dinkster_workers.boundary import ValueCodec

from tests.async_scenarios import run_scenario

INTEGER = TypeExpr.concrete("INT")
TEXT = TypeExpr.concrete("STRING")


def _connection(source: TypeExpr, destination: TypeExpr):
    producer = schema_from_wire(
        schema_to_wire(NodeSchema(node_type="source.make", body=(Output("record", source),)))
    )
    consumer = schema_from_wire(
        schema_to_wire(NodeSchema(node_type="target.read", body=(Input("record", destination),)))
    )
    graph = Graph(
        nodes={
            "make": GraphNode(producer.node_type),
            "read": GraphNode(consumer.node_type, {"record": Link("make", "record")}),
        }
    )
    return validate(graph, {producer.node_type: producer, consumer.node_type: consumer}, ["read"])


def test_record_connection_matches_names_across_packs_and_keeps_extra_fields() -> None:
    source = TypeExpr.record_of(
        "source.settings",
        RecordField("source-text", "prompt", TEXT),
        RecordField("source-count", "seed", INTEGER),
    )
    destination = TypeExpr.record_of(
        "target.settings", RecordField("target-count", "seed", INTEGER)
    )
    assert _connection(source, destination) == []
    problems = _connection(destination, source)
    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert problems[0].node_id == "read"
    assert problems[0].input_id == "record"
    assert "prompt" in problems[0].message

    wrong_type = replace(destination, fields=(RecordField("target-count", "seed", TEXT),))
    problems = _connection(source, wrong_type)
    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert all(word in problems[0].message for word in ("seed", "INT", "STRING"))
    problems = _connection(source, INTEGER)
    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert "INT" in problems[0].message
    assert "record" in problems[0].message


def test_record_rename_and_reorder_follow_only_explicit_identity() -> None:
    original = TypeExpr.record_of(
        "source.settings",
        RecordField("count", "seed", INTEGER),
        RecordField("text", "prompt", TEXT),
    )
    renamed = replace(
        original,
        fields=(original.fields[1], replace(original.fields[0], name="noise")),
    )
    assert _connection(renamed, original) == []
    assert _connection(original, renamed) == []
    cross_pack = replace(original, types=("target.settings",))
    problems = _connection(renamed, cross_pack)
    assert len(problems) == 1
    assert "seed" in problems[0].message
    replaced_identity = replace(
        original, fields=(replace(original.fields[0], id="new-count"), original.fields[1])
    )
    problems = _connection(replaced_identity, original)
    assert len(problems) == 1
    assert "seed" in problems[0].message


@pytest.mark.parametrize("container", ["record", "list", "tuple"])
def test_nested_record_connection_names_the_missing_inner_field(container: str) -> None:
    def shape(expr: TypeExpr) -> TypeExpr:
        if container == "list":
            return TypeExpr.list_of(expr)
        if container == "tuple":
            return TypeExpr.tuple_of(TEXT, expr)
        return expr

    source = TypeExpr.record_of(
        "source.outer",
        RecordField(
            "inner",
            "settings",
            shape(TypeExpr.record_of("source.inner", RecordField("s", "seed", INTEGER))),
        ),
    )
    compatible = TypeExpr.record_of(
        "target.outer",
        RecordField(
            "i",
            "settings",
            shape(TypeExpr.record_of("target.inner", RecordField("t", "seed", INTEGER))),
        ),
    )
    assert _connection(source, compatible) == []
    destination = TypeExpr.record_of(
        "target.outer",
        RecordField(
            "i",
            "settings",
            shape(TypeExpr.record_of("target.inner", RecordField("p", "prompt", TEXT))),
        ),
    )
    problems = _connection(source, destination)
    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert "settings" in problems[0].message
    assert "prompt" in problems[0].message
    wrong_type = replace(
        compatible,
        fields=(
            replace(
                compatible.fields[0],
                type=shape(TypeExpr.record_of("target.inner", RecordField("t", "seed", TEXT))),
            ),
        ),
    )
    problems = _connection(source, wrong_type)
    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert all(word in problems[0].message for word in ("settings", "seed", "INT", "STRING"))


@pytest.mark.parametrize("source", [INTEGER, TypeExpr.wildcard()])
def test_record_input_never_guesses_fields_from_an_undeclared_type(source: TypeExpr) -> None:
    destination = TypeExpr.record_of("target.settings", RecordField("count", "seed", INTEGER))
    problems = _connection(source, destination)
    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert "Expected record" in problems[0].message


@pytest.mark.parametrize("container", ["record", "list", "tuple"])
@pytest.mark.parametrize("destination_id", ["source.settings", "target.settings"])
def test_record_runtime_reads_cross_pack_fields_and_reuses_renamed_cache(
    container: str, destination_id: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        cache = MemoryLRUCache()
        produced: list[str] = []

        def engine(field_name: str, destination_id: str) -> Engine:
            record = TypeExpr.record_of(
                "source.settings",
                RecordField("count", field_name, INTEGER),
                RecordField("label", "prompt", TEXT),
                RecordField("optional", "guidance", INTEGER, optional=True),
            )
            if field_name != "seed":
                record = replace(record, fields=tuple(reversed(record.fields)))
            target = TypeExpr.record_of(
                destination_id,
                RecordField(
                    "count" if destination_id == "source.settings" else "other", field_name, INTEGER
                ),
            )

            def shape(expr: TypeExpr) -> TypeExpr:
                if container == "list":
                    return TypeExpr.list_of(expr)
                if container == "tuple":
                    return TypeExpr.tuple_of(TEXT, expr)
                return expr

            class Produce(Node):
                @classmethod
                def define_schema(cls) -> NodeSchema:
                    return NodeSchema("test.record.make", body=(Output("value", shape(record)),))

                @classmethod
                def execute(cls) -> Mapping[str, object]:
                    produced.append(field_name)
                    raw = {field_name: 29, "prompt": "seven"}
                    value = (
                        [raw]
                        if container == "list"
                        else ("first", raw)
                        if container == "tuple"
                        else raw
                    )
                    return cls.outputs(value=value)

            class Consume(Node):
                @classmethod
                def define_schema(cls) -> NodeSchema:
                    return NodeSchema(
                        "test.record.read",
                        body=(Input("value", shape(target)), Output("count", INTEGER)),
                        idempotent=False,
                    )

                @classmethod
                def execute(cls, *, value: object) -> Mapping[str, object]:
                    if container == "list":
                        assert isinstance(value, list)
                        value = value[0]
                    elif container == "tuple":
                        assert isinstance(value, tuple)
                        assert value[0] == "first"
                        value = value[1]
                    assert isinstance(value, dict)
                    assert value == {field_name: 29}
                    return cls.outputs(count=value[field_name])

            nodes = (Produce, Consume)
            return Engine(
                schemas=build_schemas(nodes),
                registry=registry,
                worker=InProcessWorker(build_node_types(nodes), registry),
                cache=cache,
            )

        graph = Graph(
            nodes={
                "make": GraphNode("test.record.make"),
                "read": GraphNode("test.record.read", {"value": Link("make", "value")}),
            }
        )
        first = await engine("seed", destination_id).run(graph, ["make", "read"])
        assert first.outputs["read"]["count"].resolve() == 29
        second = await engine("noise", destination_id).run(graph, ["make", "read"])
        assert second.outputs["read"]["count"].resolve() == 29
        assert second.cached == ("make",)
        assert produced == ["seed"]
        renamed = second.outputs["make"]["value"].resolve()
        expected = {"noise": 29, "prompt": "seven"}
        assert renamed == (
            [expected]
            if container == "list"
            else ("first", expected)
            if container == "tuple"
            else expected
        )

        codec = ValueCodec(registry, use_shm=False, accept_shm=False)
        blobs: list[bytes] = []
        original = first.outputs["make"]["value"]
        wire, _ = codec.encode(original, blobs, [])
        decoded, _ = codec.decode(wire, blobs, [])
        assert decoded.resolve() == original.resolve()
        assert decoded.fingerprint == original.fingerprint
        disk = DiskCacheStore(tmp_path, registry)
        await disk.put("record", {"value": decoded})
        restored = await DiskCacheStore(tmp_path, registry).get("record")
        assert restored is not None
        assert restored["value"].resolve() == original.resolve()
        assert restored["value"].fingerprint == original.fingerprint

    run_scenario(scenario())


@pytest.mark.parametrize("port", ["element", "broadcast", "state"])
@pytest.mark.parametrize("same_record", [False, True])
def test_region_record_port_binds_fields_before_exporting(port: str, same_record: bool) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        source_name = "noise" if same_record else "seed"
        source = TypeExpr.record_of(
            "source.settings",
            RecordField("count", source_name, INTEGER),
            RecordField("label", "prompt", TEXT),
        )
        target = TypeExpr.record_of(
            "source.settings" if same_record else "target.settings",
            RecordField("count" if same_record else "other", "seed", INTEGER),
        )

        class Produce(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema(
                    "test.region.make",
                    body=(
                        Output("value", TypeExpr.list_of(source) if port == "element" else source),
                    ),
                )

            @classmethod
            def execute(cls) -> Mapping[str, object]:
                values = [{source_name: 29, "prompt": "first"}, {source_name: 97, "prompt": "last"}]
                return cls.outputs(value=values if port == "element" else values[0])

        class Advance(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema(
                    "test.region.advance",
                    body=(
                        Input("value", target),
                        Input("increment", INTEGER),
                        Output("value", source),
                    ),
                )

            @classmethod
            def execute(cls, *, value: dict[str, int], increment: int) -> Mapping[str, object]:
                return cls.outputs(value={source_name: value["seed"] + increment, "prompt": "next"})

        region = RegionNode(
            kind="fold" if port == "state" else "map",
            body=Graph(nodes={}),
            ports={"record": target, "item": INTEGER},
            inputs={"record": Link("make", "value"), "item": [3, 7]},
            element_ports=("record",) if port == "element" else ("item",),
            state_ports=("record",) if port == "state" else (),
            outputs={
                "record": RegionOutput(
                    Link(PORTS_NODE_ID, "record"), "state" if port == "state" else "gather"
                )
            },
        )
        if port == "element":
            region = replace(
                region, ports={"record": target}, inputs={"record": Link("make", "value")}
            )
        elif port == "state":
            region = replace(
                region,
                body=Graph(
                    nodes={
                        "advance": GraphNode(
                            "test.region.advance",
                            {
                                "value": Link(PORTS_NODE_ID, "record"),
                                "increment": Link(PORTS_NODE_ID, "item"),
                            },
                        )
                    }
                ),
                outputs={"record": RegionOutput(Link("advance", "value"), "state")},
            )
        nodes = (Produce, Advance)
        schemas = build_schemas(nodes)
        graph = Graph(nodes={"make": GraphNode("test.region.make"), "region": region})
        if port == "state":
            assert validate(graph, schemas, ["region"]) == []
            for fields in (
                (RecordField("wrong", "other", INTEGER),),
                (replace(source.fields[0], type=TEXT), source.fields[1]),
            ):
                broken = TypeExpr.record_of(source.types[0], *fields)
                advance = schemas["test.region.advance"]
                broken_schemas = {
                    **schemas,
                    "test.region.advance": replace(
                        advance, body=(*advance.body[:-1], Output("value", broken))
                    ),
                }
                problems = validate(graph, broken_schemas, ["region"])
                assert any(
                    d.severity == "error"
                    and d.node_id == "region"
                    and d.input_id == "record"
                    and "seed" in d.message
                    for d in problems
                )
        engine = Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        result = await engine.run(graph, ["region"])
        value = result.outputs["region"]["record"]
        assert value.resolve() == (
            {"seed": 39}
            if port == "state"
            else [{"seed": 29}, {"seed": 97 if port == "element" else 29}]
        )
        assert value.type_id == (target.types[0] if port == "state" else f"list<{target.types[0]}>")

        optional_source = replace(
            source, fields=(replace(source.fields[0], optional=True), source.fields[1])
        )
        missing_input = False

        class MissingProduce(Produce):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                schema = Produce.define_schema()
                output_type = (
                    TypeExpr.list_of(optional_source) if port == "element" else optional_source
                )
                return replace(schema, body=(Output("value", output_type),))

            @classmethod
            def execute(cls) -> Mapping[str, object]:
                value = (
                    {"prompt": "missing"}
                    if missing_input
                    else {source_name: 29, "prompt": "present"}
                )
                return cls.outputs(value=[value] if port == "element" else value)

        class MissingAdvance(Advance):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                schema = Advance.define_schema()
                return replace(schema, body=(*schema.body[:-1], Output("value", optional_source)))

            @classmethod
            def execute(cls, *, value: dict[str, int], increment: int) -> Mapping[str, object]:
                return cls.outputs(value={"prompt": "missing"})

        for missing_at in ("input", "state") if port == "state" else ("input",):
            missing_input = missing_at == "input"
            missing_nodes = (MissingProduce, MissingAdvance)
            missing_engine = Engine(
                schemas=build_schemas(missing_nodes),
                registry=registry,
                worker=InProcessWorker(build_node_types(missing_nodes), registry),
                cache=MemoryLRUCache(),
            )
            with pytest.raises(
                ExecutionError, match="Missing required record field: seed"
            ) as raised:
                await missing_engine.run(graph, ["region"])
            assert raised.value.error.node_id == "region"
            assert (
                "state port" if missing_at == "state" else "input"
            ) in raised.value.error.message

    run_scenario(scenario())


def test_cross_pack_name_rebinding_does_not_reuse_another_fields_result() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        cache = MemoryLRUCache()
        executions: list[str] = []

        def engine(swapped: bool) -> Engine:
            first, second = ("other", "seed") if swapped else ("seed", "other")
            source = TypeExpr.record_of(
                "source.settings",
                RecordField("a", first, INTEGER),
                RecordField("b", second, INTEGER),
            )
            target = TypeExpr.record_of("target.settings", RecordField("x", "seed", INTEGER))

            class Produce(Node):
                @classmethod
                def define_schema(cls) -> NodeSchema:
                    return NodeSchema("test.make", body=(Output("value", source),))

                @classmethod
                def execute(cls) -> Mapping[str, object]:
                    executions.append("make")
                    return cls.outputs(value={first: 29, second: 97})

            class Consume(Node):
                @classmethod
                def define_schema(cls) -> NodeSchema:
                    return NodeSchema(
                        "test.read", body=(Input("value", target), Output("count", INTEGER))
                    )

                @classmethod
                def execute(cls, *, value: dict[str, int]) -> Mapping[str, object]:
                    executions.append("read")
                    return cls.outputs(count=value["seed"])

            nodes = (Produce, Consume)
            return Engine(
                schemas=build_schemas(nodes),
                registry=registry,
                worker=InProcessWorker(build_node_types(nodes), registry),
                cache=cache,
            )

        graph = Graph(
            nodes={
                "make": GraphNode("test.make"),
                "read": GraphNode("test.read", {"value": Link("make", "value")}),
            }
        )
        original = await engine(False).run(graph, ["read"])
        assert original.outputs["read"]["count"].resolve() == 29
        renamed = await engine(True).run(graph, ["read"])
        assert renamed.outputs["read"]["count"].resolve() == 97
        assert renamed.cached == ("make",)
        assert executions == ["make", "read", "read"]

    run_scenario(scenario())


@pytest.mark.parametrize("bad", [{}, {"seed": "wrong"}])
@pytest.mark.parametrize("wrapped", [False, True])
def test_worker_refuses_missing_and_wrong_type_record_fields(
    bad: dict[str, object], wrapped: bool
) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        record = TypeExpr.record_of("test.settings", RecordField("count", "seed", INTEGER))

        class Produce(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema("test.record.bad", body=(Output("value", record),))

            @classmethod
            def execute(cls) -> Mapping[str, object]:
                result = (
                    make_record_value(
                        "test.settings",
                        {"count": ("seed", registry.wrap("STRING", "wrong"))} if bad else {},
                    )
                    if wrapped
                    else bad
                )
                return cls.outputs(value=result)

        engine = Engine(
            schemas=build_schemas((Produce,)),
            registry=registry,
            worker=InProcessWorker(build_node_types((Produce,)), registry),
            cache=MemoryLRUCache(),
        )
        with pytest.raises(ExecutionError, match="seed|INT"):
            await engine.run(Graph(nodes={"bad": GraphNode("test.record.bad")}), ["bad"])

    run_scenario(scenario())


def test_record_resource_children_remain_discoverable_and_cannot_enter_disk_cache() -> None:
    registry = TypeRegistry()
    register_core_types(registry)
    register_resource_handle_type(registry)
    handle = ResourceHandle(
        resource_id="record-resource",
        kind="gpu",
        residency={"gpu": "cuda:0"},
        cost={"vram:cuda:0": 100},
        obj=object(),
    )
    value = make_record_value(
        "test.settings",
        {
            "h": ("model", registry.wrap(RESOURCE_HANDLE_TYPE, handle)),
            "s": ("seed", registry.wrap("INT", 29)),
        },
    )
    codec = ValueCodec(registry, use_shm=False, accept_shm=False)
    blobs: list[bytes] = []
    wire, _ = codec.encode(value, blobs, [])
    decoded, _ = codec.decode(wire, blobs, [])
    assert isinstance(decoded.payload, RecordPayload)
    assert value_resource_ids(decoded) == ("record-resource",)
    assert encode_entry({"value": decoded}, registry) is None
