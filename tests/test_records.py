"""Declared records have directional named-field connections."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_api.v1 import Input, Node, NodeSchema, Output, RecordField, TypeExpr
from dinkster_caches import DiskCacheStore, MemoryLRUCache, encode_entry
from dinkster_engine import Engine, ExecutionError
from dinkster_graph import (
    PORTS_NODE_ID,
    Graph,
    GraphNode,
    Link,
    RegionNode,
    RegionOutput,
    graph_from_wire,
    graph_output_types,
    graph_to_wire,
    validate,
)
from dinkster_protocol import Invocation, InvocationResult, OnInvocationEvent
from dinkster_schema import (
    build_node_types,
    build_schemas,
    schema_from_wire,
    schema_to_wire,
    type_expr_to_wire,
)
from dinkster_values import (
    RESOURCE_HANDLE_TYPE,
    RecordPayload,
    ResourceHandle,
    TypeRegistry,
    Value,
    is_absent,
    list_children,
    make_list_value,
    make_record_value,
    make_tuple_value,
    register_core_types,
    register_resource_handle_type,
    value_children,
    value_resource_ids,
)
from dinkster_workers import InProcessWorker
from dinkster_workers.boundary import (
    BoundaryError,
    ValueCodec,
    decode_invocation,
    encode_invocation,
)

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
        forwarded: list[str] = []

        def engine(field_name: str, destination_id: str) -> Engine:
            record = TypeExpr.record_of(
                "source.settings",
                RecordField(
                    "count", field_name, INTEGER, default=5 if field_name == "seed" else 31
                ),
                RecordField("label", "prompt", TEXT),
                RecordField("optional", "guidance", INTEGER, optional=True, default=19),
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

            class Forward(Node):
                @classmethod
                def define_schema(cls) -> NodeSchema:
                    generic = TypeExpr.variable("T")
                    return NodeSchema(
                        "test.record.forward",
                        body=(Input("value", generic), Output("value", generic)),
                    )

                @classmethod
                def execute(cls, *, value: object) -> Mapping[str, object]:
                    forwarded.append(field_name)
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

            nodes = (Produce, Forward, Consume)
            return Engine(
                schemas=build_schemas(nodes),
                registry=registry,
                worker=InProcessWorker(build_node_types(nodes), registry),
                cache=cache,
            )

        graph = Graph(
            nodes={
                "make": GraphNode("test.record.make"),
                "forward": GraphNode("test.record.forward", {"value": Link("make", "value")}),
                "read": GraphNode("test.record.read", {"value": Link("forward", "value")}),
            }
        )
        first = await engine("seed", destination_id).run(graph, ["make", "forward", "read"])
        assert first.outputs["read"]["count"].resolve() == 29
        second = await engine("noise", destination_id).run(graph, ["make", "forward", "read"])
        assert second.outputs["read"]["count"].resolve() == 29
        assert second.cached == ("make", "forward")
        assert produced == ["seed"]
        assert forwarded == ["seed"]
        renamed = second.outputs["make"]["value"].resolve()
        expected = {"noise": 29, "prompt": "seven"}
        assert renamed == (
            [expected]
            if container == "list"
            else ("first", expected)
            if container == "tuple"
            else expected
        )
        assert second.outputs["forward"]["value"].resolve() == renamed

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


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_generic_record_containers_keep_fields_for_downstream_and_region_contracts(
    raw: bool,
    lazy: bool,
) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        record = TypeExpr.record_of(
            "test.settings",
            RecordField("s", "seed", INTEGER),
            RecordField("p", "prompt", TEXT),
        )
        generic = TypeExpr.variable("T")

        class Produce(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema("test.make", body=(Output("value", record),))

            @classmethod
            def execute(cls) -> Mapping[str, object]:
                return cls.outputs(value={"seed": 29, "prompt": "seven"})

        class Collect(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema(
                    "test.collect",
                    body=(Input("value", generic), Output("value", TypeExpr.list_of(generic))),
                    input_values=True,
                )

            @classmethod
            def execute(cls, *, input_values: Mapping[str, Value]) -> Mapping[str, object]:
                value = input_values["value"]
                return cls.outputs(
                    value=[value.resolve()] if raw else make_list_value(value.type_id, (value,))
                )

        class Pair(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema(
                    "test.pair",
                    body=(
                        Input("value", TypeExpr.list_of(generic)),
                        Output("value", TypeExpr.tuple_of(TEXT, generic)),
                    ),
                    input_values=True,
                )

            @classmethod
            def execute(cls, *, input_values: Mapping[str, Value]) -> Mapping[str, object]:
                children = list_children(input_values["value"])
                assert children is not None
                return cls.outputs(
                    value=("first", children[0].resolve())
                    if raw
                    else make_tuple_value((registry.wrap("STRING", "first"), children[0]))
                )

        class Pick(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema(
                    "test.pick",
                    body=(
                        Input("value", generic, lazy=lazy),
                        Input("index", INTEGER, default=-1),
                        Output("value", TypeExpr.tuple_element("value", "index")),
                    ),
                    input_values=True,
                )

            @classmethod
            def check_lazy_status(cls, **inputs: object) -> list[str]:
                return ["value"] if inputs.get("value") is None else []

            @classmethod
            def execute(
                cls, *, index: int, input_values: Mapping[str, Value]
            ) -> Mapping[str, object]:
                children = value_children(input_values["value"])
                assert children is not None
                return cls.outputs(value=children[index].resolve() if raw else children[index])

        class Consume(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema(
                    "test.read", body=(Input("value", record), Output("value", INTEGER))
                )

            @classmethod
            def execute(cls, *, value: dict[str, object]) -> Mapping[str, object]:
                return cls.outputs(value=value["seed"])

        nodes = (Produce, Collect, Pair, Pick, Consume)
        schemas = build_schemas(nodes)
        graph = Graph(
            nodes={
                "make": GraphNode("test.make"),
                "collect": GraphNode("test.collect", {"value": Link("make", "value")}),
                "pair": GraphNode("test.pair", {"value": Link("collect", "value")}),
                "pick": GraphNode("test.pick", {"value": Link("pair", "value"), "index": -1}),
                "read": GraphNode("test.read", {"value": Link("pick", "value")}),
                "region": RegionNode(
                    kind="map",
                    body=Graph(
                        nodes={
                            "pick": GraphNode("test.pick", {"value": Link(PORTS_NODE_ID, "pair")})
                        }
                    ),
                    ports={"pair": TypeExpr.tuple_of(TEXT, record), "item": INTEGER},
                    inputs={"pair": Link("pair", "value"), "item": [3, 7]},
                    element_ports=("item",),
                    outputs={"value": RegionOutput(Link("pick", "value"))},
                ),
            }
        )
        assert validate(graph, schemas, ["read", "region"]) == []
        output_types = graph_output_types(graph, schemas)
        assert output_types["pick"]["value"] == record
        assert output_types["region"]["value"] == TypeExpr.list_of(record)
        engine = Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        result = await engine.run(graph, ["read", "region"])
        assert result.outputs["read"]["value"].resolve() == 29
        assert result.outputs["region"]["value"].resolve() == [
            {"seed": 29, "prompt": "seven"},
            {"seed": 29, "prompt": "seven"},
        ]
        for wanted in (
            TypeExpr.record_of("other.settings", RecordField("s", "missing", INTEGER)),
            TypeExpr.record_of("other.settings", RecordField("s", "seed", TEXT)),
        ):
            broken = {
                **schemas,
                "test.read": replace(
                    schemas["test.read"], body=(Input("value", wanted), Output("value", INTEGER))
                ),
            }
            problems = validate(graph, broken, ["read"])
            assert any(
                d.code == "record-field-mismatch" and d.node_id == "read" and d.severity == "error"
                for d in problems
            )

    run_scenario(scenario())


@pytest.mark.parametrize("empty", [False, True])
def test_raw_generic_records_retain_optional_fields_across_invocation_wire(empty: bool) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        record = TypeExpr.record_of(
            "test.settings",
            RecordField("s", "seed", INTEGER),
            RecordField("g", "guidance", INTEGER, optional=True),
        )
        generic = TypeExpr.variable("T")

        class Produce(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema("test.make", body=(Output("value", TypeExpr.list_of(record)),))

            @classmethod
            def execute(cls) -> Mapping[str, object]:
                return cls.outputs(value=[] if empty else [{"seed": 29}])

        class Append(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema(
                    "test.append",
                    body=(
                        Input("value", TypeExpr.list_of(generic)),
                        Output("value", TypeExpr.tuple_of(TEXT, TypeExpr.list_of(generic))),
                    ),
                )

            @classmethod
            def execute(cls, *, value: list[dict[str, object]]) -> Mapping[str, object]:
                return cls.outputs(value=("first", [*value, {"seed": 97, "guidance": 11}]))

        nodes = (Produce, Append)
        worker = InProcessWorker(build_node_types(nodes), registry)

        class WireWorker:
            async def prepare(self, node_types: Sequence[str]) -> None:
                await worker.prepare(node_types)

            async def invoke(
                self, invocation: Invocation, on_event: OnInvocationEvent | None = None
            ) -> InvocationResult:
                sender = ValueCodec(registry, use_shm=False, accept_shm=False)
                header, blobs, segments, _ = encode_invocation(sender, invocation)
                assert segments == []
                receiver = ValueCodec(registry, use_shm=False, accept_shm=False)
                decoded = decode_invocation(receiver, header, blobs, [])
                if invocation.input_types:
                    bad = {
                        **header,
                        "inputTypes": {"value": {"kind": "concrete", "types": ["INT"]}},
                    }
                    with pytest.raises(BoundaryError, match="input_types disagrees"):
                        decode_invocation(receiver, bad, blobs, [])
                return await worker.invoke(decoded, on_event)

        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=WireWorker(),
            cache=MemoryLRUCache(),
        )
        result = await engine.run(
            Graph(
                nodes={
                    "make": GraphNode("test.make"),
                    "append": GraphNode("test.append", {"value": Link("make", "value")}),
                }
            ),
            ["append"],
        )
        assert result.outputs["append"]["value"].resolve() == (
            "first",
            [{"seed": 97, "guidance": 11}]
            if empty
            else [{"seed": 29}, {"seed": 97, "guidance": 11}],
        )

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


def test_change_rebinds_absent_foreign_fields_without_reusing_cached_assignment() -> None:
    from dinkster.record_nodes import record_helpers

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        cache = MemoryLRUCache()
        target = TypeExpr.record_of(
            "target.settings", RecordField("target", "guidance", INTEGER, optional=True)
        )

        def engine(swapped: bool) -> Engine:
            source = TypeExpr.record_of(
                "source.settings",
                RecordField("seed", "seed", INTEGER),
                RecordField("a", "strength" if swapped else "guidance", INTEGER, optional=True),
                RecordField("b", "guidance" if swapped else "strength", INTEGER, optional=True),
            )

            class Produce(Node):
                @classmethod
                def define_schema(cls) -> NodeSchema:
                    return NodeSchema("test.source", body=(Output("record", source),))

                @classmethod
                def execute(cls) -> Mapping[str, object]:
                    return cls.outputs(record={"seed": 53})

            class Consumer(Node):
                @classmethod
                def define_schema(cls) -> NodeSchema:
                    return NodeSchema("test.consumer", body=(Input("record", target),))

            helpers = record_helpers(build_schemas((Produce, Consumer)).values())
            nodes = (Produce, *helpers.values())
            return Engine(
                schemas={
                    identity: schema_from_wire(schema_to_wire(schema))
                    for identity, schema in build_schemas(nodes).items()
                },
                registry=registry,
                worker=InProcessWorker(build_node_types(nodes), registry),
                cache=cache,
            )

        graph = Graph(
            nodes={
                "make": GraphNode("test.source"),
                "change": GraphNode(
                    "target.settings.change",
                    {"record": Link("make", "record"), "field": '["target"]', "value": 11},
                ),
                "read": GraphNode(
                    "source.settings.read",
                    {"record": Link("change", "record"), "field": '["b"]'},
                ),
            }
        )
        before = await engine(False).run(graph, ["change", "read"])
        assert before.outputs["change"]["record"].resolve() == {"seed": 53, "guidance": 11}
        assert is_absent(before.outputs["read"]["value"])
        after = await engine(True).run(graph, ["change", "read"])
        assert after.outputs["change"]["record"].resolve() == {"seed": 53, "guidance": 11}
        assert after.outputs["read"]["value"].resolve() == 11
        assert "make" in after.cached
        assert "change" not in after.cached

    run_scenario(scenario())


def test_empty_record_helpers_run_without_a_field_and_preserve_removed_selections() -> None:
    from dinkster.record_nodes import record_helpers

    async def scenario() -> None:
        record = TypeExpr.record_of("test.empty")
        declaration = NodeSchema("test.declaration", body=(Output("record", record),))
        helpers = record_helpers((declaration,))
        registry = TypeRegistry()
        register_core_types(registry)
        schemas = {
            identity: schema_from_wire(schema_to_wire(schema))
            for identity, schema in build_schemas(helpers.values()).items()
        }
        engine = Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(helpers.values()), registry),
            cache=MemoryLRUCache(),
        )
        graph = Graph(
            nodes={
                "make": GraphNode("test.empty.make"),
                "read": GraphNode("test.empty.read", {"record": Link("make", "record")}),
                "change": GraphNode("test.empty.change", {"record": Link("make", "record")}),
            }
        )
        fresh = await engine.run(graph, list(graph.nodes))
        assert fresh.diagnostics == ()
        assert is_absent(fresh.outputs["read"]["value"])
        assert (
            fresh.outputs["change"]["record"].fingerprint
            == fresh.outputs["make"]["record"].fingerprint
        )
        snapshot = json.dumps(
            {"path": ["seed"], "name": "seed", "type": type_expr_to_wire(INTEGER)}
        )
        stale_graph = replace(
            graph,
            nodes={
                **graph.nodes,
                "read": GraphNode(
                    "test.empty.read", {"record": Link("make", "record"), "field": snapshot}
                ),
                "change": GraphNode(
                    "test.empty.change", {"record": Link("make", "record"), "field": snapshot}
                ),
            },
        )
        stale = await engine.run(stale_graph, list(stale_graph.nodes))
        assert {d.node_id for d in stale.diagnostics} == {"read", "change"}
        assert all(d.severity == "warning" and "removed" in d.message for d in stale.diagnostics)
        assert is_absent(stale.outputs["read"]["value"])
        assert stale.outputs["read"]["value"].meta.entries["absent.standsFor"] == "INT"
        assert (
            stale.outputs["change"]["record"].fingerprint
            == stale.outputs["make"]["record"].fingerprint
        )
        populated = TypeExpr.record_of(
            "test.empty",
            RecordField("z", "last", INTEGER, default=19),
            RecordField("A", "first", INTEGER, default=31),
        )
        populated_helpers = record_helpers(
            (replace(declaration, body=(Output("record", populated),)),)
        )
        populated_engine = Engine(
            schemas=build_schemas(populated_helpers.values()),
            registry=registry,
            worker=InProcessWorker(build_node_types(populated_helpers.values()), registry),
            cache=MemoryLRUCache(),
        )
        populated_graph = replace(
            graph,
            nodes={
                **graph.nodes,
                "read": GraphNode(
                    "test.empty.read", {"record": Link("make", "record"), "field": "[]"}
                ),
                "change": GraphNode(
                    "test.empty.change",
                    {"record": Link("make", "record"), "field": "[]", "value": 47},
                ),
            },
        )
        resumed = await populated_engine.run(populated_graph, list(populated_graph.nodes))
        assert resumed.diagnostics == ()
        assert resumed.outputs["read"]["value"].resolve() == 31
        assert resumed.outputs["change"]["record"].resolve() == {"last": 19, "first": 47}

    run_scenario(scenario())


def test_saved_record_selection_keeps_old_wires_without_retargeting() -> None:
    from dinkster.record_nodes import record_helpers

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        cache = MemoryLRUCache()

        def engine(state: str) -> Engine:
            selected = RecordField(
                "replacement" if state == "removed" else "seed",
                "noise" if state == "renamed" else "seed",
                TEXT if state == "changed" else INTEGER,
            )
            fields = (selected, RecordField("kept", "guidance", INTEGER))
            record = TypeExpr.record_of(
                "test.settings", *(reversed(fields) if state == "renamed" else fields)
            )

            class Produce(Node):
                @classmethod
                def define_schema(cls) -> NodeSchema:
                    return NodeSchema("test.source", body=(Output("record", record),))

                @classmethod
                def execute(cls) -> Mapping[str, object]:
                    return cls.outputs(
                        record={
                            selected.name: "new text" if state == "changed" else 29,
                            "guidance": 11,
                        }
                    )

            class Consume(Node):
                @classmethod
                def define_schema(cls) -> NodeSchema:
                    return NodeSchema(
                        "test.consume", body=(Input("value", INTEGER), Output("value", INTEGER))
                    )

                @classmethod
                def execute(cls, value: int) -> Mapping[str, object]:
                    return cls.outputs(value=value + 7)

            helpers = record_helpers(build_schemas((Produce,)).values())
            nodes = (Produce, Consume, *helpers.values())
            return Engine(
                schemas={
                    identity: schema_from_wire(schema_to_wire(schema))
                    for identity, schema in build_schemas(nodes).items()
                },
                registry=registry,
                worker=InProcessWorker(build_node_types(nodes), registry),
                cache=cache,
            )

        selection = json.dumps(
            {"path": ["seed"], "name": "seed", "type": type_expr_to_wire(INTEGER)}
        )
        graph = graph_from_wire(
            json.loads(
                json.dumps(
                    graph_to_wire(
                        Graph(
                            nodes={
                                "make": GraphNode("test.source"),
                                "read": GraphNode(
                                    "test.settings.read",
                                    {"record": Link("make", "record"), "field": selection},
                                ),
                                "consume": GraphNode(
                                    "test.consume", {"value": Link("read", "value")}
                                ),
                                "change": GraphNode(
                                    "test.settings.change",
                                    {
                                        "record": Link("make", "record"),
                                        "field": selection,
                                        "value": Link("read", "value"),
                                    },
                                ),
                            }
                        )
                    )
                )
            )
        )
        before = await engine("first").run(graph, list(graph.nodes))
        assert before.outputs["consume"]["value"].resolve() == 36
        assert before.outputs["change"]["record"].resolve() == {"seed": 29, "guidance": 11}
        assert before.diagnostics == ()
        renamed = await engine("renamed").run(graph, list(graph.nodes))
        assert renamed.outputs["change"]["record"].resolve() == {"noise": 29, "guidance": 11}
        assert renamed.outputs["consume"]["value"].resolve() == 36
        assert set(graph.nodes).issubset(renamed.cached)
        assert renamed.diagnostics == ()

        for state in ("removed", "changed"):
            current = engine(state)
            stale = await current.run(graph, list(graph.nodes))
            assert is_absent(stale.outputs["read"]["value"])
            assert stale.outputs["read"]["value"].meta.entries["absent.standsFor"] == "INT"
            assert is_absent(stale.outputs["consume"]["value"])
            assert stale.skipped == ("consume",)
            expected = {"seed": "new text" if state == "changed" else 29, "guidance": 11}
            assert stale.outputs["change"]["record"].resolve() == expected
            assert {d.node_id for d in stale.diagnostics} == {"read", "change"}
            assert all(d.severity == "warning" and d.input_id == "field" for d in stale.diagnostics)
            if state == "changed":
                assert all("INT" in d.message and "STRING" in d.message for d in stale.diagnostics)
            repeated = await current.run(graph, list(graph.nodes))
            assert is_absent(repeated.outputs["read"]["value"])
            assert repeated.diagnostics == stale.diagnostics
            assert {"read", "change"}.issubset(repeated.cached)
            no_value = replace(
                graph,
                nodes={
                    **graph.nodes,
                    "change": GraphNode(
                        "test.settings.change",
                        {"record": Link("make", "record"), "field": selection},
                    ),
                },
            )
            assert (await current.run(no_value, ["change"])).outputs["change"][
                "record"
            ].resolve() == expected
            if state == "changed":
                reselected = replace(
                    graph,
                    nodes={
                        "make": graph.nodes["make"],
                        "read": GraphNode(
                            "test.settings.read",
                            {
                                "record": Link("make", "record"),
                                "field": json.dumps(
                                    {
                                        "path": ["seed"],
                                        "name": "seed",
                                        "type": type_expr_to_wire(TEXT),
                                    }
                                ),
                            },
                        ),
                    },
                )
                fresh = await current.run(reselected, ["read"])
                assert fresh.outputs["read"]["value"].resolve() == "new text"
                assert fresh.diagnostics == ()

    run_scenario(scenario())
