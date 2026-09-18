"""Dynamic slots (hazard H10): type-driven variant specialization.

The schema declares per-type variants; the DOCUMENT stores which one is
active; elaboration turns the choice into ordinary concrete inputs. Covers
the model guardrails, elaboration semantics, wire round trips, graph wire,
worker kwarg grouping (SlotValue), cache identity, and validation of the
connected type against the chosen variant.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, GraphValidationError, Invocation
from dinkster_graph import (
    Graph,
    GraphNode,
    Link,
    graph_from_wire,
    graph_to_wire,
    snapshot_graph,
    validate,
)
from dinkster_graph.wire import GraphWireError
from dinkster_schema import (
    DynamicSlotSpec,
    ElaborationError,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputCountSpec,
    OutputFamilySpec,
    OutputSpec,
    SchemaWireVersionRequirement,
    SlotValue,
    SlotVariant,
    TypeExpr,
    build_node_types,
    build_schemas,
    elaborate,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker
from scaffold_nodes import SCAFFOLD_NODES, register_scaffold_types

STRING = TypeExpr.concrete("core.string")
FLOAT = TypeExpr.concrete("core.float")
MATCHED = TypeExpr.variable("T", ("core.string", "core.float"))

SOURCE_SLOT = DynamicSlotSpec(
    id="source",
    variants=(
        SlotVariant(
            key="text",
            type=STRING,
            inputs=(InputSpec("prefix", STRING, default=""),),
        ),
        SlotVariant(
            key="number",
            type=FLOAT,
            inputs=(InputSpec("scale", FLOAT, default=1.0),),
        ),
    ),
)

STYLIZE_SCHEMA = NodeSchema(
    node_type="test.stylize",
    slots=(SOURCE_SLOT,),
    outputs=(OutputSpec("text", STRING),),
)

OPEN_SLOT = DynamicSlotSpec(
    "source",
    slot_type=STRING,
    inputs=(InputSpec("scale", FLOAT),),
    force_input=True,
)
OPEN_SCHEMA = NodeSchema(
    node_type="test.open-slot",
    slots=(OPEN_SLOT,),
    outputs=(OutputSpec("out", STRING),),
)

MATCHED_SLOT = DynamicSlotSpec(
    id="source",
    variants=(
        SlotVariant("text", STRING, inputs=(InputSpec("copy", MATCHED),)),
        SlotVariant("number", FLOAT, inputs=(InputSpec("copy", MATCHED),)),
    ),
    type_template_id="T",
)
MATCHED_SCHEMA = NodeSchema(
    node_type="test.matched-slot",
    slots=(MATCHED_SLOT,),
    outputs=(
        OutputSpec("out", MATCHED),
        OutputSpec("batch", TypeExpr.list_of(MATCHED)),
        OutputSpec("asset", TypeExpr.asset_of(MATCHED)),
    ),
)


# --- model guardrails ---


def test_variant_key_grammar_is_closed() -> None:
    for bad in ("", "has space", "dot.ted", "uni\u00e9", "a/b"):
        with pytest.raises(ValueError, match="variant key"):
            SlotVariant(key=bad, type=STRING)


def test_variant_type_must_be_recursively_concrete() -> None:
    for expr in (
        TypeExpr.wildcard(),
        TypeExpr(kind="union", types=("core.string", "core.float")),
        TypeExpr.list_of(TypeExpr.wildcard()),
    ):
        with pytest.raises(ValueError, match="recursively concrete"):
            SlotVariant(key="v", type=expr)
    # list-of-concrete IS a type choice and is allowed.
    SlotVariant(key="v", type=TypeExpr.list_of(STRING))


def test_slot_requires_variants_and_unique_keys() -> None:
    with pytest.raises(ValueError, match="at least one variant"):
        DynamicSlotSpec(id="s", variants=())
    with pytest.raises(ValueError, match="duplicate variant keys"):
        DynamicSlotSpec(
            id="s",
            variants=(SlotVariant("v", STRING), SlotVariant("v", FLOAT)),
        )


def test_variant_dependent_ids_are_local_and_unique() -> None:
    with pytest.raises(ValueError, match="duplicate dependent input ids"):
        SlotVariant(
            key="v",
            type=STRING,
            inputs=(InputSpec("x", STRING), InputSpec("x", FLOAT)),
        )
    with pytest.raises(ValueError, match="may not contain"):
        SlotVariant(key="v", type=STRING, inputs=(InputSpec("a.b", STRING),))


def test_two_variants_may_reuse_a_local_name() -> None:
    # Mutually exclusive by construction: the same conceptual knob, typed
    # differently per variant.
    DynamicSlotSpec(
        id="s",
        variants=(
            SlotVariant("a", STRING, inputs=(InputSpec("strength", FLOAT),)),
            SlotVariant("b", FLOAT, inputs=(InputSpec("strength", STRING),)),
        ),
    )


def test_slot_id_shares_the_input_namespace() -> None:
    with pytest.raises(ValueError, match="collides"):
        NodeSchema(
            node_type="test.collide",
            inputs=(InputSpec("source", STRING),),
            slots=(SOURCE_SLOT,),
        )
    with pytest.raises(ValueError, match="shadows slot"):
        NodeSchema(
            node_type="test.shadow",
            inputs=(InputSpec("source.prefix", STRING),),
            slots=(SOURCE_SLOT,),
        )
    with pytest.raises(ValueError, match="duplicate slot ids"):
        NodeSchema(
            node_type="test.dup",
            slots=(SOURCE_SLOT, SOURCE_SLOT),
        )


def test_base_schemas_never_carry_slot_choices() -> None:
    with pytest.raises(ValueError, match="slot_choices belong to elaborated"):
        NodeSchema(
            node_type="test.premature",
            slots=(SOURCE_SLOT,),
            slot_choices=(("source", "text"),),
        )


def test_slot_type_binding_guardrails() -> None:
    with pytest.raises(ValueError, match="only legal on closed slots"):
        DynamicSlotSpec("open", slot_type=STRING, type_template_id="T")
    with pytest.raises(ValueError, match="requires a required slot"):
        DynamicSlotSpec(
            "optional",
            variants=(SlotVariant("text", STRING),),
            required=False,
            type_template_id="T",
        )
    with pytest.raises(ValueError, match="no output uses it"):
        NodeSchema(
            node_type="test.unused-slot-binding",
            slots=(
                DynamicSlotSpec(
                    "source",
                    variants=(SlotVariant("text", STRING),),
                    type_template_id="T",
                ),
            ),
        )
    with pytest.raises(ValueError, match="outside output variable"):
        NodeSchema(
            node_type="test.disallowed-slot-binding",
            slots=(MATCHED_SLOT,),
            outputs=(OutputSpec("out", TypeExpr.variable("T", ("core.string",))),),
        )


def test_one_top_level_slot_may_bind_each_output_type_variable() -> None:
    duplicate = DynamicSlotSpec(
        "other",
        variants=(SlotVariant("text", STRING), SlotVariant("number", FLOAT)),
        type_template_id="T",
    )
    with pytest.raises(ValueError, match="bound by multiple slots"):
        NodeSchema(
            node_type="test.duplicate-slot-binding",
            slots=(MATCHED_SLOT, duplicate),
            outputs=(OutputSpec("out", MATCHED),),
        )

    nested = DynamicSlotSpec(
        "source",
        variants=(SlotVariant("text", STRING),),
        type_template_id="T",
    )
    with pytest.raises(ValueError, match="only legal on top-level slots"):
        NodeSchema(
            node_type="test.nested-slot-binding",
            input_families=(InputFamilySpec("items", (nested,)),),
            outputs=(OutputSpec("out", TypeExpr.variable("T")),),
        )


# --- elaboration ---


def test_active_variant_materializes_slot_and_dependents() -> None:
    effective = elaborate(STYLIZE_SCHEMA, [], slot_variants={"source": "text"})
    assert [spec.id for spec in effective.inputs] == ["source", "source.prefix"]
    socket = effective.input("source")
    assert socket is not None and socket.type == STRING and socket.required
    dep = effective.input("source.prefix")
    assert dep is not None and dep.default == ""
    assert effective.slots == ()
    assert effective.slot_choices == (("source", "text"),)
    assert effective.is_static


@pytest.mark.parametrize(
    ("variant", "expected"),
    (("text", STRING), ("number", FLOAT)),
)
def test_selected_slot_variant_binds_inputs_and_outputs(variant: str, expected: TypeExpr) -> None:
    effective = elaborate(
        MATCHED_SCHEMA,
        ["source", "source.copy"],
        slot_variants={"source": variant},
    )
    assert effective.input("source").type == expected  # type: ignore[union-attr]
    assert effective.input("source.copy").type == expected  # type: ignore[union-attr]
    assert effective.outputs[0].type == expected
    assert effective.outputs[1].type == TypeExpr.list_of(expected)
    assert effective.outputs[2].type == TypeExpr.asset_of(expected)


def test_selected_slot_variant_rejects_an_incompatible_active_input() -> None:
    slot = DynamicSlotSpec(
        id="source",
        variants=(
            SlotVariant("text", STRING),
            SlotVariant(
                "number",
                FLOAT,
                inputs=(InputSpec("copy", TypeExpr.variable("T", ("core.string",))),),
            ),
        ),
        type_template_id="T",
    )
    schema = NodeSchema(
        node_type="test.incompatible-matched-slot-input",
        slots=(slot,),
        outputs=(OutputSpec("out", MATCHED),),
    )
    with pytest.raises(ElaborationError, match="does not allow selected type 'core.float'"):
        elaborate(schema, ["source", "source.copy"], slot_variants={"source": "number"})


def test_slot_binding_resolves_count_bound_output_members() -> None:
    schema = NodeSchema(
        node_type="test.matched-output-family",
        inputs=(InputSpec("count", TypeExpr.concrete("core.int")),),
        slots=(MATCHED_SLOT,),
        output_families=(OutputFamilySpec("items", MATCHED, count=OutputCountSpec("count")),),
    )
    effective = elaborate(
        schema,
        {"count": 2, "source": "x", "source.copy": "x"},
        {"items": ("0", "1")},
        {"source": "text"},
    )
    assert [(output.id, output.type) for output in effective.outputs] == [
        ("items.0", STRING),
        ("items.1", STRING),
    ]

    empty = elaborate(
        schema,
        {"count": 0, "source": "x", "source.copy": "x"},
        {"items": ()},
        {"source": "number"},
    )
    assert empty.outputs == ()
    assert empty.slot_choices == (("source", "number"),)


def test_inactive_variant_contributes_nothing() -> None:
    effective = elaborate(STYLIZE_SCHEMA, [], slot_variants={"source": "number"})
    assert effective.input("source.prefix") is None
    assert [spec.id for spec in effective.inputs] == ["source", "source.scale"]


def test_required_slot_without_choice_refuses_loudly() -> None:
    with pytest.raises(ElaborationError, match="no stored variant choice"):
        elaborate(STYLIZE_SCHEMA, [])


def test_optional_slot_without_choice_elaborates_to_nothing() -> None:
    schema = NodeSchema(
        node_type="test.optslot",
        slots=(
            DynamicSlotSpec(
                id="extra",
                variants=(SlotVariant("text", STRING),),
                required=False,
            ),
        ),
        outputs=(OutputSpec("out", STRING),),
    )
    effective = elaborate(schema, [])
    assert effective.inputs == ()
    assert effective.slot_choices == ()


def test_absent_open_slot_is_bare_optional_socket_and_validates() -> None:
    effective = elaborate(OPEN_SCHEMA, [])
    assert [(spec.id, spec.required) for spec in effective.inputs] == [("source", False)]
    assert effective.slot_choices == ()
    graph = Graph(nodes={"n": GraphNode("test.open-slot")})
    assert validate(graph, {"test.open-slot": OPEN_SCHEMA}, ["n"]) == []


def test_connected_open_slot_activates_dependents_with_declared_requiredness() -> None:
    effective = elaborate(OPEN_SCHEMA, ["source", "source.scale"])
    assert [(spec.id, spec.required) for spec in effective.inputs] == [
        ("source", False),
        ("source.scale", True),
    ]
    assert effective.slot_choices == ()


def test_inactive_open_slot_dependent_is_unknown_input() -> None:
    graph = Graph(nodes={"n": GraphNode("test.open-slot", {"source.scale": 2.0})})
    diagnostics = validate(graph, {"test.open-slot": OPEN_SCHEMA}, ["n"])
    assert any(
        diagnostic.code == "unknown-input" and diagnostic.input_id == "source.scale"
        for diagnostic in diagnostics
    )


def test_variant_form_elaboration_remains_choice_driven() -> None:
    effective = elaborate(
        STYLIZE_SCHEMA,
        ["source", "source.prefix"],
        slot_variants={"source": "text"},
    )
    assert [spec.id for spec in effective.inputs] == ["source", "source.prefix"]
    assert effective.slot_choices == (("source", "text"),)


def test_unknown_variant_and_unknown_slot_are_document_errors() -> None:
    with pytest.raises(ElaborationError, match="unknown variant choice"):
        elaborate(STYLIZE_SCHEMA, [], slot_variants={"source": "gone"})
    with pytest.raises(ElaborationError, match="undeclared"):
        elaborate(STYLIZE_SCHEMA, [], slot_variants={"source": "text", "bogus": "x"})
    with pytest.raises(ElaborationError, match="non-string"):
        elaborate(STYLIZE_SCHEMA, [], slot_variants={"source": 3})  # type: ignore[dict-item]


def test_static_schema_rejects_stored_slot_choices() -> None:
    static = NodeSchema(node_type="test.static", outputs=(OutputSpec("out", STRING),))
    with pytest.raises(ElaborationError, match="static"):
        elaborate(static, [], slot_variants={"source": "text"})


def test_variant_choice_is_computation_identity() -> None:
    # Two variants with IDENTICAL interface shape still produce distinct
    # effective signatures - switching variants is a new computation.
    schema = NodeSchema(
        node_type="test.same_shape",
        slots=(
            DynamicSlotSpec(
                id="s",
                variants=(SlotVariant("a", STRING), SlotVariant("b", STRING)),
            ),
        ),
        outputs=(OutputSpec("out", STRING),),
    )
    sig_a = schema_signature(elaborate(schema, [], slot_variants={"s": "a"}))
    sig_b = schema_signature(elaborate(schema, [], slot_variants={"s": "b"}))
    assert sig_a != sig_b


# --- schema wire ---


def test_slots_round_trip_through_wire() -> None:
    wire = schema_to_wire(STYLIZE_SCHEMA)
    entries = [e for e in wire["interface"] if e["role"] == "dynamicSlot"]  # type: ignore[index, union-attr]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["id"] == "source"
    assert [v["key"] for v in entry["variants"]] == ["text", "number"]
    assert schema_from_wire(wire) == STYLIZE_SCHEMA


def test_slot_type_binding_round_trips_only_on_wire_32() -> None:
    wire = schema_to_wire(MATCHED_SCHEMA)
    entry = next(
        item
        for item in wire["interface"]  # type: ignore[union-attr]
        if item["role"] == "dynamicSlot"
    )
    assert entry["typeTemplateId"] == "T"
    assert schema_from_wire(wire) == MATCHED_SCHEMA

    for predecessor in (29, 30, 31):
        with pytest.raises(SchemaWireVersionRequirement) as required:
            schema_to_wire(MATCHED_SCHEMA, wire_version=predecessor)
        assert required.value.required_version == 32

    for predecessor in (29, 30, 31):
        wire["schemaVersion"] = predecessor
        with pytest.raises(ValueError, match="requires schema wire 32"):
            schema_from_wire(wire)

    wire["schemaVersion"] = 32
    entry["typeTemplateId"] = ""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        schema_from_wire(wire)


def test_slot_type_binding_joins_base_and_effective_signatures() -> None:
    unbound = NodeSchema(
        node_type=MATCHED_SCHEMA.node_type,
        slots=(dataclasses.replace(MATCHED_SLOT, type_template_id=""),),
        outputs=MATCHED_SCHEMA.outputs,
    )
    assert schema_signature(MATCHED_SCHEMA) != schema_signature(unbound)
    text = elaborate(
        MATCHED_SCHEMA,
        ["source", "source.copy"],
        slot_variants={"source": "text"},
    )
    number = elaborate(
        MATCHED_SCHEMA,
        ["source", "source.copy"],
        slot_variants={"source": "number"},
    )
    assert schema_signature(text) != schema_signature(number)


def test_recursive_slot_forms_round_trip_and_materialize_shared_inputs() -> None:
    variant_slot = DynamicSlotSpec(
        "source",
        variants=(
            SlotVariant(
                "text",
                STRING,
                inputs=(InputSpec("variant_dep", TypeExpr.variable("T")),),
            ),
        ),
        inputs=(InputSpec("shared", TypeExpr.variable("T")),),
    )
    open_slot = DynamicSlotSpec(
        "open",
        slot_type=STRING,
        inputs=(InputSpec("dependent", TypeExpr.list_of(TypeExpr.variable("T"))),),
        force_input=True,
    )
    schema = NodeSchema(
        node_type="test.recursive-slots",
        slots=(variant_slot, open_slot),
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema
    effective = elaborate(
        schema,
        ["source", "source.shared", "source.variant_dep", "open", "open.dependent"],
        slot_variants={"source": "text"},
    )
    assert [spec.id for spec in effective.inputs] == [
        "source",
        "source.shared",
        "source.variant_dep",
        "open",
        "open.dependent",
    ]
    open_socket = effective.input("open")
    assert open_socket is not None and open_socket.force_input


def test_elaborated_slot_choices_round_trip_through_wire() -> None:
    effective = elaborate(STYLIZE_SCHEMA, [], slot_variants={"source": "text"})
    wire = schema_to_wire(effective)
    assert wire["slotChoices"] == [["source", "text"]]
    assert schema_from_wire(wire) == effective


def test_slot_docs_never_join_the_signature() -> None:
    documented = NodeSchema(
        node_type="test.stylize",
        slots=(
            DynamicSlotSpec(
                id="source",
                doc="the content to stylize",
                variants=(
                    SlotVariant(
                        key="text",
                        type=STRING,
                        doc="plain text",
                        inputs=(InputSpec("prefix", STRING, default="", doc="prepended"),),
                    ),
                    SlotVariant(
                        key="number",
                        type=FLOAT,
                        inputs=(InputSpec("scale", FLOAT, default=1.0),),
                    ),
                ),
            ),
        ),
        outputs=(OutputSpec("text", STRING),),
    )
    assert schema_signature(documented) == schema_signature(STYLIZE_SCHEMA)


# --- graph model + wire ---


def test_graph_wire_round_trips_slot_variants() -> None:
    graph = Graph(
        nodes={
            "n": GraphNode(
                "test.stylize",
                {"source": "hi", "source.prefix": ">> "},
                slot_variants={"source": "text"},
            ),
            "m": GraphNode("test.other", {}),
        }
    )
    wire = graph_to_wire(graph)
    assert wire["nodes"]["n"]["slotVariants"] == {"source": "text"}  # type: ignore[index]
    assert "slotVariants" not in wire["nodes"]["m"]  # type: ignore[operator]
    back = graph_from_wire(wire)
    node = back.nodes["n"]
    assert isinstance(node, GraphNode)
    assert dict(node.slot_variants) == {"source": "text"}


def test_graph_wire_rejects_non_string_slot_variants() -> None:
    wire = {
        "nodes": {
            "n": {
                "nodeType": "test.stylize",
                "inputs": {},
                "slotVariants": {"source": 3},
            }
        }
    }
    with pytest.raises(GraphWireError, match="must be a string"):
        graph_from_wire(wire)


def test_snapshot_preserves_slot_variants() -> None:
    live: dict[str, str] = {"source": "text"}
    graph = Graph(nodes={"n": GraphNode("test.stylize", {}, slot_variants=live)})
    snap = snapshot_graph(graph)
    live["source"] = "number"  # mutating the caller's dict must not leak in
    node = snap.nodes["n"]
    assert isinstance(node, GraphNode)
    assert dict(node.slot_variants) == {"source": "text"}


# --- engine + worker end-to-end ---


class Stylize(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return STYLIZE_SCHEMA

    @classmethod
    def execute(cls, *, source: SlotValue) -> Mapping[str, object]:
        # Explicit dispatch on the variant key - never runtime type sniffing.
        if source.variant == "text":
            prefix = source.options["prefix"]
            return cls.outputs(text=f"{prefix}{source.value}")
        assert source.variant == "number"
        assert isinstance(source.value, float)
        scale = source.options["scale"]
        assert isinstance(scale, float)
        return cls.outputs(text=str(source.value * scale))


class OpenSlotProbe(Node):
    seen: list[dict[str, object]] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return OPEN_SCHEMA

    @classmethod
    def execute(cls, **kwargs: object) -> Mapping[str, object]:
        cls.seen.append(kwargs)
        return cls.outputs(out="ok")


class MatchedSlotPass(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.matched-slot-pass",
            slots=(MATCHED_SLOT,),
            outputs=(OutputSpec("out", MATCHED),),
        )

    @classmethod
    def execute(cls, *, source: SlotValue) -> Mapping[str, object]:
        return cls.outputs(out=source.value)


def make_engine() -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    register_scaffold_types(registry)
    nodes: list[type[Node]] = list(SCAFFOLD_NODES) + [Stylize]
    return Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
    )


def stylize_graph(variant: str, inputs: dict[str, object]) -> Graph:
    return Graph(nodes={"n": GraphNode("test.stylize", inputs, slot_variants={"source": variant})})


def test_slot_executes_with_grouped_slot_value() -> None:
    async def scenario() -> None:
        engine = make_engine()

        text = await engine.run(
            stylize_graph("text", {"source": "hello", "source.prefix": ">> "}), ["n"]
        )
        assert text.outputs["n"]["text"].resolve() == ">> hello"

        number = await engine.run(
            stylize_graph("number", {"source": 2.0, "source.scale": 3.0}), ["n"]
        )
        assert number.outputs["n"]["text"].resolve() == "6.0"

    asyncio.run(scenario())


def test_worker_wraps_matched_slot_output_as_the_selected_type() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        schema = MatchedSlotPass.define_schema()
        worker = InProcessWorker({schema.node_type: MatchedSlotPass}, registry)
        for variant, type_id, value in (
            ("text", "core.string", "hello"),
            ("number", "core.float", 2.5),
        ):
            effective = elaborate(
                schema,
                ["source", "source.copy"],
                slot_variants={"source": variant},
            )
            result = await worker.invoke(
                Invocation(
                    invocation_id=f"matched-{variant}",
                    node_id="n",
                    node_type=schema.node_type,
                    inputs={
                        "source": registry.wrap(type_id, value),
                        "source.copy": registry.wrap(type_id, value),
                    },
                    effective_schema=effective,
                )
            )
            assert result.error is None
            assert result.outputs is not None
            assert result.outputs["out"].type_id == type_id
            assert result.outputs["out"].resolve() == value

    asyncio.run(scenario())


def test_slot_variant_switch_is_a_cache_miss_even_with_same_shape() -> None:
    class Pick(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.pick",
                slots=(
                    DynamicSlotSpec(
                        id="s",
                        variants=(SlotVariant("a", STRING), SlotVariant("b", STRING)),
                    ),
                ),
                outputs=(OutputSpec("out", STRING),),
            )

        @classmethod
        def execute(cls, *, s: SlotValue) -> Mapping[str, object]:
            return cls.outputs(out=f"{s.variant}:{s.value}")

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        engine = Engine(
            schemas=build_schemas([Pick]),
            registry=registry,
            worker=InProcessWorker(build_node_types([Pick]), registry),
            cache=MemoryLRUCache(),
        )

        def graph(variant: str) -> Graph:
            return Graph(
                nodes={"n": GraphNode("test.pick", {"s": "x"}, slot_variants={"s": variant})}
            )

        first = await engine.run(graph("a"), ["n"])
        assert first.executed == ("n",)
        assert first.outputs["n"]["out"].resolve() == "a:x"

        # Same values, same interface SHAPE, different variant: must re-run.
        switched = await engine.run(graph("b"), ["n"])
        assert switched.executed == ("n",)
        assert switched.outputs["n"]["out"].resolve() == "b:x"

        # And back: the original entry is still valid and hits.
        back = await engine.run(graph("a"), ["n"])
        assert back.cached == ("n",)

    asyncio.run(scenario())


def test_connected_type_mismatch_is_diagnosed_never_redispatched() -> None:
    # The stored choice is authoritative for the interface: a producer whose
    # type disagrees gets the SAME advisory type-mismatch diagnostic as any
    # concrete input (the backend never silently re-dispatches to another
    # variant), because after elaboration the slot IS an ordinary input.
    schemas = build_schemas(list(SCAFFOLD_NODES) + [Stylize])
    graph = Graph(
        nodes={
            "c": GraphNode("std.string.concat", {"a": "he", "b": "llo", "separator": ""}),
            "n": GraphNode(
                "test.stylize",
                {"source": Link("c", "text"), "source.scale": 2.0},
                slot_variants={"source": "number"},
            ),
        }
    )
    diags = validate(graph, schemas, ["n"])
    mismatches = [d for d in diags if d.code == "type-mismatch"]
    assert mismatches and mismatches[0].input_id == "source"


def test_missing_choice_is_a_validation_error_at_run() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(nodes={"n": GraphNode("test.stylize", {"source": "hi"})})
        with pytest.raises(GraphValidationError):
            await engine.run(graph, ["n"])

    asyncio.run(scenario())


def test_worker_groups_slot_inputs_via_invocation_schema() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker({"test.stylize": Stylize}, registry)
        effective = elaborate(
            STYLIZE_SCHEMA,
            ["source", "source.prefix"],
            slot_variants={"source": "text"},
        )
        result = await worker.invoke(
            Invocation(
                invocation_id="i1",
                node_id="n",
                node_type="test.stylize",
                inputs={
                    "source": registry.wrap("core.string", "world"),
                    "source.prefix": registry.wrap("core.string", "* "),
                },
                effective_schema=effective,
            )
        )
        assert result.error is None
        assert result.outputs is not None
        assert result.outputs["text"].resolve() == "* world"

    asyncio.run(scenario())


def test_worker_delivers_open_slot_plain_and_omits_it_when_absent() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker({"test.open-slot": OpenSlotProbe}, registry)
        OpenSlotProbe.seen.clear()

        connected = await worker.invoke(
            Invocation(
                invocation_id="open-connected",
                node_id="n",
                node_type="test.open-slot",
                inputs={
                    "source": registry.wrap("core.string", "value"),
                    "source.scale": registry.wrap("core.float", 2.0),
                },
                effective_schema=elaborate(OPEN_SCHEMA, ["source", "source.scale"]),
            )
        )
        assert connected.error is None
        assert OpenSlotProbe.seen[-1] == {
            "source": "value",
            "source.scale": 2.0,
        }

        absent = await worker.invoke(
            Invocation(
                invocation_id="open-absent",
                node_id="n",
                node_type="test.open-slot",
                inputs={},
                effective_schema=elaborate(OPEN_SCHEMA, []),
            )
        )
        assert absent.error is None
        assert OpenSlotProbe.seen[-1] == {}

    asyncio.run(scenario())
