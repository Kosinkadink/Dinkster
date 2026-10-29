"""Dynamic interfaces (hazard H10): elaboration, cache determinism, worker
validation against the effective schema, and output-family representability."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError, GraphValidationError, Invocation
from dinkster_graph import Graph, GraphNode, Link, TypedLiteral, snapshot_graph, validate
from dinkster_schema import (
    MAX_FAMILY_MEMBERS,
    DynamicComboOption,
    DynamicComboSpec,
    DynamicSlotSpec,
    ElaborationError,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputCountSpec,
    OutputFamilySpec,
    OutputInterface,
    OutputSpec,
    SlotVariant,
    TypeExpr,
    bind_type_variables,
    build_node_types,
    build_schemas,
    elaborate,
    resolved_type_id,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)
from dinkster_values import TypeRegistry, Value, register_core_types
from dinkster_workers import InProcessWorker
from scaffold_nodes import SCAFFOLD_NODES, register_scaffold_types

STRING = TypeExpr.concrete("core.string")
FLOAT = TypeExpr.concrete("core.float")

JOIN_SCHEMA = NodeSchema(
    node_type="test.join",
    inputs=(InputSpec("separator", STRING, default=", "),),
    input_families=(InputFamilySpec("pieces", STRING, min_members=1),),
    outputs=(OutputSpec("text", STRING),),
)


# --- elaboration is pure, stable, and order-preserving ---


def test_elaboration_members_keep_stored_ids_in_document_order() -> None:
    effective = elaborate(JOIN_SCHEMA, ["pieces.b", "separator", "pieces.a"])
    member_ids = [s.id for s in effective.inputs if s.id != "separator"]
    assert member_ids == ["pieces.b", "pieces.a"]  # stored ids, document order
    pieces_b = effective.input("pieces.b")
    assert pieces_b is not None
    assert pieces_b.type == STRING
    assert effective.input_families == ()  # effective interface is concrete


def test_elaboration_is_deterministic_and_ignores_unknown_keys() -> None:
    keys = ["separator", "pieces.x", "not_a_real_input"]
    a = elaborate(JOIN_SCHEMA, keys)
    b = elaborate(JOIN_SCHEMA, list(keys))
    assert a == b
    assert schema_signature(a) == schema_signature(b)
    assert a.input("not_a_real_input") is None  # validation's job, not ours


def test_unbound_elaboration_depends_only_on_key_names() -> None:
    # Unbound families depend only on stored key names. Count-bound output
    # families additionally consume their declared stored integer value.
    assert elaborate(JOIN_SCHEMA, ["pieces.a"]) == elaborate(JOIN_SCHEMA, ["pieces.a"])


def test_static_schema_elaborates_to_itself() -> None:
    static = NodeSchema(
        node_type="test.static",
        inputs=(InputSpec("a", FLOAT),),
        outputs=(OutputSpec("out", FLOAT),),
    )
    assert elaborate(static, ["a"]) is static  # identity: zero cost, same signature


def test_member_sets_produce_distinct_signatures() -> None:
    two = schema_signature(elaborate(JOIN_SCHEMA, ["pieces.a", "pieces.b"]))
    three = schema_signature(elaborate(JOIN_SCHEMA, ["pieces.a", "pieces.b", "pieces.c"]))
    reordered = schema_signature(elaborate(JOIN_SCHEMA, ["pieces.b", "pieces.a"]))
    assert two != three
    assert two != reordered  # member order is semantics (e.g. join order)


def test_recursive_elaboration_uses_construct_paths_without_choice_keys() -> None:
    variable = TypeExpr.variable("T")
    schema = NodeSchema(
        node_type="test.recursive-elaboration",
        combos=(
            DynamicComboSpec(
                "combo",
                options=(
                    DynamicComboOption(
                        "outer",
                        inputs=(
                            DynamicComboSpec(
                                "subcombo",
                                options=(
                                    DynamicComboOption(
                                        "inner",
                                        inputs=(InputSpec("float_x", variable),),
                                    ),
                                ),
                            ),
                            InputFamilySpec(
                                "frames",
                                (InputSpec("frame", TypeExpr.list_of(variable)),),
                                member_prefix="image",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        outputs=(OutputSpec("out", variable),),
    )
    effective = elaborate(
        schema,
        ["combo.subcombo.float_x", "combo.frames.beta", "combo.frames.alpha"],
        slot_variants={"combo": "outer", "combo.subcombo": "inner"},
    )
    assert [spec.id for spec in effective.inputs] == [
        "combo.subcombo.float_x",
        "combo.frames.beta",
        "combo.frames.alpha",
    ]
    assert effective.slot_choices == (
        ("combo", "outer"),
        ("combo.subcombo", "inner"),
    )
    assert all("outer" not in spec.id and "inner" not in spec.id for spec in effective.inputs)

    bindings = bind_type_variables(
        effective.inputs,
        {
            "combo.subcombo.float_x": "core.string",
            "combo.frames.beta": "list<core.string>",
            "combo.frames.alpha": "list<core.string>",
        },
    )
    assert bindings == {"T": "core.string"}
    assert resolved_type_id(effective.outputs[0].type, bindings) == "core.string"


def test_single_entry_family_omits_leaf_and_keeps_stable_suffixes() -> None:
    family = InputFamilySpec(
        "images",
        (InputSpec("image", STRING),),
        member_prefix="image",
    )
    schema = NodeSchema(node_type="test.single-family", input_families=(family,))
    first = elaborate(schema, ["images.b", "images.a"])
    reordered = elaborate(schema, ["images.a", "images.b"])
    assert [spec.id for spec in first.inputs] == ["images.b", "images.a"]
    assert [spec.id for spec in reordered.inputs] == ["images.a", "images.b"]


def test_grouped_family_appends_leaf_and_rejects_projected_collisions() -> None:
    grouped = InputFamilySpec(
        "group",
        (InputSpec("left", STRING), InputSpec("right", STRING)),
    )
    effective = elaborate(
        NodeSchema(node_type="test.grouped", input_families=(grouped,)),
        ["group.member.left", "group.member.right"],
    )
    assert [spec.id for spec in effective.inputs] == [
        "group.member.left",
        "group.member.right",
    ]

    colliding_slot = DynamicSlotSpec(
        "slot",
        variants=(SlotVariant("active", STRING, inputs=(InputSpec("value", STRING),)),),
        inputs=(InputSpec("value", STRING),),
    )
    collision = NodeSchema(
        node_type="test.projected-collision",
        slots=(colliding_slot,),
    )
    with pytest.raises(ElaborationError, match="projected input namespace collision"):
        elaborate(
            collision,
            ["slot", "slot.value"],
            slot_variants={"slot": "active"},
        )


def test_combo_choices_join_effective_signature() -> None:
    combo = DynamicComboSpec(
        "mode",
        options=(
            DynamicComboOption("a", inputs=(InputSpec("value", STRING),)),
            DynamicComboOption("b", inputs=(InputSpec("value", STRING),)),
        ),
    )
    schema = NodeSchema(node_type="test.combo-signature", combos=(combo,))
    sig_a = schema_signature(elaborate(schema, ["mode.value"], slot_variants={"mode": "a"}))
    sig_b = schema_signature(elaborate(schema, ["mode.value"], slot_variants={"mode": "b"}))
    assert sig_a != sig_b


def test_space_containing_combo_choice_elaborates_exact_option() -> None:
    schema = NodeSchema(
        node_type="test.space-combo-elaboration",
        combos=(
            DynamicComboSpec(
                "resize_type",
                options=(
                    DynamicComboOption(
                        "scale dimensions",
                        inputs=(InputSpec("width", FLOAT),),
                    ),
                    DynamicComboOption("scale by multiplier"),
                ),
            ),
        ),
    )

    effective = elaborate(
        schema,
        ["resize_type.width"],
        slot_variants={"resize_type": "scale dimensions"},
    )
    assert [spec.id for spec in effective.inputs] == ["resize_type.width"]
    assert effective.slot_choices == (("resize_type", "scale dimensions"),)
    with pytest.raises(ElaborationError, match="unknown option"):
        elaborate(
            schema,
            [],
            slot_variants={"resize_type": "scale dimension"},
        )


def test_combo_default_does_not_synthesize_document_state_and_empty_is_inactive() -> None:
    defaulted = NodeSchema(
        node_type="test.combo-default",
        combos=(
            DynamicComboSpec(
                "mode",
                options=(DynamicComboOption("a"),),
                default="a",
                required=False,
            ),
        ),
    )
    assert elaborate(defaulted, []).slot_choices == ()

    empty = NodeSchema(
        node_type="test.combo-empty",
        combos=(DynamicComboSpec("mode", options=(), required=True),),
    )
    assert elaborate(empty, []).slot_choices == ()


def test_family_members_can_be_discovered_from_nested_choice_state() -> None:
    family = InputFamilySpec(
        "items",
        (
            DynamicComboSpec(
                "mode",
                options=(DynamicComboOption("on", inputs=()),),
            ),
        ),
        min_members=1,
    )
    schema = NodeSchema(node_type="test.choice-only-family", input_families=(family,))
    effective = elaborate(
        schema,
        [],
        slot_variants={"items.stable_id.mode": "on"},
    )
    assert effective.inputs == ()
    assert effective.slot_choices == (("items.stable_id.mode", "on"),)
    graph = Graph(
        nodes={
            "n": GraphNode(
                schema.node_type,
                {},
                slot_variants={"items.stable_id.mode": "on"},
            )
        }
    )
    assert validate(graph, {schema.node_type: schema}, ["n"]) == []


def test_recursive_family_cardinality_counts_members_not_grouped_leaves() -> None:
    grouped = InputFamilySpec(
        "items",
        (InputSpec("left", STRING), InputSpec("right", STRING)),
        min_members=1,
        max_members=1,
    )
    schema = NodeSchema(node_type="test.grouped-cardinality", input_families=(grouped,))
    effective = elaborate(schema, ["items.one.left", "items.one.right"])
    assert len(effective.inputs) == 2
    with pytest.raises(ElaborationError, match="allows <= 1"):
        elaborate(
            schema,
            ["items.one.left", "items.one.right", "items.two.left"],
        )

    nested = InputFamilySpec(
        "outer",
        (InputFamilySpec("inner", (InputSpec("value", STRING),), min_members=1),),
    )
    with pytest.raises(ElaborationError, match="outer.one.inner.*needs >= 1"):
        elaborate(
            NodeSchema(node_type="test.nested-cardinality", input_families=(nested,)),
            ["outer.one.unrelated"],
        )


# --- schema-level guardrails ---


def test_static_input_may_not_shadow_a_family() -> None:
    with pytest.raises(ValueError, match="shadows family"):
        NodeSchema(
            node_type="test.shadow",
            inputs=(InputSpec("pieces.a", STRING),),
            input_families=(InputFamilySpec("pieces", STRING),),
        )


def test_family_id_may_not_collide_with_input_id() -> None:
    with pytest.raises(ValueError, match="collides"):
        NodeSchema(
            node_type="test.collide",
            inputs=(InputSpec("pieces", STRING),),
            input_families=(InputFamilySpec("pieces", STRING),),
        )


# --- wire format: input and output families round-trip ---


def test_families_round_trip_through_wire() -> None:
    schema = NodeSchema(
        node_type="test.wire",
        inputs=(InputSpec("seed", FLOAT, default=0.0),),
        input_families=(InputFamilySpec("items", STRING, min_members=1, max_members=8),),
        outputs=(OutputSpec("summary", STRING),),
        output_families=(
            OutputFamilySpec("results", STRING, min_members=1, max_members=4, doc="one per item"),
        ),
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema


# --- graph validation on the effective interface ---


def _schemas() -> dict[str, NodeSchema]:
    schemas = build_schemas(SCAFFOLD_NODES)
    schemas[JOIN_SCHEMA.node_type] = JOIN_SCHEMA
    return schemas


def test_family_member_count_is_validated() -> None:
    graph = Graph(nodes={"j": GraphNode("test.join", {"separator": "-"})})
    diags = validate(graph, _schemas(), ["j"])
    assert any(d.code == "family-too-few" for d in diags)

    bounded = NodeSchema(
        node_type="test.bounded",
        input_families=(InputFamilySpec("items", STRING, max_members=1),),
        outputs=(OutputSpec("out", STRING),),
    )
    graph = Graph(nodes={"n": GraphNode("test.bounded", {"items.a": "x", "items.b": "y"})})
    diags = validate(graph, {"test.bounded": bounded}, ["n"])
    assert any(d.code == "family-too-many" for d in diags)


FANOUT_SCHEMA = NodeSchema(
    node_type="test.fanout",
    inputs=(InputSpec("seed", STRING),),
    output_families=(OutputFamilySpec("results", STRING),),
)


def test_links_resolve_against_elaborated_producer_outputs() -> None:
    schemas = _schemas()
    schemas["test.fanout"] = FANOUT_SCHEMA
    graph = Graph(
        nodes={
            "p": GraphNode("test.fanout", {"seed": "s"}, output_members={"results": ("0", "1")}),
            "j": GraphNode("test.join", {"pieces.a": Link("p", "results.0"), "separator": "-"}),
        }
    )
    assert validate(graph, schemas, ["j"]) == []


def test_dangling_dynamic_member_is_a_document_time_error() -> None:
    schemas = _schemas()
    schemas["test.fanout"] = FANOUT_SCHEMA
    graph = Graph(
        nodes={
            # Document stores only member "0", but the consumer links to "2".
            "p": GraphNode("test.fanout", {"seed": "s"}, output_members={"results": ("0",)}),
            "j": GraphNode("test.join", {"pieces.a": Link("p", "results.2"), "separator": "-"}),
        }
    )
    diags = validate(graph, schemas, ["j"])
    dangling = [d for d in diags if d.code == "dangling-output"]
    assert dangling and "currently has members" in dangling[0].message


# --- engine + worker end-to-end ---


def make_engine(extra_nodes: list[type[Node]] | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    register_scaffold_types(registry)
    nodes = list(SCAFFOLD_NODES) + (extra_nodes or [])
    return Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
    )


def join_graph(*pieces: tuple[str, str], separator: str = " ") -> Graph:
    inputs: dict[str, object] = {f"pieces.{k}": v for k, v in pieces}
    inputs["separator"] = separator
    return Graph(nodes={"j": GraphNode("std.string.join", inputs)})


def test_dynamic_inputs_execute_and_cache_deterministically() -> None:
    async def scenario() -> None:
        engine = make_engine()

        first = await engine.run(join_graph(("a", "x"), ("b", "y")), ["j"])
        assert first.outputs["j"]["text"].resolve() == "x y"
        assert first.executed == ("j",)

        # Same member set and values -> cache hit.
        rerun = await engine.run(join_graph(("a", "x"), ("b", "y")), ["j"])
        assert rerun.cached == ("j",)

        # Growing the family -> different effective interface -> re-executes.
        grown = await engine.run(join_graph(("a", "x"), ("b", "y"), ("c", "z")), ["j"])
        assert grown.executed == ("j",)
        assert grown.outputs["j"]["text"].resolve() == "x y z"

        # Shrinking back -> the original entry is still valid and hits.
        back = await engine.run(join_graph(("a", "x"), ("b", "y")), ["j"])
        assert back.cached == ("j",)

    asyncio.run(scenario())


def test_family_members_accept_links() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                "c": GraphNode("std.string.concat", {"a": "he", "b": "llo", "separator": ""}),
                "j": GraphNode(
                    "std.string.join",
                    {"pieces.a": Link("c", "text"), "pieces.b": "world", "separator": " "},
                ),
            }
        )
        result = await engine.run(graph, ["j"])
        assert result.outputs["j"]["text"].resolve() == "hello world"

    asyncio.run(scenario())


def test_min_members_enforced_at_run() -> None:
    async def scenario() -> None:
        engine = make_engine()
        with pytest.raises(GraphValidationError):
            await engine.run(join_graph(), ["j"])

    asyncio.run(scenario())


class WrongOutputs(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.wrong_outputs",
            inputs=(InputSpec("a", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    def execute(cls, *, a: str) -> Mapping[str, object]:
        return {"not_out": a}  # bypasses cls.outputs() on purpose


def test_worker_validates_against_the_effective_schema() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = InProcessWorker({"test.wrong_outputs": WrongOutputs}, registry)
        wrapped = registry.wrap("core.string", "v")
        effective = elaborate(WrongOutputs.define_schema(), ["a"])
        result = await worker.invoke(
            Invocation(
                invocation_id="i1",
                node_id="n",
                node_type="test.wrong_outputs",
                inputs={"a": wrapped},
                effective_schema=effective,
            )
        )
        assert result.error is not None
        assert "not_out" in result.error.message

        # And through the engine: the same violation surfaces as ExecutionError.
        engine = make_engine([WrongOutputs])
        graph = Graph(nodes={"n": GraphNode("test.wrong_outputs", {"a": "v"})})
        with pytest.raises(ExecutionError):
            await engine.run(graph, ["n"])

    asyncio.run(scenario())


# --- output families: elaboration is document-determined ---


def test_output_family_elaborates_stable_ids_in_document_order() -> None:
    effective = elaborate(FANOUT_SCHEMA, ["seed"], {"results": ("b", "a")})
    assert [out.id for out in effective.outputs] == ["results.b", "results.a"]
    results_b = effective.output("results.b")
    assert results_b is not None
    assert results_b.type == STRING
    assert effective.output_families == ()  # effective interface is concrete


def test_output_family_preview_propagates_to_elaborated_members() -> None:
    schema = NodeSchema(
        node_type="test.preview-fanout",
        output_families=(OutputFamilySpec("results", STRING, preview=True),),
    )
    effective = elaborate(schema, [], {"results": ("first", "second")})
    assert [(out.id, out.preview) for out in effective.outputs] == [
        ("results.first", True),
        ("results.second", True),
    ]


def test_output_member_sets_and_order_produce_distinct_signatures() -> None:
    two = schema_signature(elaborate(FANOUT_SCHEMA, ["seed"], {"results": ("a", "b")}))
    three = schema_signature(elaborate(FANOUT_SCHEMA, ["seed"], {"results": ("a", "b", "c")}))
    reordered = schema_signature(elaborate(FANOUT_SCHEMA, ["seed"], {"results": ("b", "a")}))
    assert two != three
    assert two != reordered  # member order is semantics


def test_counted_output_family_requires_exact_literal_and_membership() -> None:
    schema = NodeSchema(
        node_type="test.counted",
        inputs=(InputSpec("count", TypeExpr.concrete("core.int")),),
        output_families=(OutputFamilySpec("items", STRING, count=OutputCountSpec("count")),),
    )
    effective = elaborate(
        schema,
        {"count": 3},
        {"items": ("0", "1", "2")},
    )
    assert [output.id for output in effective.outputs] == ["items.0", "items.1", "items.2"]
    assert schema_signature(effective) != schema_signature(
        elaborate(schema, {"count": 1}, {"items": ("0",)})
    )

    for hostile in (None, True, 2.0, Link("n", "out"), TypedLiteral("core.int", 2)):
        with pytest.raises(ElaborationError, match="untyped integer literal"):
            elaborate(
                schema,
                {"count": hostile},
                {"items": ("0", "1")},
            )
    for suffixes in (("1", "0"), ("0", "2"), ("0",), ("0", "1", "2")):
        with pytest.raises(ElaborationError, match="canonical count suffixes"):
            elaborate(schema, {"count": 2}, {"items": suffixes})

    for count, reason in (
        (-1, "must be >= 0"),
        (MAX_FAMILY_MEMBERS + 1, "member budget"),
    ):
        with pytest.raises(ElaborationError, match=reason):
            elaborate(schema, {"count": count}, {"items": ()})
        graph = Graph(
            nodes={
                "n": GraphNode(
                    schema.node_type,
                    {"count": count},
                    output_members={"items": ()},
                )
            }
        )
        diagnostics = validate(graph, {schema.node_type: schema}, ["n"])
        assert [(diagnostic.code, reason in diagnostic.message) for diagnostic in diagnostics] == [
            ("elaboration-failed", True)
        ]


def test_output_membership_state_is_validated_deterministically() -> None:
    with pytest.raises(ElaborationError, match="undeclared"):
        elaborate(FANOUT_SCHEMA, ["seed"], {"nope": ("a",)})
    with pytest.raises(ElaborationError, match="duplicate member suffix"):
        elaborate(FANOUT_SCHEMA, ["seed"], {"results": ("a", "a")})
    with pytest.raises(ElaborationError, match="empty or non-string"):
        elaborate(FANOUT_SCHEMA, ["seed"], {"results": ("",)})
    static = NodeSchema(
        node_type="test.static",
        inputs=(InputSpec("a", STRING),),
        outputs=(OutputSpec("out", STRING),),
    )
    with pytest.raises(ElaborationError, match="static"):
        elaborate(static, ["a"], {"results": ("a",)})


def test_output_family_member_may_not_collide_with_static_output() -> None:
    schema = NodeSchema(
        node_type="test.collide_out",
        outputs=(OutputSpec("results.a", STRING),),
    )
    # Declaring the family alongside the shadowing static output is already
    # rejected at schema construction time.
    with pytest.raises(ValueError, match="shadows family"):
        NodeSchema(
            node_type="test.collide_out2",
            outputs=(OutputSpec("results.a", STRING),),
            output_families=(OutputFamilySpec("results", STRING),),
        )
    assert schema.is_static  # the plain static version stays legal


def test_reserved_output_spec_id_cannot_be_declared() -> None:
    with pytest.raises(ValueError, match="reserved"):
        NodeSchema(
            node_type="test.reserved",
            inputs=(InputSpec("output_spec", STRING),),
        )
    with pytest.raises(ValueError, match="reserved"):
        NodeSchema(
            node_type="test.reserved2",
            input_families=(InputFamilySpec("output_spec", STRING),),
        )


def test_output_family_bounds_are_validated_on_the_document() -> None:
    bounded = NodeSchema(
        node_type="test.bounded_out",
        output_families=(OutputFamilySpec("results", STRING, min_members=1, max_members=2),),
    )
    too_few = Graph(nodes={"n": GraphNode("test.bounded_out")})
    diags = validate(too_few, {"test.bounded_out": bounded}, ["n"])
    assert any(d.code == "family-too-few" for d in diags)

    too_many = Graph(
        nodes={"n": GraphNode("test.bounded_out", output_members={"results": ("a", "b", "c")})}
    )
    diags = validate(too_many, {"test.bounded_out": bounded}, ["n"])
    assert any(d.code == "family-too-many" for d in diags)


# --- output families: end to end through engine and worker ---


def split_graph(members: tuple[str, ...], text: str = "a b c") -> Graph:
    return Graph(
        nodes={
            "s": GraphNode(
                "std.string.split",
                {"text": text, "separator": " "},
                output_members={"parts": members},
            )
        }
    )


def test_dynamic_outputs_execute_and_cache_deterministically() -> None:
    async def scenario() -> None:
        engine = make_engine()

        first = await engine.run(split_graph(("a", "b")), ["s"])
        assert first.executed == ("s",)
        assert first.outputs["s"]["parts.a"].resolve() == "a"
        assert first.outputs["s"]["parts.b"].resolve() == "b c"

        # Same membership and inputs -> cache hit.
        rerun = await engine.run(split_graph(("a", "b")), ["s"])
        assert rerun.cached == ("s",)

        # Different membership -> different effective interface -> re-executes.
        grown = await engine.run(split_graph(("a", "b", "c")), ["s"])
        assert grown.executed == ("s",)
        assert grown.outputs["s"]["parts.c"].resolve() == "c"

    asyncio.run(scenario())


def test_dynamic_outputs_link_into_downstream_nodes() -> None:
    async def scenario() -> None:
        engine = make_engine()
        graph = Graph(
            nodes={
                "s": GraphNode(
                    "std.string.split",
                    {"text": "x y z", "separator": " "},
                    output_members={"parts": ("p", "q", "r")},
                ),
                "j": GraphNode(
                    "std.string.join",
                    {
                        "pieces.a": Link("s", "parts.r"),
                        "pieces.b": Link("s", "parts.p"),
                        "separator": "-",
                    },
                ),
            }
        )
        result = await engine.run(graph, ["j"])
        assert result.outputs["j"]["text"].resolve() == "z-x"

    asyncio.run(scenario())


def test_min_output_members_enforced_at_run() -> None:
    async def scenario() -> None:
        engine = make_engine()
        with pytest.raises(GraphValidationError):
            await engine.run(split_graph(()), ["s"])  # min_members=1

    asyncio.run(scenario())


class SpecProbe(Node):
    """Records the OutputInterface it receives, for boundary assertions."""

    seen: OutputInterface | None = None

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.spec_probe",
            inputs=(InputSpec("seed", STRING),),
            output_families=(OutputFamilySpec("results", STRING),),
        )

    @classmethod
    def execute(cls, *, seed: str, output_spec: OutputInterface) -> Mapping[str, object]:
        SpecProbe.seen = output_spec
        return cls.outputs(results={s: seed for s in output_spec.family("results")})


def test_output_spec_exposes_exact_ordered_membership() -> None:
    async def scenario() -> None:
        SpecProbe.seen = None
        engine = make_engine([SpecProbe])
        graph = Graph(
            nodes={
                "n": GraphNode(
                    "test.spec_probe",
                    {"seed": "v"},
                    output_members={"results": ("z", "a", "m")},
                )
            }
        )
        result = await engine.run(graph, ["n"])
        assert SpecProbe.seen is not None
        assert SpecProbe.seen.family("results") == ("z", "a", "m")  # document order
        assert SpecProbe.seen.family("unknown") == ()
        assert set(result.outputs["n"]) == {"results.z", "results.a", "results.m"}

    asyncio.run(scenario())


class CountSpecProbe(Node):
    seen: OutputInterface | None = None

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.count_spec_probe",
            inputs=(
                InputSpec("count", TypeExpr.concrete("core.int")),
                InputSpec("seed", STRING),
            ),
            output_families=(OutputFamilySpec("results", STRING, count=OutputCountSpec("count")),),
        )

    @classmethod
    def execute(
        cls, *, count: int, seed: str, output_spec: OutputInterface
    ) -> Mapping[str, object]:
        CountSpecProbe.seen = output_spec
        assert len(output_spec.family("results")) == count
        return cls.outputs(results={suffix: seed for suffix in output_spec.family("results")})


def test_counted_output_members_reach_worker_output_spec() -> None:
    async def scenario() -> None:
        CountSpecProbe.seen = None
        engine = make_engine([CountSpecProbe])
        graph = Graph(
            nodes={
                "n": GraphNode(
                    "test.count_spec_probe",
                    {"count": 3, "seed": "v"},
                    output_members={"results": ("0", "1", "2")},
                )
            }
        )
        result = await engine.run(graph, ["n"])
        assert CountSpecProbe.seen is not None
        assert CountSpecProbe.seen.family("results") == ("0", "1", "2")
        assert set(result.outputs["n"]) == {"results.0", "results.1", "results.2"}

    asyncio.run(scenario())


class BadFanout(Node):
    """Returns members that do not match the document's membership."""

    return_suffixes: tuple[str, ...] = ("wrong",)

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.bad_fanout",
            inputs=(InputSpec("seed", STRING),),
            output_families=(OutputFamilySpec("results", STRING),),
        )

    @classmethod
    def execute(cls, *, seed: str, output_spec: OutputInterface) -> Mapping[str, object]:
        return {"results": {s: seed for s in cls.return_suffixes}}


def test_wrong_grouped_return_members_fail_at_the_worker_boundary() -> None:
    async def scenario() -> None:
        engine = make_engine([BadFanout])
        graph = Graph(
            nodes={
                "n": GraphNode("test.bad_fanout", {"seed": "v"}, output_members={"results": ("a",)})
            }
        )
        # Extra member "wrong", missing member "a".
        BadFanout.return_suffixes = ("wrong",)
        with pytest.raises(ExecutionError, match="results.a"):
            await engine.run(graph, ["n"])
        # Missing member only.
        BadFanout.return_suffixes = ()
        with pytest.raises(ExecutionError, match="missing outputs: results.a"):
            await engine.run(graph, ["n"])

    asyncio.run(scenario())


def test_worker_rejects_invocation_provenance_skew() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_scaffold_types(registry)
        worker = InProcessWorker(build_node_types(SCAFFOLD_NODES), registry)
        schema = build_schemas(SCAFFOLD_NODES)["std.string.split"]
        effective = elaborate(schema, ["text", "separator"], {"parts": ("a", "b")})
        inputs = {
            "text": registry.wrap("core.string", "x y"),
            "separator": registry.wrap("core.string", " "),
        }

        # Provenance says one member; the effective schema says two.
        result = await worker.invoke(
            Invocation(
                invocation_id="i1",
                node_id="n",
                node_type="std.string.split",
                inputs=inputs,
                effective_schema=effective,
                output_members=(("parts", ("a",)),),
            )
        )
        assert result.error is not None
        assert "schema skew" in result.error.message

        # Provenance omits the family entirely.
        result = await worker.invoke(
            Invocation(
                invocation_id="i2",
                node_id="n",
                node_type="std.string.split",
                inputs=inputs,
                effective_schema=effective,
                output_members=(),
            )
        )
        assert result.error is not None
        assert "schema skew" in result.error.message

    asyncio.run(scenario())


class WrongShapeCache:
    """Returns a hit whose output ids do not match the effective schema."""

    def __init__(self, wrong: Mapping[str, Value]) -> None:
        self._wrong = wrong
        self.puts = 0

    async def get(self, key: str) -> Mapping[str, Value] | None:
        return self._wrong

    async def put(self, key: str, outputs: Mapping[str, Value]) -> None:
        self.puts += 1


def test_cache_hit_with_wrong_dynamic_shape_is_a_miss() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_scaffold_types(registry)
        # The stale entry has member "a" only; the document declares a and b.
        cache = WrongShapeCache({"parts.a": registry.wrap("core.string", "stale")})
        engine = Engine(
            schemas=build_schemas(SCAFFOLD_NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(SCAFFOLD_NODES), registry),
            cache=cache,
        )
        result = await engine.run(split_graph(("a", "b")), ["s"])
        assert result.executed == ("s",)  # treated as a miss, re-executed
        assert result.outputs["s"]["parts.b"].resolve() == "b c"
        assert cache.puts == 1  # and overwritten

    asyncio.run(scenario())


class PlainStatic(Node):
    """execute() takes exactly its declared inputs: injecting output_spec
    into a static node would TypeError here."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.plain_static",
            inputs=(InputSpec("a", STRING),),
            outputs=(OutputSpec("out", STRING),),
        )

    @classmethod
    def execute(cls, *, a: str) -> Mapping[str, object]:
        return cls.outputs(out=a)


def test_static_nodes_never_receive_output_spec() -> None:
    async def scenario() -> None:
        engine = make_engine([PlainStatic])
        graph = Graph(nodes={"n": GraphNode("test.plain_static", {"a": "v"})})
        result = await engine.run(graph, ["n"])
        assert result.outputs["n"]["out"].resolve() == "v"

    asyncio.run(scenario())


# --- the run-scoped snapshot: topology is fixed at run() entry ---


def test_snapshot_graph_is_deep_and_immutable() -> None:
    inputs: dict[str, object] = {"seed": "s"}
    members: dict[str, tuple[str, ...]] = {"results": ("a",)}
    graph = Graph(nodes={"n": GraphNode("test.fanout", inputs, output_members=members)})
    snap = snapshot_graph(graph)

    inputs["seed"] = "MUTATED"
    inputs["new_key"] = "x"
    members["results"] = ("a", "b", "c")

    node = snap.nodes["n"]
    assert isinstance(node, GraphNode)
    assert dict(node.inputs) == {"seed": "s"}
    assert dict(node.output_members) == {"results": ("a",)}
    with pytest.raises(TypeError):
        node.inputs["seed"] = "nope"  # type: ignore[index]
    with pytest.raises(TypeError):
        node.output_members["results"] = ()  # type: ignore[index]


class MutatingCache(MemoryLRUCache):
    """Simulates a frontend mutating the document mid-run: the mutation runs
    while the engine is inside an awaited cache call."""

    def __init__(self, mutate: object) -> None:
        super().__init__()
        self._mutate = mutate

    async def get(self, key: str):  # type: ignore[override]
        self._mutate()  # type: ignore[operator]
        return await super().get(key)


def test_run_executes_the_snapshot_even_if_the_document_mutates_mid_run() -> None:
    async def scenario() -> None:
        inputs: dict[str, object] = {"pieces.a": "x", "pieces.b": "y", "separator": " "}
        members: dict[str, tuple[str, ...]] = {"parts": ("a", "b")}
        graph = Graph(
            nodes={
                "s": GraphNode(
                    "std.string.split",
                    {"text": "q r", "separator": " "},
                    output_members=members,
                ),
                "j": GraphNode("std.string.join", inputs),
            }
        )

        def mutate() -> None:
            inputs["pieces.a"] = "MUTATED"
            inputs["pieces.c"] = "NEW MEMBER"
            members["parts"] = ("a",)  # shrink the output family mid-run

        registry = TypeRegistry()
        register_core_types(registry)
        register_scaffold_types(registry)
        engine = Engine(
            schemas=build_schemas(SCAFFOLD_NODES),
            registry=registry,
            worker=InProcessWorker(build_node_types(SCAFFOLD_NODES), registry),
            cache=MutatingCache(mutate),
        )
        result = await engine.run(graph, ["s", "j"])
        # The run reflects the document as of run() entry, not the mutations.
        assert result.outputs["j"]["text"].resolve() == "x y"
        assert set(result.outputs["s"]) == {"parts.a", "parts.b"}

    asyncio.run(scenario())
