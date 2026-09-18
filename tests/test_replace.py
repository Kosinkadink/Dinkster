"""Replacement rules: the declarative migration vocabulary (DESIGN 3.1).

Field-for-field mirror of Dinkster-Frontend's replace/model.ts. The tests
freeze three contracts: the exact wire shapes (the frontend cross-checks
field names 1:1 against these), the closed-vocabulary validation (unknown
kinds and malformed shapes reject at construction and at decode), and the
lifecycle-metadata rule (rules ride the schema wire but never the schema
signature).
"""

import dataclasses
import json
from typing import Any, cast

import pytest
from dinkster_schema import (
    DynamicComboOption,
    DynamicComboSpec,
    DynamicSlotSpec,
    InputFamilyMapping,
    InputFamilyMember,
    InputFamilySpec,
    InputSpec,
    MappingSource,
    NodeSchema,
    OutputCountSpec,
    OutputFamilyMapping,
    OutputFamilyMember,
    OutputFamilySpec,
    OutputSpec,
    ReplacementCase,
    ReplacementLink,
    ReplacementMigration,
    ReplacementNode,
    ReplacementPredicate,
    ReplacementRule,
    SlotVariant,
    TypeExpr,
    ValueTransform,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
    validate_replacement_references,
)
from dinkster_schema.replace import rule_from_wire, rule_to_wire

INT = TypeExpr.concrete("core.int")


def _rule(*cases: ReplacementCase, from_type: str = "old.node") -> ReplacementRule:
    return ReplacementRule(from_type=from_type, cases=tuple(cases))


def _wire(*cases: ReplacementCase) -> Any:
    """Encode a rule and hand back the wire as Any for shape assertions."""
    return cast("Any", rule_to_wire(_rule(*cases)))


FALLBACK = ReplacementCase.build("new.node")


# -- predicate wire shapes ----------------------------------------------------


def test_predicate_wire_shapes() -> None:
    """Every predicate kind encodes exactly the TS union member's fields."""
    p = ReplacementPredicate
    assert _wire(ReplacementCase.build("new.node", when=p.always()), FALLBACK)["cases"][0][
        "when"
    ] == {"kind": "always"}
    cases = {
        p.input_connected("image"): {"kind": "inputConnected", "input": "image"},
        p.value_present("seed"): {"kind": "valuePresent", "input": "seed"},
        p.value_equals("mode", "linear"): {
            "kind": "valueEquals",
            "input": "mode",
            "value": "linear",
        },
        p.value_equals("mask", None): {
            "kind": "valueEquals",
            "input": "mask",
            "value": None,
        },
        p.not_(p.value_present("seed")): {
            "kind": "not",
            "of": {"kind": "valuePresent", "input": "seed"},
        },
        p.all_of(p.value_present("a"), p.input_connected("b")): {
            "kind": "all",
            "of": [
                {"kind": "valuePresent", "input": "a"},
                {"kind": "inputConnected", "input": "b"},
            ],
        },
        p.any_of(): {"kind": "any", "of": []},
    }
    for predicate, wire in cases.items():
        case = ReplacementCase.build("new.node", when=predicate)
        assert _wire(case, FALLBACK)["cases"][0]["when"] == wire


def test_stale_target_choices_field_refuses_decode() -> None:
    wire = _wire(FALLBACK)
    wire["cases"][0]["targetChoices"] = []
    with pytest.raises(ValueError, match="targetChoices"):
        rule_from_wire(wire)


def test_predicate_deep_recursion_roundtrips() -> None:
    p = ReplacementPredicate
    deep = p.not_(p.all_of(p.any_of(p.value_equals("x", [1, {"y": None}]), p.always())))
    rule = _rule(ReplacementCase.build("new.node", when=deep), FALLBACK)
    assert rule_from_wire(rule_to_wire(rule)) == rule


def test_predicate_validation() -> None:
    p = ReplacementPredicate
    with pytest.raises(ValueError, match="unknown predicate kind"):
        p(kind="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires an input"):
        p(kind="inputConnected")
    with pytest.raises(ValueError, match="exactly one child"):
        p(kind="not")
    with pytest.raises(ValueError, match="carries no input"):
        p(kind="all", input="x")
    with pytest.raises(ValueError, match="carries no fields"):
        p(kind="always", input="x")
    with pytest.raises(ValueError, match="carries no value"):
        p(kind="valuePresent", input="x", value=3)
    with pytest.raises(ValueError, match="JSON-serializable"):
        p.value_equals("x", object())


# -- transforms and mapping sources -------------------------------------------


def test_transform_wire_shapes() -> None:
    rename = ValueTransform.enum_rename({"lcm": "lcm_scheduler", "ddim": "ddim_v2"})
    assert _wire(
        ReplacementCase.build(
            "new.node", inputs={"sched": MappingSource.from_value("sched", rename)}
        ),
    )["cases"][0]["inputs"]["sched"] == {
        "kind": "value",
        "input": "sched",
        "transform": {
            "kind": "enumRename",
            "map": {"lcm": "lcm_scheduler", "ddim": "ddim_v2"},
        },
    }
    scaled = MappingSource.from_value("cfg", ValueTransform.scale(0.5, offset=1.0))
    wire = _wire(ReplacementCase.build("new.node", inputs={"cfg": scaled}))
    entry = wire["cases"][0]["inputs"]["cfg"]
    assert entry["transform"] == {"kind": "scale", "factor": 0.5, "offset": 1.0}
    # offset omitted at its default, like every optional wire field
    plain = MappingSource.from_value("cfg", ValueTransform.scale(2.0))
    wire = _wire(ReplacementCase.build("new.node", inputs={"cfg": plain}))
    entry = wire["cases"][0]["inputs"]["cfg"]
    assert entry["transform"] == {"kind": "scale", "factor": 2.0}


def test_transform_validation() -> None:
    with pytest.raises(ValueError, match="unknown transform kind"):
        ValueTransform(kind="clamp")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no factor/offset"):
        ValueTransform(kind="enumRename", factor=2.0)
    with pytest.raises(ValueError, match="no map"):
        ValueTransform(kind="scale", map=(("a", "b"),))
    with pytest.raises(ValueError, match="finite numbers"):
        ValueTransform.scale(float("inf"))
    with pytest.raises(ValueError, match="finite numbers"):
        ValueTransform.scale(10**400)
    with pytest.raises(ValueError, match="must be strings"):
        ValueTransform(kind="enumRename", map=(("a", 3),))  # type: ignore[arg-type]


def test_mapping_source_wire_shapes() -> None:
    sources = {
        MappingSource.copy("image"): {"kind": "copy", "input": "image"},
        MappingSource.from_value("seed"): {"kind": "value", "input": "seed"},
        MappingSource.link("model"): {"kind": "link", "input": "model"},
        MappingSource.constant(42): {"kind": "constant", "value": 42},
        MappingSource.constant(None): {"kind": "constant", "value": None},
    }
    for source, wire in sources.items():
        case = ReplacementCase.build("new.node", inputs={"target": source})
        assert _wire(case)["cases"][0]["inputs"]["target"] == wire


def test_mapping_source_validation() -> None:
    with pytest.raises(ValueError, match="unknown mapping source kind"):
        MappingSource(kind="merge", input="a")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires a source input"):
        MappingSource(kind="copy")
    with pytest.raises(ValueError, match="carries no constant value"):
        MappingSource(kind="link", input="a", value=3)
    with pytest.raises(ValueError, match="carries no transform"):
        MappingSource(kind="copy", input="a", transform=ValueTransform.scale(2.0))
    with pytest.raises(ValueError, match="only a value"):
        MappingSource(kind="constant", input="a", value=3)
    with pytest.raises(ValueError, match="JSON-serializable"):
        MappingSource.constant({1, 2})
    with pytest.raises(ValueError, match="JSON-serializable"):
        MappingSource.constant(float("nan"))


def test_input_family_mapping_wire_shapes_and_roundtrip() -> None:
    copied = ReplacementCase.build(
        "new.node",
        input_families={
            "values": InputFamilyMapping.copy(
                "operands", inputs={"value": MappingSource.copy("item")}
            )
        },
    )
    copied_wire = _wire(copied)
    assert copied_wire["cases"][0]["inputFamilies"] == {
        "values": {
            "kind": "copy",
            "sourceFamily": "operands",
            "inputs": {"value": {"kind": "copy", "input": "item"}},
        }
    }
    assert rule_from_wire(copied_wire) == _rule(copied)

    explicit = ReplacementCase.build(
        "new.node",
        input_families={
            "values": InputFamilyMapping.from_members(
                InputFamilyMember.build("value1", inputs={"value": MappingSource.copy("a")}),
                InputFamilyMember.build("value2", inputs={"value": MappingSource.constant(False)}),
            )
        },
    )
    explicit_wire = _wire(explicit)
    assert explicit_wire["cases"][0]["inputFamilies"] == {
        "values": {
            "kind": "members",
            "members": [
                {
                    "suffix": "value1",
                    "inputs": {"value": {"kind": "copy", "input": "a"}},
                },
                {
                    "suffix": "value2",
                    "inputs": {"value": {"kind": "constant", "value": False}},
                },
            ],
        }
    }
    assert rule_from_wire(explicit_wire) == _rule(explicit)


def test_input_family_mapping_validation() -> None:
    member = InputFamilyMember.build("value1", inputs={"value": MappingSource.copy("a")})
    with pytest.raises(ValueError, match="unknown input family mapping kind"):
        InputFamilyMapping(kind="rename")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source family id"):
        InputFamilyMapping.copy("", inputs={"value": MappingSource.copy("value")})
    with pytest.raises(ValueError, match="template input mappings"):
        InputFamilyMapping.copy("values", inputs={})
    with pytest.raises(ValueError, match="requires explicit members"):
        InputFamilyMapping.from_members()
    with pytest.raises(ValueError, match="duplicate suffixes"):
        InputFamilyMapping.from_members(member, member)
    with pytest.raises(ValueError, match="member suffix"):
        InputFamilyMember.build("bad.suffix", inputs={"value": MappingSource.copy("a")})
    with pytest.raises(ValueError, match="requires input mappings"):
        InputFamilyMember.build("value1", inputs={})


def test_output_family_mapping_wire_shapes_and_roundtrip() -> None:
    copied = ReplacementCase.build(
        "new.node",
        output_families={"values": OutputFamilyMapping.copy("results")},
    )
    copied_wire = _wire(copied)
    assert copied_wire["cases"][0]["outputFamilies"] == {
        "values": {"kind": "copy", "sourceFamily": "results"}
    }
    assert rule_from_wire(copied_wire) == _rule(copied)

    explicit = ReplacementCase.build(
        "new.node",
        output_families={
            "values": OutputFamilyMapping.from_members(
                OutputFamilyMember("0", "left"),
                OutputFamilyMember("1", "right"),
            )
        },
    )
    explicit_wire = _wire(explicit)
    assert explicit_wire["cases"][0]["outputFamilies"] == {
        "values": {
            "kind": "members",
            "members": [
                {"suffix": "0", "output": "left"},
                {"suffix": "1", "output": "right"},
            ],
        }
    }
    assert rule_from_wire(explicit_wire) == _rule(explicit)


def test_output_family_mapping_validation() -> None:
    member = OutputFamilyMember("0", "result")
    with pytest.raises(ValueError, match="unknown output family mapping kind"):
        OutputFamilyMapping(kind="rename")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source family id"):
        OutputFamilyMapping.copy("")
    with pytest.raises(ValueError, match="requires explicit members"):
        OutputFamilyMapping.from_members()
    with pytest.raises(ValueError, match="duplicate suffixes"):
        OutputFamilyMapping.from_members(member, member)
    with pytest.raises(ValueError, match="member suffix"):
        OutputFamilyMember("bad.suffix", "result")
    with pytest.raises(ValueError, match="source output id"):
        OutputFamilyMember("0", "bad.output")


# -- cases and rules -----------------------------------------------------------


def test_case_and_rule_validation() -> None:
    with pytest.raises(ValueError, match="target node type"):
        ReplacementCase.build("")
    with pytest.raises(ValueError, match="source node type"):
        ReplacementRule(from_type="", cases=(FALLBACK,))
    with pytest.raises(ValueError, match="at least one case"):
        ReplacementRule(from_type="old.node", cases=())
    with pytest.raises(ValueError, match="last case must be unconditional"):
        _rule(ReplacementCase.build("new.node", when=ReplacementPredicate.value_present("x")))
    # explicit {"kind":"always"} counts as unconditional, matching the TS
    # validator ("when omitted or kind 'always'")
    _rule(ReplacementCase.build("new.node", when=ReplacementPredicate.always()))
    # a source output feeding two target outputs would duplicate links
    with pytest.raises(ValueError, match="feeds two target outputs"):
        ReplacementCase.build("new.node", outputs={"out_a": "out", "out_b": "out"})
    with pytest.raises(ValueError, match="feeds two target outputs"):
        ReplacementCase.build(
            "new.node",
            outputs={"out": "result"},
            output_families={
                "items": OutputFamilyMapping.from_members(OutputFamilyMember("0", "result"))
            },
        )
    with pytest.raises(ValueError, match="feeds two target families"):
        ReplacementCase.build(
            "new.node",
            output_families={
                "first": OutputFamilyMapping.copy("results"),
                "second": OutputFamilyMapping.copy("results"),
            },
        )


def test_guarded_fanout_to_multiple_successors() -> None:
    """A rule's cases name their own targets: guarded fan-out is legitimate
    and never constrained to the carrying schema's node type."""
    rule = ReplacementRule(
        from_type="old.node",
        note="split in v0.4",
        cases=(
            ReplacementCase.build(
                "new.node_a",
                when=ReplacementPredicate.input_connected("mask"),
                inputs={"mask": MappingSource.copy("mask")},
            ),
            ReplacementCase.build("new.node_b"),
        ),
    )
    wire = cast("Any", rule_to_wire(rule))
    assert wire["from"] == "old.node"
    assert wire["note"] == "split in v0.4"
    assert [case["to"] for case in wire["cases"]] == ["new.node_a", "new.node_b"]
    assert rule_from_wire(wire) == rule
    # ...and it rides a schema whose node_type matches neither target
    carrier = NodeSchema(node_type="pack.registrar", replacements=(rule,))
    assert schema_to_wire(carrier)["replacements"] == [wire]


def test_rule_wire_roundtrip_full() -> None:
    rule = ReplacementRule(
        from_type="old.sampler",
        cases=(
            ReplacementCase.build(
                "new.sampler",
                when=ReplacementPredicate.all_of(
                    ReplacementPredicate.value_equals("scheduler", "karras"),
                    ReplacementPredicate.not_(ReplacementPredicate.input_connected("latent")),
                ),
                inputs={
                    "model": MappingSource.copy("model"),
                    "steps": MappingSource.from_value(
                        "steps", ValueTransform.scale(2.0, offset=-1.0)
                    ),
                    "noise": MappingSource.link("noise"),
                    "mode": MappingSource.constant("default"),
                },
                outputs={"latent_out": "latent", "info": "meta"},
            ),
            ReplacementCase.build("new.sampler"),
        ),
    )
    wire = cast("Any", rule_to_wire(rule))
    assert rule_from_wire(wire) == rule
    # optional fields are omitted, never null
    assert "note" not in wire
    assert "when" not in wire["cases"][1]
    assert "inputs" not in wire["cases"][1]
    assert "outputs" not in wire["cases"][1]


def test_same_type_migration_wire_and_validation() -> None:
    migration = ReplacementMigration(("legacy.policy", "color_source", "color_value"))
    rule = ReplacementRule(
        from_type="new.node",
        migration=migration,
        cases=(
            ReplacementCase.build(
                "new.node",
                slot_variants={"policy": "exact_color", "policy.color_source": "integer"},
                inputs={"policy.color_source.color_value": MappingSource.copy("color_value")},
            ),
        ),
    )
    wire = cast("Any", rule_to_wire(rule))
    assert wire["migration"] == {
        "historicalInputs": ["legacy.policy", "color_source", "color_value"]
    }
    assert rule_from_wire(wire) == rule

    with pytest.raises(ValueError, match="requires historical input ids"):
        ReplacementMigration(())
    with pytest.raises(ValueError, match="requires historical input ids"):
        ReplacementMigration(["value"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate historical input ids"):
        ReplacementMigration(("value", "value"))
    with pytest.raises(ValueError, match="invalid historical input id"):
        ReplacementMigration(("bad..path",))
    with pytest.raises(ValueError, match="every case must preserve the node type"):
        ReplacementRule(
            from_type="old.node",
            migration=ReplacementMigration(("value",)),
            cases=(ReplacementCase.build("new.node"),),
        )
    with pytest.raises(ValueError, match="must be ReplacementMigration"):
        ReplacementRule(
            from_type="new.node",
            migration={"historicalInputs": ["value"]},  # type: ignore[arg-type]
            cases=(ReplacementCase.build("new.node"),),
        )


@pytest.mark.parametrize(
    "migration",
    (
        None,
        [],
        {"historicalInputs": []},
        {"historicalInputs": ["value"], "extra": True},
        {"historicalInputs": "value"},
        {"historicalInputs": [1]},
        {"historicalInputs": ["value", "value"]},
    ),
)
def test_migration_wire_rejects_malformed_data(migration: object) -> None:
    with pytest.raises(ValueError):
        rule_from_wire(
            {
                "from": "same.node",
                "migration": migration,
                "cases": [{"to": "same.node"}],
            }
        )


def test_legacy_cases_without_nodes_encode_byte_identically() -> None:
    case = ReplacementCase.build(
        "new.node",
        inputs={"legacy:input": MappingSource.copy("picture")},
        outputs={"legacy:output": "result"},
    )
    encoded = json.dumps(rule_to_wire(_rule(case)), separators=(",", ":"))
    assert encoded == (
        '{"from":"old.node","cases":[{"to":"new.node",'
        '"inputs":{"legacy:input":{"kind":"copy","input":"picture"}},'
        '"outputs":{"legacy:output":"result"}}]}'
    )
    assert rule_from_wire(json.loads(encoded)) == _rule(case)


def test_one_to_n_case_round_trips_with_nodes_links_and_addresses() -> None:
    case = ReplacementCase.build(
        "new.node",
        nodes={"join": ReplacementNode.build("helper.join", values={"count": 2})},
        inputs={"join:item": MappingSource.copy("picture")},
        links=(ReplacementLink("join:items", "image"),),
        outputs={"join:preview": "result"},
    )
    wire = cast("Any", rule_to_wire(_rule(case)))
    assert wire["cases"][0] == {
        "to": "new.node",
        "nodes": {"join": {"type": "helper.join", "values": {"count": 2}}},
        "inputs": {"join:item": {"kind": "copy", "input": "picture"}},
        "links": [{"from": "join:items", "to": "image"}],
        "outputs": {"join:preview": "result"},
    }
    assert rule_from_wire(wire) == _rule(case)
    empty = _rule(ReplacementCase.build("new.node", nodes={}))
    empty_wire = rule_to_wire(empty)
    assert empty_wire["cases"][0]["nodes"] == {}  # type: ignore[index]
    assert rule_from_wire(empty_wire).cases[0].nodes == ()


def test_helper_node_and_address_construction_error_matrix() -> None:
    node = ReplacementNode.build("helper.node")
    failures = (
        lambda: ReplacementCase("new.node", nodes=(("bad.id", node),)),
        lambda: ReplacementCase("new.node", nodes=(("dup", node), ("dup", node))),
        lambda: ReplacementNode.build(""),
        lambda: ReplacementNode.build("helper.node", values={"bad:port": 1}),
        lambda: ReplacementNode.build("helper.node", values={"bad..port": 1}),
        lambda: ReplacementNode.build("helper.node", values={"x": object()}),
        lambda: ReplacementNode.build("helper.node", values={"x": float("nan")}),
        lambda: ReplacementNode.build("helper.node", values={"x": float("inf")}),
        lambda: ReplacementNode.build("helper.node", values={"x": float("-inf")}),
        lambda: ReplacementCase.build("new.node", inputs={"": MappingSource.copy("x")}),
        lambda: ReplacementCase.build(
            "new.node", nodes={"h": node}, inputs={"h:x:y": MappingSource.copy("x")}
        ),
        lambda: ReplacementCase.build(
            "new.node", nodes={"h": node}, inputs={":x": MappingSource.copy("x")}
        ),
        lambda: ReplacementCase.build("new.node", nodes={"h": node}, outputs={"h:": "out"}),
        lambda: ReplacementCase.build(
            "new.node", nodes={"h": node}, inputs={"other:x": MappingSource.copy("x")}
        ),
        lambda: ReplacementCase.build("new.node", nodes={}, outputs={"h:x": "out"}),
    )
    for construct in failures:
        with pytest.raises(ValueError):
            construct()


def test_slot_variants_wire_shape_and_validation() -> None:
    case = ReplacementCase.build(
        "new.node",
        nodes={"helper": ReplacementNode.build("helper.node")},
        slot_variants={
            "policy": "tolerance_color",
            "policy.color_source": "integer",
            "helper:source": "source_kind",
        },
    )
    wire = _wire(case)
    assert wire["cases"][0]["slotVariants"] == {
        "policy": "tolerance_color",
        "policy.color_source": "integer",
        "helper:source": "source_kind",
    }
    assert rule_from_wire(wire) == _rule(case)

    node = ReplacementNode.build("helper.node")
    failures = (
        lambda: ReplacementCase(
            "new.node",
            slot_variants=(
                ("policy", "a"),
                ("policy", "b"),
            ),
        ),
        lambda: ReplacementCase.build("new.node", slot_variants={"": "active"}),
        lambda: ReplacementCase.build(
            "new.node",
            slot_variants={"policy": 1},  # type: ignore[dict-item]
        ),
        lambda: ReplacementCase.build(
            "new.node",
            nodes={"helper": node},
            slot_variants={"missing:source": "number"},
        ),
        lambda: ReplacementCase(
            "new.node",
            slot_variants=(("policy", ""),),
        ),
        lambda: ReplacementCase.build("new.node", slot_variants={".policy": "active"}),
        lambda: ReplacementCase.build("new.node", slot_variants={"policy.": "active"}),
        lambda: ReplacementCase.build("new.node", slot_variants={"policy..mode": "active"}),
        lambda: ReplacementCase.build("new.node", slot_variants={"helper:mode": "active"}),
    )
    for construct in failures:
        with pytest.raises(ValueError):
            construct()

    for malformed in ({"policy": 1}, ["policy", {"kind": "constant", "value": "active"}]):
        with pytest.raises(ValueError):
            rule_from_wire(
                {"from": "old.node", "cases": [{"to": "new.node", "slotVariants": malformed}]}
            )
    with pytest.raises(ValueError, match="unexpected fields: targetChoices"):
        rule_from_wire(
            {
                "from": "old.node",
                "cases": [
                    {
                        "to": "new.node",
                        "targetChoices": {"policy": {"kind": "constant", "value": "active"}},
                    }
                ],
            }
        )


def test_single_feeder_and_fan_in_rules() -> None:
    node = ReplacementNode.build("helper.node")
    with pytest.raises(ValueError, match="two feeders"):
        ReplacementCase.build(
            "new.node",
            nodes={"h": node},
            inputs={"image": MappingSource.copy("picture")},
            links=(ReplacementLink("h:out", "image"),),
        )
    with pytest.raises(ValueError, match="two feeders"):
        ReplacementCase.build(
            "new.node",
            nodes={"h": node},
            links=(
                ReplacementLink("h:out", "image"),
                ReplacementLink("h:other", "image"),
            ),
        )
    with pytest.raises(ValueError, match="feeds two target outputs"):
        ReplacementCase.build("new.node", outputs={"out_a": "source", "out_b": "source"})


def test_internal_links_must_be_acyclic() -> None:
    node = ReplacementNode.build("helper.node")
    with pytest.raises(ValueError, match="acyclic"):
        ReplacementCase.build(
            "new.node",
            nodes={"h": node},
            links=(
                ReplacementLink("out", "h:input"),
                ReplacementLink("h:out", "input"),
            ),
        )
    with pytest.raises(ValueError, match="acyclic"):
        ReplacementCase.build(
            "new.node",
            nodes={"h": node},
            links=(ReplacementLink("h:out", "h:input"),),
        )
    with pytest.raises(ValueError, match="acyclic"):
        ReplacementCase.build("new.node", links=(ReplacementLink("out", "input"),))


def test_decode_rejects_malformed_wire() -> None:
    good = rule_to_wire(_rule(FALLBACK))
    with pytest.raises(ValueError, match="unknown predicate kind"):
        rule_from_wire(
            {
                "from": "a",
                "cases": [{"to": "b", "when": {"kind": "whenever"}}, {"to": "b"}],
            }
        )
    with pytest.raises(ValueError, match="unknown transform kind"):
        rule_from_wire(
            {
                "from": "a",
                "cases": [
                    {
                        "to": "b",
                        "inputs": {
                            "x": {
                                "kind": "value",
                                "input": "y",
                                "transform": {"kind": "clamp"},
                            }
                        },
                    }
                ],
            }
        )
    with pytest.raises(ValueError, match="unknown mapping source kind"):
        rule_from_wire({"from": "a", "cases": [{"to": "b", "inputs": {"x": {"kind": "merge"}}}]})
    with pytest.raises(ValueError, match="unknown input family mapping kind"):
        rule_from_wire(
            {
                "from": "a",
                "cases": [
                    {
                        "to": "b",
                        "inputFamilies": {"items": {"kind": "rename"}},
                    }
                ],
            }
        )
    with pytest.raises(ValueError, match="unknown output family mapping kind"):
        rule_from_wire(
            {
                "from": "a",
                "cases": [
                    {
                        "to": "b",
                        "outputFamilies": {"items": {"kind": "rename"}},
                    }
                ],
            }
        )
    with pytest.raises(ValueError, match="exactly kind and sourceFamily"):
        rule_from_wire(
            {
                "from": "a",
                "cases": [
                    {
                        "to": "b",
                        "outputFamilies": {
                            "items": {
                                "kind": "copy",
                                "sourceFamily": "old_items",
                                "members": [],
                            }
                        },
                    }
                ],
            }
        )
    with pytest.raises(ValueError, match="exactly suffix and output"):
        rule_from_wire(
            {
                "from": "a",
                "cases": [
                    {
                        "to": "b",
                        "outputFamilies": {
                            "items": {
                                "kind": "members",
                                "members": [{"suffix": "0", "output": "value", "extra": True}],
                            }
                        },
                    }
                ],
            }
        )
    with pytest.raises((KeyError, TypeError)):
        rule_from_wire({"cases": [{"to": "b"}]})  # missing "from"
    with pytest.raises((KeyError, TypeError)):
        rule_from_wire({"from": "a"})  # missing "cases"
    with pytest.raises(ValueError, match="unconditional"):
        rule_from_wire(
            {
                "from": "a",
                "cases": [{"to": "b", "when": {"kind": "valuePresent", "input": "x"}}],
            }
        )
    assert rule_from_wire(good) == _rule(FALLBACK)


# -- schema wire + signature ---------------------------------------------------


SUCCESSOR = NodeSchema(
    node_type="new.node",
    inputs=(InputSpec("image", INT), InputSpec("mode", INT, required=False)),
    outputs=(OutputSpec("out", INT),),
    replacements=(
        ReplacementRule(
            from_type="old.node",
            cases=(
                ReplacementCase.build("new.node", inputs={"image": MappingSource.copy("picture")}),
            ),
        ),
    ),
)


def test_replacements_ride_the_schema_wire() -> None:
    wire = schema_to_wire(SUCCESSOR)
    assert wire["replacements"] == [
        {
            "from": "old.node",
            "cases": [
                {
                    "to": "new.node",
                    "inputs": {"image": {"kind": "copy", "input": "picture"}},
                }
            ],
        }
    ]
    assert schema_from_wire(wire) == SUCCESSOR
    plain = schema_to_wire(dataclasses.replace(SUCCESSOR, replacements=()))
    assert "replacements" not in plain


def test_replacements_never_join_the_signature() -> None:
    """Lifecycle metadata: shipping or changing migration advice must not
    invalidate the node's caches."""
    bare = dataclasses.replace(SUCCESSOR, replacements=())
    assert schema_signature(SUCCESSOR) == schema_signature(bare)


def test_wire_27_omits_dynamic_and_migration_rules_without_hiding_the_schema() -> None:
    dynamic_rule = _rule(
        ReplacementCase.build(
            "new.node",
            slot_variants={"policy": "tolerance_color"},
        )
    )
    migration_rule = ReplacementRule(
        from_type="new.node",
        migration=ReplacementMigration(("legacy",)),
        cases=(ReplacementCase.build("new.node"),),
    )
    dynamic = dataclasses.replace(
        SUCCESSOR,
        replacements=(*SUCCESSOR.replacements, dynamic_rule, migration_rule),
    )
    wire = schema_to_wire(dynamic)
    assert wire["schemaVersion"] == 40
    assert wire["replacements"][1]["cases"][0]["slotVariants"] == {  # type: ignore[index]
        "policy": "tolerance_color"
    }
    assert wire["replacements"][2]["migration"] == {  # type: ignore[index]
        "historicalInputs": ["legacy"]
    }
    assert schema_from_wire(wire) == dynamic
    for version in (26, 27):
        downgraded_wire = schema_to_wire(dynamic, wire_version=version)
        assert downgraded_wire["nodeType"] == "new.node"
        assert downgraded_wire["replacements"] == [rule_to_wire(SUCCESSOR.replacements[0])]

    downgraded = dict(wire)
    downgraded["schemaVersion"] = 27
    with pytest.raises(ValueError, match="slotVariants require schema wire 28"):
        schema_from_wire(downgraded)
    migration_only = schema_to_wire(dataclasses.replace(dynamic, replacements=(migration_rule,)))
    migration_only["schemaVersion"] = 27
    with pytest.raises(ValueError, match="migration requires schema wire 28"):
        schema_from_wire(migration_only)
    assert schema_signature(dynamic) == schema_signature(
        dataclasses.replace(dynamic, replacements=())
    )


def test_wire_27_omits_rules_that_target_open_dynamic_slots() -> None:
    dynamic_rule = _rule(
        ReplacementCase.build(
            "new.open-slot",
            inputs={"source": MappingSource.copy("picture")},
        )
    )
    target = NodeSchema(
        node_type="new.open-slot",
        slots=(DynamicSlotSpec("source", slot_type=INT),),
        replacements=(dynamic_rule,),
    )

    assert schema_to_wire(target)["replacements"] == [rule_to_wire(dynamic_rule)]
    predecessor = schema_to_wire(target, wire_version=27)
    assert predecessor["nodeType"] == target.node_type
    assert "replacements" not in predecessor


# -- cross-schema reference checks ----------------------------------------------


PREDECESSOR = NodeSchema(
    node_type="old.node",
    inputs=(InputSpec("picture", INT), InputSpec("mode", INT, required=False)),
    outputs=(OutputSpec("result", INT),),
    input_families=(InputFamilySpec("operands", INT),),
)


def _successor_with(rule: ReplacementRule) -> NodeSchema:
    return dataclasses.replace(SUCCESSOR, replacements=(rule,))


def test_reference_check_clean() -> None:
    rule = ReplacementRule(
        from_type="old.node",
        cases=(
            ReplacementCase.build(
                "new.node",
                when=ReplacementPredicate.value_present("mode"),
                inputs={
                    "image": MappingSource.copy("picture"),
                    "mode": MappingSource.constant("fast"),
                },
                outputs={"out": "result"},
            ),
            ReplacementCase.build("new.node"),
        ),
    )
    schemas = {"old.node": PREDECESSOR, "new.node": _successor_with(rule)}
    assert validate_replacement_references(schemas) == ()


def _dynamic_target_schema() -> NodeSchema:
    return NodeSchema(
        node_type="new.dynamic",
        inputs=(InputSpec("fixed", INT),),
        outputs=(OutputSpec("out", INT),),
        combos=(
            DynamicComboSpec(
                "policy",
                options=(
                    DynamicComboOption("channel", inputs=(InputSpec("channel", INT),)),
                    DynamicComboOption(
                        "tolerance_color",
                        inputs=(
                            DynamicComboSpec(
                                "color_source",
                                options=(
                                    DynamicComboOption(
                                        "integer", inputs=(InputSpec("color_value", INT),)
                                    ),
                                    DynamicComboOption(
                                        "channels",
                                        inputs=(
                                            InputSpec("red", INT),
                                            InputSpec("green", INT),
                                            InputSpec("blue", INT),
                                        ),
                                    ),
                                ),
                            ),
                            InputSpec("tolerance", INT),
                            InputSpec("metric", INT),
                        ),
                    ),
                ),
            ),
        ),
        slots=(
            DynamicSlotSpec(
                "source",
                variants=(SlotVariant("number", INT, inputs=(InputSpec("scale", INT),)),),
            ),
        ),
    )


def test_reference_check_accepts_declared_historical_source_inputs() -> None:
    target = _dynamic_target_schema()
    selected = {
        "policy": "tolerance_color",
        "policy.color_source": "integer",
        "source": "number",
    }
    inputs = {
        "fixed": MappingSource.copy("fixed"),
        "policy.color_source.color_value": MappingSource.copy("old_color_value"),
    }
    rule = ReplacementRule(
        from_type=target.node_type,
        migration=ReplacementMigration(("old_policy", "old_color_value")),
        cases=(
            ReplacementCase.build(
                target.node_type,
                when=ReplacementPredicate.value_equals("old_policy", "tolerance_color"),
                slot_variants=selected,
                inputs=inputs,
            ),
            ReplacementCase.build(
                target.node_type,
                slot_variants=selected,
                inputs=inputs,
            ),
        ),
    )
    schema = dataclasses.replace(target, replacements=(rule,))
    assert validate_replacement_references({schema.node_type: schema}) == ()

    bad_rule = dataclasses.replace(
        rule,
        cases=(
            rule.cases[0],
            dataclasses.replace(
                rule.cases[1],
                inputs=(
                    (
                        "policy.color_source.color_value",
                        MappingSource.copy("old_color_typo"),
                    ),
                ),
            ),
        ),
    )
    bad_schema = dataclasses.replace(schema, replacements=(bad_rule,))
    problems = validate_replacement_references({bad_schema.node_type: bad_schema})
    assert len(problems) == 1
    assert problems[0].ref == "old_color_typo"
    assert "not a static input id of new.dynamic" in problems[0].message


def test_reference_check_elaborates_nested_dynamic_target_paths() -> None:
    target = _dynamic_target_schema()
    rule = _rule(
        ReplacementCase.build(
            target.node_type,
            slot_variants={
                "policy": "tolerance_color",
                "policy.color_source": "integer",
                "source": "number",
            },
            inputs={
                "fixed": MappingSource.copy("picture"),
                "policy.color_source.color_value": MappingSource.from_value("picture"),
                "policy.tolerance": MappingSource.constant(0),
                "policy.metric": MappingSource.constant("euclidean"),
                "source": MappingSource.from_value("picture"),
                "source.scale": MappingSource.constant(1),
            },
        )
    )
    schemas = {
        "old.node": PREDECESSOR,
        target.node_type: dataclasses.replace(target, replacements=(rule,)),
    }
    assert validate_replacement_references(schemas) == ()

    inactive = _rule(
        ReplacementCase.build(
            target.node_type,
            slot_variants={
                "policy": "tolerance_color",
                "policy.color_source": "integer",
                "source": "number",
            },
            inputs={
                "policy.channel": MappingSource.from_value("picture"),
                "source": MappingSource.copy("picture"),
                "source.scale": MappingSource.constant(1),
            },
        )
    )
    schemas[target.node_type] = dataclasses.replace(target, replacements=(inactive,))
    problems = validate_replacement_references(schemas)
    assert len(problems) == 1
    assert problems[0].ref == "policy.channel"
    assert "selected interface" in problems[0].message

    unknown = _rule(
        ReplacementCase.build(
            target.node_type,
            slot_variants={
                "policy": "tolerance_color",
                "policy.color_source": "missing",
                "source": "number",
            },
            inputs={"source": MappingSource.copy("picture")},
        )
    )
    schemas[target.node_type] = dataclasses.replace(target, replacements=(unknown,))
    problems = validate_replacement_references(schemas)
    assert len(problems) == 1
    assert problems[0].ref == "policy.color_source"
    assert "names unknown choice" in problems[0].message


def test_reference_check_requires_dynamic_target_choices() -> None:
    target = _dynamic_target_schema()
    rule = _rule(ReplacementCase.build(target.node_type))
    schemas = {
        PREDECESSOR.node_type: PREDECESSOR,
        target.node_type: dataclasses.replace(target, replacements=(rule,)),
    }
    problems = validate_replacement_references(schemas)
    assert len(problems) == 1
    assert "required dynamic construct 'policy' has no stored choice" in problems[0].message


def test_dynamic_slot_link_choice_preserves_the_source_connection() -> None:
    mask = TypeExpr.concrete("core.mask")
    source = NodeSchema(
        node_type="old.masked",
        inputs=(InputSpec("mask", mask, required=False), InputSpec("mask_polarity", INT)),
    )
    target = NodeSchema(
        node_type="new.masked",
        slots=(
            DynamicSlotSpec(
                "mask",
                variants=(SlotVariant("mask", mask, inputs=(InputSpec("mask_polarity", INT),)),),
                required=False,
            ),
        ),
    )
    rule = ReplacementRule(
        from_type=source.node_type,
        cases=(
            ReplacementCase.build(
                target.node_type,
                slot_variants={"mask": "mask"},
                inputs={
                    "mask": MappingSource.copy("mask"),
                    "mask.mask_polarity": MappingSource.copy("mask_polarity"),
                },
            ),
        ),
    )
    schemas = {
        source.node_type: source,
        target.node_type: dataclasses.replace(target, replacements=(rule,)),
    }
    assert validate_replacement_references(schemas) == ()
    assert rule_from_wire(rule_to_wire(rule)) == rule

    connected = ReplacementCase.build(
        target.node_type,
        slot_variants={"mask": "mask"},
        inputs={"mask": MappingSource.copy("mask")},
    )
    assert connected.inputs == (("mask", MappingSource.copy("mask")),)


def test_dynamic_slot_link_choice_filters_incompatible_variants() -> None:
    mask = TypeExpr.concrete("core.mask")
    image = TypeExpr.concrete("core.image")
    source = NodeSchema(
        node_type="old.typed-slot",
        inputs=(InputSpec("source", mask),),
    )
    target = NodeSchema(
        node_type="new.typed-slot",
        slots=(
            DynamicSlotSpec(
                "source",
                variants=(
                    SlotVariant("mask", mask, inputs=(InputSpec("mask_cfg", INT),)),
                    SlotVariant("image", image, inputs=(InputSpec("image_cfg", INT),)),
                ),
            ),
        ),
    )
    rule = ReplacementRule(
        from_type=source.node_type,
        cases=(
            ReplacementCase.build(
                target.node_type,
                slot_variants={"source": "mask"},
                inputs={"source.image_cfg": MappingSource.constant(1)},
            ),
        ),
    )
    schemas = {
        source.node_type: source,
        target.node_type: dataclasses.replace(target, replacements=(rule,)),
    }
    problems = validate_replacement_references(schemas)
    assert any(
        problem.ref == "source.image_cfg"
        and "not an input id of the selected interface" in problem.message
        for problem in problems
    )


@pytest.mark.parametrize(
    ("left_mapping", "right_mapping"),
    (
        (MappingSource.link("shared"), MappingSource.link("shared")),
        (MappingSource.link("shared"), MappingSource.copy("shared")),
        (MappingSource.copy("shared"), MappingSource.copy("shared")),
    ),
)
def test_selected_slots_cannot_consume_one_source_connection_twice(
    left_mapping: MappingSource,
    right_mapping: MappingSource,
) -> None:
    source = NodeSchema(
        node_type="old.shared-slot",
        inputs=(InputSpec("shared", INT),),
    )
    target = NodeSchema(
        node_type="new.shared-slot",
        slots=(
            DynamicSlotSpec(
                "left",
                variants=(SlotVariant("integer", INT, inputs=(InputSpec("left_only", INT),)),),
            ),
            DynamicSlotSpec(
                "right",
                variants=(SlotVariant("integer", INT, inputs=(InputSpec("right_only", INT),)),),
            ),
        ),
    )
    rule = ReplacementRule(
        from_type=source.node_type,
        cases=(
            ReplacementCase.build(
                target.node_type,
                slot_variants={
                    "left": "integer",
                    "right": "integer",
                },
                inputs={
                    "left": left_mapping,
                    "left.left_only": MappingSource.constant(1),
                    "right": right_mapping,
                    "right.right_only": MappingSource.constant(2),
                },
            ),
        ),
    )
    schemas = {
        source.node_type: source,
        target.node_type: dataclasses.replace(target, replacements=(rule,)),
    }
    problems = validate_replacement_references(schemas)
    assert len(problems) == 1
    assert problems[0].ref == "shared"
    assert "consumed by multiple target mappings" in problems[0].message


def test_connection_consumption_is_accounted_across_inputs_and_members() -> None:
    source = NodeSchema(
        node_type="old.shared-input",
        inputs=(InputSpec("shared", INT),),
    )
    target = NodeSchema(
        node_type="new.shared-input",
        inputs=(InputSpec("fixed", INT),),
        slots=(
            DynamicSlotSpec(
                "choice",
                variants=(SlotVariant("integer", INT),),
            ),
        ),
        input_families=(InputFamilySpec("items", (InputSpec("value", INT),), min_members=0),),
    )
    cases = (
        ReplacementCase.build(
            target.node_type,
            slot_variants={"choice": "integer"},
            inputs={
                "choice": MappingSource.link("shared"),
                "fixed": MappingSource.copy("shared"),
            },
        ),
        ReplacementCase.build(
            target.node_type,
            slot_variants={"choice": "integer"},
            inputs={"fixed": MappingSource.link("shared")},
            input_families={
                "items": InputFamilyMapping.from_members(
                    InputFamilyMember.build(
                        "first",
                        inputs={"value": MappingSource.copy("shared")},
                    )
                )
            },
        ),
    )
    for case in cases:
        rule = ReplacementRule(from_type=source.node_type, cases=(case,))
        schemas = {
            source.node_type: source,
            target.node_type: dataclasses.replace(target, replacements=(rule,)),
        }
        problems = validate_replacement_references(schemas)
        assert len(problems) == 1
        assert problems[0].ref == "shared"
        assert "consumed by multiple target mappings" in problems[0].message


def test_connection_consumption_is_accounted_across_family_copy_mappings() -> None:
    source = NodeSchema(
        node_type="old.shared-family",
        input_families=(InputFamilySpec("items", (InputSpec("value", INT),), min_members=0),),
    )
    target = NodeSchema(
        node_type="new.shared-family",
        input_families=(
            InputFamilySpec("left", (InputSpec("value", INT),), min_members=0),
            InputFamilySpec("right", (InputSpec("value", INT),), min_members=0),
        ),
    )
    rule = ReplacementRule(
        from_type=source.node_type,
        cases=(
            ReplacementCase.build(
                target.node_type,
                input_families={
                    "left": InputFamilyMapping.copy(
                        "items", inputs={"value": MappingSource.link("value")}
                    ),
                    "right": InputFamilyMapping.copy(
                        "items", inputs={"value": MappingSource.copy("value")}
                    ),
                },
            ),
        ),
    )
    schemas = {
        source.node_type: source,
        target.node_type: dataclasses.replace(target, replacements=(rule,)),
    }
    problems = validate_replacement_references(schemas)
    assert len(problems) == 1
    assert problems[0].ref == "value"
    assert "items.*.value" in problems[0].message
    assert "consumed by multiple target mappings" in problems[0].message


def test_source_dependent_slot_variants_use_guarded_literal_cases() -> None:
    source = NodeSchema(
        node_type="old.guarded-slot",
        inputs=(InputSpec("kind", TypeExpr.concrete("core.string")), InputSpec("shared", INT)),
    )
    target = NodeSchema(
        node_type="new.guarded-slot",
        slots=(
            DynamicSlotSpec(
                "source",
                variants=(
                    SlotVariant("integer", INT),
                    SlotVariant("string", TypeExpr.concrete("core.string")),
                ),
            ),
        ),
    )
    rule = ReplacementRule(
        from_type=source.node_type,
        cases=(
            ReplacementCase.build(
                target.node_type,
                when=ReplacementPredicate.value_equals("kind", "integer"),
                slot_variants={"source": "integer"},
                inputs={"source": MappingSource.copy("shared")},
            ),
            ReplacementCase.build(
                target.node_type,
                slot_variants={"source": "string"},
                inputs={"source": MappingSource.copy("shared")},
            ),
        ),
    )
    schemas = {
        source.node_type: source,
        target.node_type: dataclasses.replace(target, replacements=(rule,)),
    }
    assert validate_replacement_references(schemas) == ()
    assert rule_to_wire(rule)["cases"] == [
        {
            "to": target.node_type,
            "when": {"kind": "valueEquals", "input": "kind", "value": "integer"},
            "slotVariants": {"source": "integer"},
            "inputs": {"source": {"kind": "copy", "input": "shared"}},
        },
        {
            "to": target.node_type,
            "slotVariants": {"source": "string"},
            "inputs": {"source": {"kind": "copy", "input": "shared"}},
        },
    ]


@pytest.mark.parametrize(
    ("slot_variants", "message"),
    (
        ({"missing": "number"}, "not a dynamic construct path"),
        ({"policy": "missing"}, "names unknown choice"),
    ),
)
def test_reference_check_rejects_invalid_slot_variants(
    slot_variants: dict[str, str], message: str
) -> None:
    target = _dynamic_target_schema()
    rule = _rule(ReplacementCase.build(target.node_type, slot_variants=slot_variants))
    schemas = {
        PREDECESSOR.node_type: PREDECESSOR,
        target.node_type: dataclasses.replace(target, replacements=(rule,)),
    }
    assert any(message in problem.message for problem in validate_replacement_references(schemas))


def test_reference_check_rejects_impossible_paths_with_literal_choices() -> None:
    target = _dynamic_target_schema()
    rule = _rule(
        ReplacementCase.build(
            target.node_type,
            slot_variants={"policy": "channel"},
            inputs={"policy.missing": MappingSource.copy("picture")},
        )
    )
    schemas = {
        PREDECESSOR.node_type: PREDECESSOR,
        target.node_type: dataclasses.replace(target, replacements=(rule,)),
    }
    problems = validate_replacement_references(schemas)
    assert any(
        problem.ref == "policy.missing" and "not a materializable input path" in problem.message
        for problem in problems
    )

    mixed = _rule(
        ReplacementCase.build(
            target.node_type,
            slot_variants={
                "policy": "missing",
                "source": "number",
            },
        )
    )
    schemas[target.node_type] = dataclasses.replace(target, replacements=(mixed,))
    problems = validate_replacement_references(schemas)
    assert any(
        problem.ref == "policy" and "names unknown choice" in problem.message
        for problem in problems
    )

    inactive_nested = _rule(
        ReplacementCase.build(
            target.node_type,
            slot_variants={
                "policy": "channel",
                "policy.color_source": "integer",
                "source": "number",
            },
        )
    )
    schemas[target.node_type] = dataclasses.replace(target, replacements=(inactive_nested,))
    problems = validate_replacement_references(schemas)
    assert any(
        problem.ref == "policy.color_source" and "inactive construct" in problem.message
        for problem in problems
    )


def test_reference_check_rejects_inputs_from_an_inactive_literal_branch() -> None:
    def inner_combo(key: str) -> DynamicComboSpec:
        return DynamicComboSpec(
            "inner",
            options=(DynamicComboOption(key, inputs=(InputSpec(f"{key}_only", INT),)),),
        )

    target = NodeSchema(
        node_type="new.correlated",
        combos=(
            DynamicComboSpec(
                "outer",
                options=(
                    DynamicComboOption("a", inputs=(InputSpec("a_only", INT), inner_combo("x"))),
                    DynamicComboOption("b", inputs=(InputSpec("b_only", INT), inner_combo("y"))),
                ),
            ),
        ),
    )
    rule = _rule(
        ReplacementCase.build(
            target.node_type,
            slot_variants={
                "outer": "b",
                "outer.inner": "y",
            },
            inputs={
                "outer.a_only": MappingSource.copy("picture"),
                "outer.inner.y_only": MappingSource.copy("picture"),
            },
        )
    )
    schemas = {
        PREDECESSOR.node_type: PREDECESSOR,
        target.node_type: dataclasses.replace(target, replacements=(rule,)),
    }
    problems = validate_replacement_references(schemas)
    assert any(
        problem.ref == "outer.a_only"
        and "not an input id of the selected interface" in problem.message
        for problem in problems
    )


def test_reference_check_rejects_nested_family_paths() -> None:
    nested = NodeSchema(
        node_type="new.nested-family",
        input_families=(
            InputFamilySpec(
                "items",
                (
                    DynamicComboSpec(
                        "mode",
                        options=(DynamicComboOption("value", inputs=(InputSpec("value", INT),)),),
                    ),
                ),
                member_names=("first",),
            ),
        ),
    )
    rule = _rule(
        ReplacementCase.build(
            nested.node_type,
            slot_variants={"items.first.mode": "value"},
            inputs={"items.first.mode.value": MappingSource.copy("picture")},
        )
    )
    schemas = {
        PREDECESSOR.node_type: PREDECESSOR,
        nested.node_type: dataclasses.replace(nested, replacements=(rule,)),
    }
    problems = validate_replacement_references(schemas)
    assert any(
        problem.ref == "items.first.mode" and "not a dynamic construct path" in problem.message
        for problem in problems
    )
    assert any(
        problem.ref == "items.first.mode.value"
        and "not a materializable input path" in problem.message
        for problem in problems
    )


def test_reference_check_elaborates_dynamic_helper_paths() -> None:
    helper = _dynamic_target_schema()
    case = ReplacementCase.build(
        "new.node",
        nodes={
            "helper": ReplacementNode.build(
                helper.node_type,
                values={"policy.tolerance": 0},
            )
        },
        slot_variants={
            "helper:policy": "tolerance_color",
            "helper:policy.color_source": "integer",
            "helper:source": "number",
        },
        inputs={
            "helper:policy.color_source.color_value": MappingSource.from_value("picture"),
            "helper:source": MappingSource.copy("picture"),
            "helper:source.scale": MappingSource.constant(1),
        },
        links=(ReplacementLink("out", "helper:policy.metric"),),
    )
    rule = _rule(case)
    schemas = {
        "old.node": PREDECESSOR,
        "new.node": _successor_with(rule),
        helper.node_type: helper,
    }
    assert validate_replacement_references(schemas) == ()

    primary = dataclasses.replace(helper, node_type="new.primary")
    helper_only = _rule(
        ReplacementCase.build(
            primary.node_type,
            nodes={"helper": ReplacementNode.build(helper.node_type)},
            slot_variants={
                "helper:policy": "channel",
                "helper:source": "number",
            },
        )
    )
    schemas = {
        "old.node": PREDECESSOR,
        primary.node_type: dataclasses.replace(primary, replacements=(helper_only,)),
        helper.node_type: helper,
    }
    problems = validate_replacement_references(schemas)
    assert len(problems) == 1
    assert problems[0].target == primary.node_type
    assert "required dynamic construct 'policy' has no stored choice" in problems[0].message


def test_dynamic_slot_variants_coexist_with_family_mappings() -> None:
    source = dataclasses.replace(
        PREDECESSOR,
        input_families=(
            InputFamilySpec(
                "operands",
                (InputSpec("item", INT),),
                min_members=1,
                member_names=("a", "b"),
            ),
        ),
        output_families=(OutputFamilySpec("results", INT, max_members=2),),
    )
    target = dataclasses.replace(
        _dynamic_target_schema(),
        input_families=(
            InputFamilySpec(
                "values",
                (InputSpec("value", INT),),
                min_members=1,
                member_names=("a", "b"),
            ),
        ),
        output_families=(OutputFamilySpec("values", INT, max_members=2),),
    )
    rule = _rule(
        ReplacementCase.build(
            target.node_type,
            slot_variants={
                "policy": "channel",
                "source": "number",
            },
            inputs={
                "policy.channel": MappingSource.from_value("picture"),
                "source": MappingSource.copy("picture"),
                "source.scale": MappingSource.constant(1),
            },
            input_families={
                "values": InputFamilyMapping.copy(
                    "operands", inputs={"value": MappingSource.copy("item")}
                )
            },
            output_families={"values": OutputFamilyMapping.copy("results")},
        )
    )
    schemas = {
        source.node_type: source,
        target.node_type: dataclasses.replace(target, replacements=(rule,)),
    }
    assert validate_replacement_references(schemas) == ()
    del schemas[source.node_type]
    assert validate_replacement_references(schemas) == ()


def test_literal_slot_variants_include_copied_family_evidence() -> None:
    source = NodeSchema(
        node_type="old.family-choice",
        inputs=(InputSpec("mode", INT),),
        input_families=(
            InputFamilySpec(
                "operands",
                (InputSpec("item", INT),),
                min_members=1,
                member_names=("first",),
            ),
        ),
    )
    target = NodeSchema(
        node_type="new.family-choice",
        combos=(
            DynamicComboSpec(
                "policy",
                options=(DynamicComboOption("a"), DynamicComboOption("b")),
            ),
        ),
        input_families=(
            InputFamilySpec(
                "values",
                (InputSpec("value", INT),),
                min_members=1,
                member_names=("first",),
            ),
        ),
    )
    rule = ReplacementRule(
        from_type=source.node_type,
        cases=(
            ReplacementCase.build(
                target.node_type,
                slot_variants={"policy": "a"},
                input_families={
                    "values": InputFamilyMapping.copy(
                        "operands", inputs={"value": MappingSource.copy("item")}
                    )
                },
            ),
        ),
    )
    schemas = {
        source.node_type: source,
        target.node_type: dataclasses.replace(target, replacements=(rule,)),
    }
    assert validate_replacement_references(schemas) == ()


def test_reference_check_flags_unknown_static_ids() -> None:
    rule = ReplacementRule(
        from_type="old.node",
        cases=(
            ReplacementCase.build(
                "new.node",
                when=ReplacementPredicate.value_present("missing_in"),
                inputs={"missing_target": MappingSource.copy("also_missing")},
                outputs={"missing_out": "nope"},
            ),
            ReplacementCase.build("new.node"),
        ),
    )
    schemas = {"old.node": PREDECESSOR, "new.node": _successor_with(rule)}
    problems = validate_replacement_references(schemas)
    joined = "\n".join(p.message for p in problems)
    assert "'missing_in' is not a static input id of old.node" in joined
    assert "'also_missing' is not a static input id of old.node" in joined
    assert "'nope' is not a static output id of old.node" in joined
    assert "'missing_target' is not a static input id of new.node" in joined
    assert "'missing_out' is not a static output id of new.node" in joined
    # Machine anchors: every problem names its carrier, rule, case, and the
    # schema the ref was checked against - source-side refs check against
    # the predecessor, target-side refs against the case's `to`.
    assert all(p.carrier == "new.node" and p.from_type == "old.node" for p in problems)
    assert all(p.case_index == 0 for p in problems)
    by_ref = {p.ref: p for p in problems}
    assert by_ref["missing_in"].target == "old.node"
    assert by_ref["missing_in"].ref_kind == "input"
    assert by_ref["nope"].target == "old.node"
    assert by_ref["nope"].ref_kind == "output"
    assert by_ref["missing_target"].target == "new.node"
    assert by_ref["missing_out"].ref_kind == "output"


@pytest.mark.parametrize(
    "ref",
    [
        "policy",
        "policy.color_source",
        "policy.channel",
        "policy.color_source.color_value",
        "policy.color_source.red",
        "source",
        "source.scale",
    ],
)
@pytest.mark.parametrize("mapping_kind", ["copy", "link", "value", "member", "predicate"])
def test_reference_check_accepts_declared_dynamic_source_paths(ref, mapping_kind) -> None:
    source = _dynamic_target_schema()
    mapping = {
        "copy": MappingSource.copy,
        "link": MappingSource.link,
        "value": MappingSource.from_value,
    }.get(mapping_kind, MappingSource.copy)(ref)
    case = ReplacementCase.build(
        "new.node",
        when=ReplacementPredicate.value_present(ref) if mapping_kind == "predicate" else None,
        inputs={"image": mapping} if mapping_kind in ("copy", "link", "value") else {},
        input_families={
            "values": InputFamilyMapping.from_members(
                InputFamilyMember.build("first", inputs={"value": mapping})
            )
        }
        if mapping_kind == "member"
        else {},
    )
    target = dataclasses.replace(
        SUCCESSOR,
        input_families=(InputFamilySpec("values", (InputSpec("value", INT),)),),
        replacements=(ReplacementRule(from_type=source.node_type, cases=(case, FALLBACK)),),
    )
    before = schema_to_wire(source)
    assert (
        validate_replacement_references({source.node_type: source, target.node_type: target}) == ()
    )
    assert schema_to_wire(source) == before
    assert schema_from_wire(before) == source


@pytest.mark.parametrize(
    "ref",
    [
        "missing",
        "policy.missing",
        "policy.color_source.missing",
        "policy.channel.missing",
        "policy.tolerance_color",
        "policy.color_source.integer",
        "source.number",
        "source.missing",
        "policy.items",
        "policy.items.first.value",
    ],
)
def test_reference_check_rejects_undeclared_dynamic_source_paths(ref) -> None:
    source = _dynamic_target_schema()
    combo = source.combos[0]
    source = dataclasses.replace(
        source,
        combos=(
            dataclasses.replace(
                combo,
                options=(
                    *combo.options,
                    DynamicComboOption(
                        "many", inputs=(InputFamilySpec("items", (InputSpec("value", INT),)),)
                    ),
                ),
            ),
        ),
    )
    target = _successor_with(
        ReplacementRule(
            from_type=source.node_type,
            cases=(ReplacementCase.build("new.node", inputs={"image": MappingSource.copy(ref)}),),
        )
    )
    problems = validate_replacement_references({source.node_type: source, target.node_type: target})
    assert len(problems) == 1
    assert problems[0].ref == ref
    assert problems[0].target == source.node_type


def test_reference_check_rejects_dynamic_family_paths_in_ordinary_inputs() -> None:
    """Family mappings have a dedicated field, not ordinary input paths."""
    for ref in ("operands", "operands.x"):
        rule = ReplacementRule(
            from_type="old.node",
            cases=(ReplacementCase.build("new.node", inputs={"image": MappingSource.copy(ref)}),),
        )
        schemas = {"old.node": PREDECESSOR, "new.node": _successor_with(rule)}
        problems = validate_replacement_references(schemas)
        assert len(problems) == 1
        assert "dynamic family input" in problems[0].message


def test_reference_check_validates_dynamic_family_mappings() -> None:
    source = dataclasses.replace(
        PREDECESSOR,
        inputs=(InputSpec("a", INT), InputSpec("b", INT)),
        input_families=(
            InputFamilySpec(
                "operands",
                (InputSpec("item", INT),),
                member_names=("a", "b", "c"),
            ),
        ),
    )
    target = dataclasses.replace(
        SUCCESSOR,
        input_families=(
            InputFamilySpec(
                "values",
                (InputSpec("value", INT),),
                member_names=("a", "b", "c"),
            ),
        ),
    )
    copy_rule = _rule(
        ReplacementCase.build(
            "new.node",
            input_families={
                "values": InputFamilyMapping.copy(
                    "operands", inputs={"value": MappingSource.copy("item")}
                )
            },
        )
    )
    schemas = {
        "old.node": source,
        "new.node": dataclasses.replace(target, replacements=(copy_rule,)),
    }
    assert validate_replacement_references(schemas) == ()

    members_rule = _rule(
        ReplacementCase.build(
            "new.node",
            input_families={
                "values": InputFamilyMapping.from_members(
                    InputFamilyMember.build("a", inputs={"value": MappingSource.copy("a")}),
                    InputFamilyMember.build("b", inputs={"value": MappingSource.copy("b")}),
                )
            },
        )
    )
    schemas["new.node"] = dataclasses.replace(target, replacements=(members_rule,))
    assert validate_replacement_references(schemas) == ()

    helper_rule = _rule(
        ReplacementCase.build(
            "new.node",
            nodes={
                "calculate": ReplacementNode.build("helper.node"),
                "source": ReplacementNode.build("producer.node"),
            },
            input_families={
                "calculate:values": InputFamilyMapping.from_members(
                    InputFamilyMember.build("a", inputs={"value": MappingSource.copy("a")}),
                    InputFamilyMember.build("b", inputs={"value": MappingSource.copy("b")}),
                )
            },
            links=(ReplacementLink("source:out", "calculate:values.b"),),
        )
    )
    schemas["helper.node"] = dataclasses.replace(
        target,
        node_type="helper.node",
        replacements=(),
    )
    schemas["producer.node"] = NodeSchema(
        node_type="producer.node",
        outputs=(OutputSpec("out", INT),),
    )
    schemas["new.node"] = dataclasses.replace(target, replacements=(helper_rule,))
    assert validate_replacement_references(schemas) == ()

    with pytest.raises(ValueError, match="invalid target input family id"):
        ReplacementCase.build(
            "new.node",
            nodes={"calculate": ReplacementNode.build("helper.node")},
            input_families={
                "calculate:bad.path": InputFamilyMapping.from_members(
                    InputFamilyMember.build("a", inputs={"value": MappingSource.copy("a")})
                )
            },
        )

    bad_rule = _rule(
        ReplacementCase.build(
            "new.node",
            input_families={
                "missing_target": InputFamilyMapping.copy(
                    "missing_source", inputs={"missing_local": MappingSource.copy("missing_local")}
                )
            },
        )
    )
    schemas["new.node"] = dataclasses.replace(target, replacements=(bad_rule,))
    joined = "\n".join(problem.message for problem in validate_replacement_references(schemas))
    assert "'missing_source' is not an input family id of old.node" in joined
    assert "'missing_target' is not an input family id of new.node" in joined

    constrained_target = dataclasses.replace(
        target,
        input_families=(
            InputFamilySpec(
                "values",
                (InputSpec("value", INT),),
                min_members=1,
                member_names=("a", "b"),
            ),
        ),
        replacements=(copy_rule,),
    )
    schemas["new.node"] = constrained_target
    joined = "\n".join(problem.message for problem in validate_replacement_references(schemas))
    assert "requires more members" in joined
    assert "cannot admit every copied source-family member count" in joined
    assert "cannot admit every copied source-family suffix" in joined


def test_reference_check_validates_output_family_mappings() -> None:
    source = dataclasses.replace(
        PREDECESSOR,
        outputs=(OutputSpec("left", INT), OutputSpec("right", INT)),
        output_families=(OutputFamilySpec("results", INT, max_members=3),),
    )
    target = dataclasses.replace(
        SUCCESSOR,
        output_families=(OutputFamilySpec("values", INT, max_members=3),),
    )
    copy_rule = _rule(
        ReplacementCase.build(
            "new.node",
            output_families={"values": OutputFamilyMapping.copy("results")},
        )
    )
    schemas = {
        "old.node": source,
        "new.node": dataclasses.replace(target, replacements=(copy_rule,)),
    }
    assert validate_replacement_references(schemas) == ()

    members_rule = _rule(
        ReplacementCase.build(
            "new.node",
            output_families={
                "values": OutputFamilyMapping.from_members(
                    OutputFamilyMember("0", "left"),
                    OutputFamilyMember("1", "right"),
                )
            },
        )
    )
    schemas["new.node"] = dataclasses.replace(target, replacements=(members_rule,))
    assert validate_replacement_references(schemas) == ()

    counted_source = dataclasses.replace(
        source,
        inputs=(*source.inputs, InputSpec("source_count", INT)),
        output_families=(
            OutputFamilySpec("results", INT, max_members=3, count=OutputCountSpec("source_count")),
        ),
    )
    counted_target = dataclasses.replace(
        target,
        inputs=(*target.inputs, InputSpec("target_count", INT)),
        output_families=(
            OutputFamilySpec("values", INT, max_members=3, count=OutputCountSpec("target_count")),
        ),
    )
    counted_copy_rule = _rule(
        ReplacementCase.build(
            "new.node",
            inputs={"target_count": MappingSource.copy("source_count")},
            output_families={"values": OutputFamilyMapping.copy("results")},
        )
    )
    schemas = {
        "old.node": counted_source,
        "new.node": dataclasses.replace(counted_target, replacements=(counted_copy_rule,)),
    }
    assert validate_replacement_references(schemas) == ()

    counted_members_rule = _rule(
        ReplacementCase.build(
            "new.node",
            inputs={"target_count": MappingSource.constant(2)},
            output_families={
                "values": OutputFamilyMapping.from_members(
                    OutputFamilyMember("0", "left"),
                    OutputFamilyMember("1", "right"),
                )
            },
        )
    )
    schemas["new.node"] = dataclasses.replace(counted_target, replacements=(counted_members_rule,))
    assert validate_replacement_references(schemas) == ()

    bad_count_rule = _rule(
        ReplacementCase.build(
            "new.node",
            inputs={"target_count": MappingSource.constant(3)},
            output_families={
                "values": OutputFamilyMapping.from_members(
                    OutputFamilyMember("0", "left"),
                    OutputFamilyMember("1", "right"),
                )
            },
        )
    )
    schemas["new.node"] = dataclasses.replace(counted_target, replacements=(bad_count_rule,))
    joined = "\n".join(problem.message for problem in validate_replacement_references(schemas))
    assert "must be the constant member count 2" in joined

    bad_copy_count_rule = _rule(
        ReplacementCase.build(
            "new.node",
            inputs={"target_count": MappingSource.copy("left")},
            output_families={"values": OutputFamilyMapping.copy("results")},
        )
    )
    schemas["new.node"] = dataclasses.replace(counted_target, replacements=(bad_copy_count_rule,))
    joined = "\n".join(problem.message for problem in validate_replacement_references(schemas))
    assert "must preserve source count input 'source_count'" in joined

    bad_rule = _rule(
        ReplacementCase.build(
            "new.node",
            output_families={
                "missing_target": OutputFamilyMapping.copy("missing_source"),
                "values": OutputFamilyMapping.from_members(
                    OutputFamilyMember("0", "missing_output")
                ),
            },
        )
    )
    schemas["new.node"] = dataclasses.replace(target, replacements=(bad_rule,))
    joined = "\n".join(problem.message for problem in validate_replacement_references(schemas))
    assert "'missing_source' is not an output family id of old.node" in joined
    assert "'missing_target' is not an output family id of new.node" in joined
    assert "'missing_output' is not a static output id of old.node" in joined

    constrained_target = dataclasses.replace(
        target,
        inputs=(*target.inputs, InputSpec("count", INT)),
        output_families=(
            OutputFamilySpec(
                "values",
                INT,
                min_members=1,
                max_members=2,
                count=OutputCountSpec("count"),
            ),
        ),
        replacements=(copy_rule,),
    )
    schemas = {"old.node": source, "new.node": constrained_target}
    joined = "\n".join(problem.message for problem in validate_replacement_references(schemas))
    assert "requires more members" in joined
    assert "cannot admit every copied source-family member count" in joined
    assert "cannot copy an unbound source family" in joined

    noncanonical = _rule(
        ReplacementCase.build(
            "new.node",
            output_families={
                "values": OutputFamilyMapping.from_members(OutputFamilyMember("1", "left"))
            },
        )
    )
    schemas["new.node"] = dataclasses.replace(constrained_target, replacements=(noncanonical,))
    joined = "\n".join(problem.message for problem in validate_replacement_references(schemas))
    assert "requires canonical zero-based explicit member suffixes" in joined


def test_reference_check_rejects_output_family_mapping_over_budget() -> None:
    members = tuple(OutputFamilyMember(str(index), f"result_{index}") for index in range(513))
    rule = _rule(
        ReplacementCase.build(
            "new.node",
            output_families={"values": OutputFamilyMapping.from_members(*members)},
        )
    )
    source = dataclasses.replace(
        PREDECESSOR,
        outputs=tuple(OutputSpec(member.output, INT) for member in members),
    )
    target = dataclasses.replace(
        SUCCESSOR,
        output_families=(OutputFamilySpec("values", INT),),
        replacements=(rule,),
    )
    problems = validate_replacement_references({"old.node": source, "new.node": target})
    assert len(problems) == 1
    assert "exceeds the 512 member budget" in problems[0].message


def test_reference_check_skips_absent_schemas() -> None:
    """Rules legitimately migrate nodes from never-installed packs: an
    unknown predecessor or target is unchecked, never an error."""
    rule = ReplacementRule(
        from_type="uninstalled.node",
        cases=(
            ReplacementCase.build(
                "also.uninstalled",
                inputs={"anything": MappingSource.copy("whatever")},
            ),
        ),
    )
    schemas = {"new.node": _successor_with(rule)}
    assert validate_replacement_references(schemas) == ()


def test_reference_check_rejects_duplicate_connection_consumers_without_source_schema() -> None:
    rule = ReplacementRule(
        from_type="uninstalled.node",
        cases=(
            ReplacementCase.build(
                "new.node",
                inputs={
                    "image": MappingSource.copy("same"),
                    "mode": MappingSource.link("same"),
                },
            ),
        ),
    )

    problems = validate_replacement_references({"new.node": _successor_with(rule)})
    assert len(problems) == 1
    assert problems[0].ref == "same"
    assert "consumed by multiple target mappings" in problems[0].message


def test_validate_replacement_references_resolves_helper_addresses() -> None:
    helper = NodeSchema(
        node_type="helper.node",
        inputs=(InputSpec("item", INT), InputSpec("count", INT)),
        outputs=(OutputSpec("items", INT),),
        input_families=(InputFamilySpec("dynamic", INT),),
    )
    case = ReplacementCase.build(
        "new.node",
        nodes={
            "h": ReplacementNode.build("helper.node", values={"count": 2, "missing_value": 3}),
            "unknown": ReplacementNode.build("not.installed", values={"anything": 1}),
        },
        inputs={
            "h:item": MappingSource.copy("picture"),
            "h:missing_input": MappingSource.constant(1),
            "unknown:anything": MappingSource.constant(1),
        },
        links=(ReplacementLink("h:missing_output", "image"),),
        outputs={"h:items": "result", "missing_primary": "other"},
    )
    rule = _rule(case)
    predecessor = dataclasses.replace(
        PREDECESSOR, outputs=(OutputSpec("result", INT), OutputSpec("other", INT))
    )
    schemas = {
        "old.node": predecessor,
        "new.node": _successor_with(rule),
        "helper.node": helper,
    }
    problems = validate_replacement_references(schemas)
    by_ref = {problem.ref: problem for problem in problems}
    assert set(by_ref) == {
        "missing_input",
        "missing_output",
        "missing_primary",
        "missing_value",
    }
    assert by_ref["missing_input"].target == "helper.node"
    assert by_ref["missing_input"].ref_kind == "input"
    assert by_ref["missing_output"].target == "helper.node"
    assert by_ref["missing_output"].ref_kind == "output"
    assert by_ref["missing_primary"].target == "new.node"
    assert by_ref["missing_value"].target == "helper.node"
    assert "not a static input id of helper.node" in by_ref["missing_value"].message

    legacy = _rule(
        ReplacementCase.build(
            "new.node",
            inputs={"missing:legacy": MappingSource.copy("picture")},
        )
    )
    legacy_problems = validate_replacement_references(
        {
            "old.node": PREDECESSOR,
            "new.node": _successor_with(legacy),
        }
    )
    assert len(legacy_problems) == 1
    assert legacy_problems[0].ref == "missing:legacy"
    assert legacy_problems[0].target == "new.node"
