"""Replacement rules: the declarative migration vocabulary (DESIGN 3.1).

Field-for-field mirror of Dinkster-Frontend's replace/model.ts. The tests
freeze three contracts: the exact wire shapes (the frontend cross-checks
field names 1:1 against these), the closed-vocabulary validation (unknown
kinds and malformed shapes reject at construction and at decode), and the
lifecycle-metadata rule (rules ride the schema wire but never the schema
signature).
"""

import dataclasses
from typing import Any, cast

import pytest
from dinkster_schema import (
    Choice,
    Group,
    Input,
    InputFamilyMapping,
    InputFamilyMember,
    MappingSource,
    Named,
    NodeSchema,
    Numbered,
    Option,
    Output,
    OutputFamilyMapping,
    OutputFamilyMember,
    ReplacementCase,
    ReplacementLink,
    ReplacementNode,
    ReplacementPredicate,
    ReplacementRule,
    TypeChoice,
    TypeExpr,
    ValueTransform,
    Variant,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
    validate_replacement_references,
)
from dinkster_schema.replace import rule_from_wire, rule_to_wire

INT = TypeExpr.concrete("INT")


def _rule(*cases: ReplacementCase, from_type: str = "old.node") -> ReplacementRule:
    return ReplacementRule(from_type=from_type, cases=tuple(cases))


FALLBACK = ReplacementCase.build("new.node")


# -- predicate wire shapes ----------------------------------------------------


# -- transforms and mapping sources -------------------------------------------


def test_input_family_source_template_checks_an_available_predecessor() -> None:
    mapping = InputFamilyMapping.copy(
        "operands", inputs={"value": MappingSource.copy("item")}, source_template=("item",)
    )
    target = NodeSchema(
        node_type="new.node",
        body=(Group("values", member=(Input("value", INT),), keys=Numbered("")),),
        replacements=(
            _rule(ReplacementCase.build("new.node", input_families={"values": mapping})),
        ),
    )
    predecessor = NodeSchema(
        node_type="old.node",
        body=(
            Group(
                "operands",
                member=(Input("weight", INT), Input("item", INT)),
                keys=Numbered(""),
            ),
        ),
    )
    problems = validate_replacement_references(
        {target.node_type: target, predecessor.node_type: predecessor}
    )
    assert len(problems) == 1
    assert "source template ids do not match" in problems[0].message


# -- cases and rules -----------------------------------------------------------


def test_rule_wire_roundtrip_full() -> None:
    rule = ReplacementRule(
        from_type="old.sampler",
        from_signature="old-interface",
        from_version="1.2.3",
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
    assert wire["fromSignature"] == "old-interface"
    assert wire["fromVersion"] == "1.2.3"
    for field in ("fromSignature", "fromVersion"):
        for invalid in (None, "", 3, False):
            with pytest.raises(ValueError, match=field):
                rule_from_wire({**wire, field: invalid})
    # optional fields are omitted, never null
    assert "note" not in wire
    assert "when" not in wire["cases"][1]
    assert "inputs" not in wire["cases"][1]
    assert "outputs" not in wire["cases"][1]


# -- schema wire + signature ---------------------------------------------------


SUCCESSOR = NodeSchema(
    node_type="new.node",
    body=(Input("image", INT), Input("mode", INT, required=False), Output("out", INT)),
    replacements=(
        ReplacementRule(
            from_type="old.node",
            cases=(
                ReplacementCase.build("new.node", inputs={"image": MappingSource.copy("picture")}),
            ),
        ),
    ),
)


def test_replacements_never_join_the_signature() -> None:
    """Lifecycle metadata: shipping or changing migration advice must not
    invalidate the node's caches."""
    bare = dataclasses.replace(SUCCESSOR, replacements=())
    assert schema_signature(SUCCESSOR) == schema_signature(bare)


# -- cross-schema reference checks ----------------------------------------------


PREDECESSOR = NodeSchema(
    node_type="old.node",
    body=(
        Input("picture", INT),
        Input("mode", INT, required=False),
        Output("result", INT),
        Group("operands", value=Input("operand", INT), keys=Numbered("")),
    ),
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


def _policy_choice(*extra: Option) -> Choice:
    return Choice(
        "policy",
        options=(
            Option("channel", rows=(Input("channel", INT),)),
            Option(
                "tolerance_color",
                rows=(
                    Choice(
                        "color_source",
                        options=(
                            Option("integer", rows=(Input("color_value", INT),)),
                            Option(
                                "channels",
                                rows=(Input("red", INT), Input("green", INT), Input("blue", INT)),
                            ),
                        ),
                    ),
                    Input("tolerance", INT),
                    Input("metric", INT),
                ),
            ),
            *extra,
        ),
    )


def _dynamic_target_schema(*extra: Option) -> NodeSchema:
    return NodeSchema(
        node_type="new.dynamic",
        body=(
            Input("fixed", INT),
            Output("out", INT),
            _policy_choice(*extra),
            TypeChoice("source", variants=(Variant("number", INT, rows=(Input("scale", INT),)),)),
        ),
    )


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
    assert "required choice 'policy' has no stored option" in problems[0].message


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
        body=(Input("shared", INT),),
    )
    target = NodeSchema(
        node_type="new.shared-slot",
        body=(
            TypeChoice(
                "left", variants=(Variant("integer", INT, rows=(Input("left_only", INT),)),)
            ),
            TypeChoice(
                "right", variants=(Variant("integer", INT, rows=(Input("right_only", INT),)),)
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
        body=(Input("shared", INT),),
    )
    target = NodeSchema(
        node_type="new.shared-input",
        body=(
            Input("fixed", INT),
            TypeChoice("choice", variants=(Variant("integer", INT),)),
            Group("items", member=(Input("value", INT),), keys=Named(("first",))),
        ),
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
    assert "required choice 'policy' has no stored option" in problems[0].message


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
        body=(
            *SUCCESSOR.body,
            Group("values", member=(Input("value", INT),), keys=Named(("first",))),
        ),
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
    source = _dynamic_target_schema(
        Option(
            "many",
            rows=(Group("items", member=(Input("value", INT),), keys=Named(("first",))),),
        )
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


def test_reference_check_validates_dynamic_family_mappings() -> None:
    source = dataclasses.replace(
        PREDECESSOR,
        body=(
            Input("a", INT),
            Input("b", INT),
            Group("operands", member=(Input("item", INT),), keys=Named(("a", "b", "c"))),
        ),
    )
    target = dataclasses.replace(
        SUCCESSOR,
        body=(
            *SUCCESSOR.body,
            Group("values", value=Input("value", INT), keys=Named(("a", "b", "c"))),
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
        body=(Output("out", INT),),
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
        body=(
            *SUCCESSOR.body,
            Group(
                "values",
                value=Input("value", INT),
                keys=Named(("a", "b")),
                min_members=1,
                max_members=2,
            ),
        ),
        replacements=(copy_rule,),
    )
    schemas["new.node"] = constrained_target
    joined = "\n".join(problem.message for problem in validate_replacement_references(schemas))
    assert "requires more members" in joined
    assert "cannot admit every copied source-family member count" in joined
    assert "cannot admit every copied source-family suffix" in joined


def test_spread_mapping_round_trips_and_names_a_member_input_only_for_records() -> None:
    rule = _rule(
        ReplacementCase.build(
            "new.node",
            input_families={
                "values": InputFamilyMapping.spread("picture"),
                "records": InputFamilyMapping.spread("picture", member_input="item"),
            },
        )
    )
    wire = cast("Any", rule_to_wire(rule))
    assert wire["cases"][0]["inputFamilies"] == {
        "values": {"kind": "spread", "input": "picture"},
        "records": {"kind": "spread", "input": "picture", "member": "item"},
    }
    assert rule_from_wire(wire) == rule
    for invalid in (
        {"kind": "spread"},
        {"kind": "spread", "input": "picture", "member": ""},
        {"kind": "spread", "input": "picture", "sourceFamily": "operands"},
    ):
        with pytest.raises(ValueError, match="spread"):
            rule_from_wire(
                {**wire, "cases": [{"to": "new.node", "inputFamilies": {"values": invalid}}]}
            )

    target = dataclasses.replace(
        SUCCESSOR,
        body=(
            *SUCCESSOR.body,
            Group("values", value=Input("value", INT), keys=Numbered("v", start=0)),
            Group(
                "records",
                member=(Input("item", INT), Input("label", INT)),
                keys=Numbered("r"),
            ),
        ),
    )
    schemas = {
        "old.node": PREDECESSOR,
        "new.node": dataclasses.replace(target, replacements=(rule,)),
    }
    assert validate_replacement_references(schemas) == ()

    wrong = _rule(
        ReplacementCase.build(
            "new.node",
            input_families={
                "values": InputFamilyMapping.spread("missing", member_input="value"),
                "records": InputFamilyMapping.spread("picture"),
            },
        )
    )
    schemas["new.node"] = dataclasses.replace(target, replacements=(wrong,))
    messages = [problem.message for problem in validate_replacement_references(schemas)]
    assert len(messages) == 3
    joined = "\n".join(messages)
    assert "'missing'" in joined
    assert "is a group of values; a spread into it names no member input" in joined
    assert "names no template input of its members to take each item" in joined


def test_reference_check_validates_output_family_mappings() -> None:
    source = dataclasses.replace(
        PREDECESSOR,
        body=(
            Output("left", INT),
            Output("right", INT),
            Group(
                "results",
                value=Output("result", INT),
                keys=Numbered("", start=0),
                max_members=3,
            ),
        ),
    )
    target = dataclasses.replace(
        SUCCESSOR,
        body=(
            *SUCCESSOR.body,
            Group("values", value=Output("value", INT), keys=Numbered("", start=0), max_members=3),
        ),
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
        body=(
            *SUCCESSOR.body,
            Group(
                "values",
                value=Output("value", INT),
                keys=Numbered("", start=0),
                min_members=1,
                max_members=2,
            ),
        ),
        replacements=(copy_rule,),
    )
    schemas = {"old.node": source, "new.node": constrained_target}
    joined = "\n".join(problem.message for problem in validate_replacement_references(schemas))
    assert "requires more members" in joined
    assert "cannot admit every copied source-family member count" in joined


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


def test_validate_replacement_references_resolves_helper_addresses() -> None:
    helper = NodeSchema(
        node_type="helper.node",
        body=(
            Input("item", INT),
            Input("count", INT),
            Output("items", INT),
            Group("dynamic", value=Input("value", INT), keys=Numbered("")),
        ),
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
        PREDECESSOR,
        body=(*PREDECESSOR.body, Output("other", INT)),
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
