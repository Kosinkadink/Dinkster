"""Generate the replacement-rule golden fixtures (goldens/replacements/).

Everything here goes through the real encoders (schema_to_wire, which calls
rule_to_wire for carried rules) so the committed JSON is encoder-authored,
never hand-written. tests/test_goldens.py imports build_goldens() and fails
when the files on disk drift from what the current encoders produce; run
this script to refresh them after an intentional encoder change.

The fixture content is the coverage set agreed with Dinkster-Frontend: every
predicate/mapping/transform union member, deep predicate nesting,
multi-successor fan-out, deprecation + searchVisibility + replacements on
one schema, a two-hop replacement chain with per-hop transforms, and the
core.combo socket contract with the unchanged ComboWidget shape. The dynamic
fixture selects nested target constructs and maps their
materialized input paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from dinkster_schema import (
    ColorWidget,
    ComboOption,
    ComboWidget,
    Deprecation,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    MappingSource,
    MultiComboWidget,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    ReplacementCase,
    ReplacementPredicate,
    ReplacementRule,
    StringWidget,
    TypeExpr,
    ValueTransform,
    WidgetRepresentation,
    WidgetRepresentations,
    schema_to_wire,
    validate_replacement_references,
)

GOLDENS_DIR = Path(__file__).resolve().parent.parent / "goldens" / "replacements"

_INT = TypeExpr.concrete("core.int")
_FLOAT = TypeExpr.concrete("core.float")
_STRING = TypeExpr.concrete("core.string")
_COMBO = TypeExpr.concrete("core.combo")
_COMBO_LIST = TypeExpr.list_of(_COMBO)
_BOOLEAN = TypeExpr.concrete("core.boolean")
_IMAGE_LIST = TypeExpr.list_of(TypeExpr.concrete("core.string"))


def _vocabulary_schemas() -> dict[str, NodeSchema]:
    """One rule exercising every union member of the closed vocabulary."""
    predecessor = NodeSchema(
        node_type="fixture.legacy",
        inputs=(
            InputSpec("image", _STRING),
            InputSpec("strength", _FLOAT, required=False),
            InputSpec("mode", _STRING, required=False),
            InputSpec("label", _STRING, required=False),
            InputSpec("enabled", _BOOLEAN, required=False),
            InputSpec("frames", _IMAGE_LIST, required=False),
        ),
        outputs=(OutputSpec("result", _STRING), OutputSpec("mask", _STRING)),
    )
    modern = NodeSchema(
        node_type="fixture.modern",
        inputs=(
            InputSpec("picture", _STRING),
            InputSpec(
                "intensity",
                _FLOAT,
                required=False,
                widget=NumberWidget(display="slider"),
            ),
            InputSpec("quality", _STRING, required=False),
            InputSpec("caption", _STRING, required=False),
            InputSpec("frames", _IMAGE_LIST, required=False),
        ),
        outputs=(OutputSpec("output", _STRING), OutputSpec("alpha", _STRING)),
        replacements=(
            ReplacementRule(
                from_type="fixture.legacy",
                note="renamed and re-ranged in fixture pack 2.0",
                cases=(
                    # Deeply nested guard: all[ inputConnected, any[
                    # valueEquals, not(valuePresent) ] ] - every predicate
                    # kind except bare `always` (the fallback covers that).
                    ReplacementCase.build(
                        "fixture.modern",
                        when=ReplacementPredicate.all_of(
                            ReplacementPredicate.input_connected("image"),
                            ReplacementPredicate.any_of(
                                ReplacementPredicate.value_equals("mode", "fast"),
                                ReplacementPredicate.not_(
                                    ReplacementPredicate.value_present("label")
                                ),
                            ),
                        ),
                        inputs={
                            # All 4 mapping kinds; both transforms.
                            "picture": MappingSource.copy("image"),
                            "intensity": MappingSource.from_value(
                                "strength",
                                ValueTransform.scale(0.01, offset=0.5),
                            ),
                            "quality": MappingSource.from_value(
                                "mode",
                                ValueTransform.enum_rename({"fast": "draft", "slow": "final"}),
                            ),
                            "frames": MappingSource.link("frames"),
                            "caption": MappingSource.constant("migrated"),
                        },
                        outputs={"output": "result", "alpha": "mask"},
                    ),
                    # Fan-out: a guarded case may target a DIFFERENT
                    # successor than the carrying schema.
                    ReplacementCase.build(
                        "fixture.alternate",
                        when=ReplacementPredicate.value_equals("enabled", False),
                        inputs={
                            "source": MappingSource.copy("image"),
                            # scale without offset: the wire omits offset at 0.
                            "amount": MappingSource.from_value(
                                "strength", ValueTransform.scale(2.0)
                            ),
                        },
                        outputs={"out": "result"},
                    ),
                    # The required unconditional fallback.
                    ReplacementCase.build(
                        "fixture.modern",
                        inputs={"picture": MappingSource.copy("image")},
                        outputs={"output": "result"},
                    ),
                ),
            ),
        ),
    )
    alternate = NodeSchema(
        node_type="fixture.alternate",
        inputs=(
            InputSpec("source", _STRING),
            InputSpec("amount", _FLOAT, required=False),
        ),
        outputs=(OutputSpec("out", _STRING),),
    )
    return {schema.node_type: schema for schema in (predecessor, modern, alternate)}


def _chain_schemas() -> dict[str, NodeSchema]:
    """A->B on B's schema, B->C on C's schema, per-hop transforms, with
    deprecation pointers and searchVisibility riding the same wire - B is
    the requested deprecation + searchVisibility + replacements combo."""
    chain_a = NodeSchema(
        node_type="fixture.chain-a",
        inputs=(InputSpec("level", _INT), InputSpec("mode", _STRING, required=False)),
        outputs=(OutputSpec("value", _INT),),
        deprecation=Deprecation(
            message="chain-a is superseded; use chain-b",
            since="1.2.0",
            replacement="fixture.chain-b",
        ),
        search_visibility="deprecated",
    )
    chain_b = NodeSchema(
        node_type="fixture.chain-b",
        inputs=(
            InputSpec("amount", _INT),
            InputSpec("preset", _STRING, required=False),
        ),
        outputs=(OutputSpec("value", _INT),),
        deprecation=Deprecation(
            message="chain-b was an interim shape; use chain-c",
            since="2.0.0",
            replacement="fixture.chain-c",
        ),
        search_visibility="hidden",
        replacements=(
            ReplacementRule(
                from_type="fixture.chain-a",
                note="hop 1: level scaled x10 into amount",
                cases=(
                    ReplacementCase.build(
                        "fixture.chain-b",
                        inputs={
                            "amount": MappingSource.from_value("level", ValueTransform.scale(10.0)),
                            "preset": MappingSource.copy("mode"),
                        },
                        outputs={"value": "value"},
                    ),
                ),
            ),
        ),
    )
    chain_c = NodeSchema(
        node_type="fixture.chain-c",
        inputs=(
            InputSpec("amount", _INT),
            InputSpec("profile", _STRING, required=False),
        ),
        outputs=(OutputSpec("value", _INT),),
        replacements=(
            ReplacementRule(
                from_type="fixture.chain-b",
                note="hop 2: preset vocabulary renamed into profile",
                cases=(
                    ReplacementCase.build(
                        "fixture.chain-c",
                        inputs={
                            "amount": MappingSource.copy("amount"),
                            "profile": MappingSource.from_value(
                                "preset",
                                ValueTransform.enum_rename({"lo": "low", "hi": "high"}),
                            ),
                        },
                        outputs={"value": "value"},
                    ),
                ),
            ),
        ),
    )
    return {schema.node_type: schema for schema in (chain_a, chain_b, chain_c)}


def _combo_schemas() -> dict[str, NodeSchema]:
    """The frontend announcement fixture for current widget facts."""
    combo = NodeSchema(
        node_type="fixture.combo-contract",
        inputs=(
            InputSpec(
                "choice",
                _COMBO,
                default="alpha",
                widget=ComboWidget(
                    options=(
                        ComboOption(
                            value="alpha",
                            label="Alpha",
                            info="Primary choice",
                            folder="Featured",
                        ),
                        "beta",
                    ),
                    remote_route="/api/choices/fixture.combo",
                    refresh_button=True,
                    control_after_generate="randomize",
                    control_after_refresh="last",
                    remote_timeout_ms=4096,
                    remote_max_retries=2,
                    remote_refresh_ms=0,
                ),
            ),
            InputSpec(
                "providers",
                _COMBO_LIST,
                required=False,
                default=["beta", "alpha", "beta"],
                widget=MultiComboWidget(
                    options=(
                        ComboOption(
                            value="beta",
                            label="Beta provider",
                            info="Preferred provider",
                            folder="Providers/Featured",
                        ),
                        "alpha",
                        "beta",
                    ),
                    remote_route="/api/choices/fixture.providers",
                    refresh_button=True,
                    control_after_refresh="last",
                    remote_timeout_ms=4096,
                    remote_max_retries=2,
                    remote_refresh_ms=0,
                    placeholder="Select providers",
                    chip=False,
                ),
            ),
            InputSpec(
                "prompt",
                _STRING,
                widget=WidgetRepresentations(
                    representations=(
                        WidgetRepresentation(
                            "single-line",
                            StringWidget(
                                multiline=False,
                                placeholder="Describe an image",
                                dynamic_prompts=False,
                            ),
                            display_name="Single line",
                        ),
                        WidgetRepresentation(
                            "multiline",
                            StringWidget(
                                multiline=True,
                                placeholder="Describe an image",
                                dynamic_prompts=True,
                            ),
                            display_name="Multiline",
                        ),
                    ),
                    default="multiline",
                    user_switchable=True,
                ),
            ),
            InputSpec("scale", _FLOAT, widget=NumberWidget(round=0.001)),
            InputSpec("color", _STRING, widget=ColorWidget()),
        ),
        outputs=(OutputSpec("choice", _COMBO), OutputSpec("providers", _COMBO_LIST)),
    )
    return {combo.node_type: combo}


def _dynamic_schemas() -> dict[str, NodeSchema]:
    def color_source() -> DynamicComboSpec:
        return DynamicComboSpec(
            "color_source",
            options=(
                DynamicComboOption("hex", inputs=(InputSpec("color", _STRING),)),
                DynamicComboOption("integer", inputs=(InputSpec("color_value", _INT),)),
                DynamicComboOption(
                    "channels",
                    inputs=(
                        InputSpec("red", _INT),
                        InputSpec("green", _INT),
                        InputSpec("blue", _INT),
                    ),
                ),
            ),
        )

    predecessor = NodeSchema(
        node_type="fixture.dynamic-legacy",
        inputs=(
            InputSpec("color_value", _INT),
            InputSpec("tolerance", _FLOAT),
            InputSpec("metric", _STRING),
        ),
    )
    target = NodeSchema(
        node_type="fixture.dynamic-target",
        combos=(
            DynamicComboSpec(
                "policy",
                options=(
                    DynamicComboOption("channel", inputs=(InputSpec("channel", _STRING),)),
                    DynamicComboOption(
                        "exact_color",
                        inputs=(color_source(),),
                    ),
                    DynamicComboOption(
                        "tolerance_color",
                        inputs=(
                            color_source(),
                            InputSpec("tolerance", _FLOAT),
                            InputSpec("metric", _STRING),
                        ),
                    ),
                ),
            ),
        ),
        replacements=(
            ReplacementRule(
                from_type=predecessor.node_type,
                cases=(
                    ReplacementCase.build(
                        "fixture.dynamic-target",
                        slot_variants={
                            "policy": "tolerance_color",
                            "policy.color_source": "integer",
                        },
                        inputs={
                            "policy.color_source.color_value": MappingSource.copy("color_value"),
                            "policy.tolerance": MappingSource.copy("tolerance"),
                            "policy.metric": MappingSource.copy("metric"),
                        },
                    ),
                ),
            ),
        ),
    )
    return {schema.node_type: schema for schema in (predecessor, target)}


def build_goldens() -> dict[str, dict[str, object]]:
    """Filename -> JSON content, everything through the real encoders."""
    goldens: dict[str, dict[str, object]] = {}
    fixtures = [
        ("vocabulary.json", _vocabulary_schemas()),
        ("chain.json", _chain_schemas()),
        ("combo.json", _combo_schemas()),
        ("dynamic.json", _dynamic_schemas()),
    ]
    for name, schemas in fixtures:
        problems = validate_replacement_references(schemas)
        if problems:
            raise AssertionError(
                f"{name}: fixture schemas must validate clean, got {[p.message for p in problems]}"
            )
        goldens[name] = {
            "schemas": [
                schema_to_wire(schema, replacement_schemas=schemas) for schema in schemas.values()
            ]
        }
    return goldens


def main() -> None:
    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in build_goldens().items():
        path = GOLDENS_DIR / name
        path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
