"""Change one declared field at a time and report which of a node's two signatures moves.

Usage: python tools/probe_declaration_signatures.py

A node has two signatures. The declaration signature (``schema_signature`` of
the declared schema) is stamped into saved workflows. The cache key's first
part is the signature of the flattened node, ``schema_signature`` of
``elaborate(schema, stored)`` for one stored instance.

Each case builds the base declaration below and a copy that differs in one
field, and compares the schema wire, both signatures and the flattened
interface (the ids and types of the flattened inputs and outputs). The
expected result comes from the field's class in the node plan's tables
("The contract: every row kind and every field"):

- structure: the declaration signature moves; the cache key moves exactly
  when the flattened interface of the instance changes;
- execution: both move;
- layout: neither moves.

Every case must also change the wire; a case that does not is a field the
wire drops. Exit status 1 when any case differs from its class.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeVar

from dinkster_schema.body import (
    BodyRow,
    Button,
    Choice,
    Count,
    Described,
    Formula,
    Group,
    Lookup,
    Named,
    Note,
    Numbered,
    Option,
    Reset,
    Row,
    Rule,
    Section,
    State,
    TypeChoice,
    Use,
    Variant,
    lookup,
)
from dinkster_schema.elaborate import elaborate
from dinkster_schema.model import (
    AssetWidget,
    ColorWidget,
    ComboWidget,
    CurveWidget,
    Input,
    NodeSchema,
    NumberWidget,
    Output,
    OutputKnownValue,
    StringWidget,
    TypeExpr,
)
from dinkster_schema.wire import schema_signature, schema_to_wire
from dinkster_values import (
    ASSET_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    CURVE_TYPE,
    HISTOGRAM_TYPE,
)

INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)
ASSET = TypeExpr.concrete(ASSET_TYPE)
COMBO = TypeExpr.concrete(CORE_COMBO)
CURVE = TypeExpr.concrete(CURVE_TYPE)
HISTOGRAM = TypeExpr.concrete(HISTOGRAM_TYPE)

type Class = Literal["structure", "execution", "layout"]
type Over = Mapping[str, Mapping[str, object]]

T = TypeVar("T")


@lookup
def probe_seed(prompt: str) -> int:
    """A lookup for the automatic seed case; never called."""
    return len(prompt)


@lookup
def probe_limit(steps: int) -> int | None:
    return steps * 2


@lookup
def probe_samplers() -> list[str]:
    return ["euler", "ddim"]


def declaration(over: Over) -> NodeSchema:
    """The base declaration, with ``over[name]`` replacing fields of the row called ``name``."""

    def make(cls: Callable[..., T], name: str, **fields: object) -> T:
        return cls(**{**fields, **over.get(name, {})})

    member = (
        make(Input, "member input", id="strength", type=FLOAT, default=1.0),
        make(Output, "member output", id="lora_out", type=FLOAT),
    )
    choice = make(
        Choice,
        "choice",
        id="format",
        options=(
            make(Option, "option mp4", key="mp4", rows=(make(Use, "use", id="crf"),)),
            Option("webm", rows=(Use("crf"), Input("deadline", STRING, default="good"))),
        ),
        shared=(Input("crf", INT, default=23),),
        default="mp4",
    )
    rows: tuple[BodyRow, ...] = (
        make(Input, "input", id="seed", type=INT, default=1, widget=NumberWidget(min=0)),
        Input("prompt", STRING, default="", widget=make(StringWidget, "string widget")),
        Input("tint", STRING, default="#ffffff", widget=make(ColorWidget, "color widget")),
        Input("levels", CURVE, widget=make(CurveWidget, "curve widget")),
        Input("bins", HISTOGRAM, required=False),
        Input("floor", INT, required=False),
        Input(
            "model",
            COMBO,
            widget=make(AssetWidget, "asset widget", kind="model/vae"),
        ),
        Input(
            "sampler",
            COMBO,
            default="euler",
            widget=make(ComboWidget, "combo widget", options=("euler", "ddim")),
        ),
        make(Output, "output", id="out", type=INT),
        make(State, "state", id="camera", type=STRING, default=""),
        make(
            Section,
            "section",
            id="advanced",
            rows=(Input("steps", INT, default=20),),
            label="Advanced",
        ),
        Row(rows=(Input("width", INT, default=512), Input("height", INT, default=512))),
        make(Button, "button", id="clear", label="Clear", steps=(Reset("prompt"),)),
        make(Note, "note", id="info", text="Hello"),
        make(
            Group,
            "group",
            id="loras",
            member=member,
            keys=Numbered(""),
            grows="button",
            max_members=8,
        ),
        make(
            Group,
            "values group",
            id="values",
            value=Input("v", TypeExpr.variable("T"), required=False),
            keys=Named(("a", "b", "c")),
            grows="link",
        ),
        choice,
        make(
            TypeChoice,
            "type choice",
            id="pixels",
            variants=(
                make(
                    Variant,
                    "variant",
                    key="int",
                    type=INT,
                    rows=(Input("scale", INT, default=1),),
                ),
                Variant("float", FLOAT),
            ),
        ),
        make(
            Described,
            "described",
            id="exposed",
            kinds=(Input("number", INT), Output("result", INT)),
            max_entries=8,
        ),
    )
    if "body" in over:
        rows = over["body"]["rows"](rows)  # type: ignore[operator]
    return make(NodeSchema, "node", node_type="probe.node", body=rows)


STORED: dict[str, object] = {
    "seed": 1,
    "prompt": "a cat",
    "sampler": "euler",
    "camera": "",
    "steps": 20,
    "width": 512,
    "height": 512,
    "loras.1.strength": 0.5,
    "loras.2.strength": 0.7,
    "values.a": 3,
    "format.crf": 20,
    "pixels.scale": 2,
    "exposed": json.dumps({"entries": [{"id": "n", "label": "N", "kind": "number"}]}),
    "exposed.n": 4,
}
CHOICES = {"pixels": "int", "format": "mp4"}


def flattened(
    schema: NodeSchema,
    stored: Mapping[str, object] | None = None,
    choices: Mapping[str, str] = CHOICES,
) -> NodeSchema:
    return elaborate(schema, {**STORED, **(stored or {})}, {"loras": ("1", "2")}, choices)


def interface(schema: NodeSchema) -> tuple[object, ...]:
    return tuple((row.id, str(row.type)) for row in (*schema.inputs, *schema.outputs))


@dataclass(frozen=True)
class Case:
    kind: str
    field: str
    over: Over
    expected: Class
    stored: Mapping[str, object] | None = None
    choices: Mapping[str, str] = dataclasses.field(default_factory=lambda: CHOICES)


def move_into_section(rows: tuple[BodyRow, ...]) -> tuple[BodyRow, ...]:
    """The first row moved into the section."""
    seed, rest = rows[0], rows[1:]
    return tuple(
        Section(row.id, rows=(*row.rows, seed), label=row.label)
        if isinstance(row, Section)
        else row
        for row in rest
    )


CASES: tuple[Case, ...] = (
    # Input
    Case("Input", "id", {"input": {"id": "seed2"}}, "structure"),
    Case("Input", "type", {"input": {"type": FLOAT}}, "execution"),
    Case("Input", "required", {"input": {"required": False}}, "execution"),
    Case("Input", "default", {"input": {"default": 7}}, "execution"),
    Case("Input", "on_absent", {"input": {"on_absent": "fail"}}, "execution"),
    Case("Input", "on_absent written as its default", {"input": {"on_absent": "skip"}}, "layout"),
    Case("Input", "lazy", {"input": {"lazy": True}}, "execution"),
    Case("Input", "accepts_list", {"input": {"accepts_list": True}}, "execution"),
    Case("Input", "accepts_storage", {"input": {"accepts_storage": True}}, "execution"),
    Case("Input", "accepts_stream", {"input": {"accepts_stream": True}}, "execution"),
    Case("Input", "alpha_policy", {"input": {"alpha_policy": "drop"}}, "execution"),
    Case("Input", "mask_polarity", {"input": {"mask_polarity": "coverage"}}, "execution"),
    Case("Input", "mask_semantic", {"input": {"mask_semantic": "alpha"}}, "execution"),
    Case("Input", "label", {"input": {"label": "Seed"}}, "layout"),
    Case("Input", "tooltip", {"input": {"tooltip": "The seed"}}, "layout"),
    Case("Input", "widget", {"input": {"widget": NumberWidget(min=1)}}, "layout"),
    Case("Input", "pin", {"input": {"pin": False}}, "layout"),
    Case("Input", "pin_first", {"input": {"pin_first": True}}, "layout"),
    Case("Input", "hidden", {"input": {"hidden": True}}, "layout"),
    Case("Input", "on_edit", {"input": {"on_edit": (Reset("prompt"),)}}, "layout"),
    Case("Input", "after_run", {"input": {"after_run": "randomize"}}, "layout"),
    Case("Input", "visible", {"input": {"visible": Formula("seed > 0")}}, "layout"),
    Case(
        "Input",
        "automatic (a formula)",
        {"input": {"automatic": Formula("len(prompt)")}},
        "execution",
    ),
    Case(
        "Input",
        "automatic (a lookup)",
        {"input": {"automatic": Lookup(probe_seed, args=("prompt",))}},
        "layout",
    ),
    Case("Input", "sent", {"input": {"sent": False, "pin": False}}, "structure"),
    # Widgets
    Case("NumberWidget", "max", {"input": {"widget": NumberWidget(min=0, max=9)}}, "layout"),
    Case("NumberWidget", "step", {"input": {"widget": NumberWidget(min=0, step=2)}}, "layout"),
    Case(
        "NumberWidget",
        "offset",
        {"input": {"widget": NumberWidget(min=0, step=2, offset=1)}},
        "layout",
    ),
    Case(
        "NumberWidget",
        "special",
        {"input": {"widget": NumberWidget(min=0, special={-1: "random"})}},
        "layout",
    ),
    Case(
        "NumberWidget",
        "reason",
        {"input": {"widget": NumberWidget(min=0, reason="Up to 8")}},
        "layout",
    ),
    Case(
        "NumberWidget",
        "max as a lookup",
        {"input": {"widget": NumberWidget(min=0, max=Lookup(probe_limit, args=("steps",)))}},
        "layout",
    ),
    Case(
        "NumberWidget",
        "min, step and offset as formulas",
        {
            "input": {
                "widget": NumberWidget(
                    min=Formula("width // 2"), step=Formula("steps"), offset=Formula("height")
                )
            }
        },
        "layout",
    ),
    Case(
        "NumberWidget",
        "offset as a lookup",
        {"input": {"widget": NumberWidget(step=2, offset=Lookup(probe_limit, args=("steps",)))}},
        "layout",
    ),
    Case(
        "CurveWidget",
        "input_strip",
        {"curve widget": {"input_strip": ("#000000", "#ffffff")}},
        "layout",
    ),
    Case(
        "CurveWidget",
        "output_strip",
        {"curve widget": {"output_strip": ("#000000", "#ff0000")}},
        "layout",
    ),
    Case("CurveWidget", "histogram", {"curve widget": {"histogram": "bins"}}, "layout"),
    Case(
        "StringWidget", "dynamic_prompts", {"string widget": {"dynamic_prompts": True}}, "execution"
    ),
    Case("StringWidget", "multiline", {"string widget": {"multiline": False}}, "layout"),
    Case("StringWidget", "placeholder", {"string widget": {"placeholder": "Type"}}, "layout"),
    Case(
        "StringWidget",
        "completions as a formula",
        {"string widget": {"completions": Formula('["cat", "dog"]')}},
        "layout",
    ),
    Case("ColorWidget", "alpha", {"color widget": {"alpha": True}}, "execution"),
    Case("AssetWidget", "kind", {"asset widget": {"kind": "model/diffusers"}}, "execution"),
    Case("AssetWidget", "selection", {"asset widget": {"selection": True}}, "execution"),
    Case("AssetWidget", "accept", {"asset widget": {"accept": (".safetensors",)}}, "layout"),
    Case(
        "ComboWidget", "options", {"combo widget": {"options": ("euler", "ddim", "lms")}}, "layout"
    ),
    Case(
        "ComboWidget",
        "options as a lookup",
        {"combo widget": {"options": Lookup(probe_samplers)}},
        "layout",
    ),
    # Output
    Case("Output", "id", {"output": {"id": "out2"}}, "structure"),
    Case("Output", "type", {"output": {"type": FLOAT}}, "execution"),
    Case("Output", "optional", {"output": {"optional": True}}, "execution"),
    Case("Output", "alpha_policy", {"output": {"alpha_policy": "drop"}}, "execution"),
    Case("Output", "mask_polarity", {"output": {"mask_polarity": "coverage"}}, "execution"),
    Case("Output", "mask_semantic", {"output": {"mask_semantic": "alpha"}}, "execution"),
    Case("Output", "preview", {"output": {"preview": True}}, "layout"),
    Case("Output", "label", {"output": {"label": "Out"}}, "layout"),
    Case("Output", "tooltip", {"output": {"tooltip": "The result"}}, "layout"),
    Case(
        "OutputKnownValue",
        "limits",
        {"output": {"known_value": OutputKnownValue("seed", limits={"min": "floor"})}},
        "layout",
    ),
    # State
    Case("State", "id", {"state": {"id": "camera2"}}, "structure"),
    Case("State", "type", {"state": {"type": INT, "default": 0}}, "execution"),
    Case("State", "default", {"state": {"default": "front"}}, "execution"),
    Case("State", "part_of_run", {"state": {"part_of_run": False}}, "structure"),
    # Section
    Case("Section", "id", {"section": {"id": "more"}}, "layout"),
    Case("Section", "label", {"section": {"label": "More"}}, "layout"),
    Case("Section", "tooltip", {"section": {"tooltip": "More settings"}}, "layout"),
    Case("Section", "visible", {"section": {"visible": Formula("seed > 0")}}, "layout"),
    Case("Section", "collapsed", {"section": {"collapsed": True}}, "layout"),
    Case("Section", "open_name", {"section": {"open_name": False}}, "layout"),
    Case("Section", "rows (a row moved in)", {"body": {"rows": move_into_section}}, "layout"),
    # Row order in the body
    Case(
        "Body",
        "row order",
        {"body": {"rows": lambda rows: (*rows[1:], rows[0])}},
        "layout",
    ),
    # Button and Note
    Case("Button", "id", {"button": {"id": "wipe"}}, "layout"),
    Case("Button", "label", {"button": {"label": "Wipe"}}, "layout"),
    Case("Button", "enabled", {"button": {"enabled": Formula("seed > 0")}}, "layout"),
    Case("Button", "disabled_reason", {"button": {"disabled_reason": "No seed"}}, "layout"),
    Case("Button", "icon", {"button": {"icon": "x"}}, "layout"),
    Case("Button", "steps", {"button": {"steps": (Reset("steps"),)}}, "layout"),
    Case("Note", "id", {"note": {"id": "hint"}}, "layout"),
    Case("Note", "text", {"note": {"text": Formula('"seed " + str(seed)')}}, "layout"),
    Case("Note", "unknown", {"note": {"unknown": "..."}}, "layout"),
    Case("Note", "visible", {"note": {"visible": False}}, "layout"),
    # Group
    Case("Group", "id", {"group": {"id": "adapters"}}, "structure"),
    Case(
        "Group",
        "member",
        {"group": {"member": (Input("strength", FLOAT, default=1.0),)}},
        "structure",
    ),
    Case("Group", "keys", {"group": {"keys": Numbered("", start=0)}}, "structure"),
    Case("Group", "min_members", {"group": {"min_members": 1}}, "structure"),
    Case("Group", "max_members", {"group": {"max_members": 9}}, "structure"),
    Case("Group", "grows (button to filled)", {"group": {"grows": "filled"}}, "layout"),
    Case(
        "Group", "grows (button to count)", {"group": {"grows": Count(Formula("3"))}}, "structure"
    ),
    Case("Group", "independent_types", {"values group": {"independent_types": True}}, "execution"),
    Case(
        "Group", "optional_after_minimum", {"group": {"optional_after_minimum": True}}, "execution"
    ),
    Case("Group", "reorderable", {"group": {"reorderable": True}}, "layout"),
    Case("Group", "add_label", {"group": {"add_label": "Add LoRA"}}, "layout"),
    Case(
        "Group",
        "member_label",
        {"group": {"member_label": Formula('"LoRA " + str(index() + 1)')}},
        "layout",
    ),
    Case("Group", "label", {"group": {"label": "LoRAs"}}, "layout"),
    Case("Group", "tooltip", {"group": {"tooltip": "Adapters"}}, "layout"),
    Case("Group", "visible", {"group": {"visible": False}}, "layout"),
    Case("Group", "open_name", {"group": {"open_name": False}}, "layout"),
    Case("Group member", "Input default", {"member input": {"default": 0.8}}, "execution"),
    Case("Group member", "Input label", {"member input": {"label": "S"}}, "layout"),
    Case("Group member", "Output type", {"member output": {"type": INT}}, "execution"),
    # Choice
    Case(
        "Choice",
        "id",
        {"choice": {"id": "container"}},
        "structure",
        choices={"pixels": "int", "container": "mp4"},
    ),
    Case(
        "Choice",
        "options (an unchosen option added)",
        {
            "choice": {
                "options": (
                    Option("mp4", rows=(Use("crf"),)),
                    Option("webm", rows=(Use("crf"), Input("deadline", STRING, default="good"))),
                    Option("gif"),
                )
            }
        },
        "structure",
    ),
    Case(
        "Choice",
        "order of options",
        {
            "choice": {
                "options": (
                    Option("webm", rows=(Use("crf"), Input("deadline", STRING, default="good"))),
                    Option("mp4", rows=(Use("crf"),)),
                )
            }
        },
        "layout",
    ),
    Case(
        "Choice",
        "shared",
        {"choice": {"shared": (Input("crf", INT, default=23), Input("fps", INT, default=24))}},
        "structure",
    ),
    Case("Choice", "leader", {"choice": {"leader": Formula('"mp4"')}}, "structure"),
    Case("Choice", "default", {"choice": {"default": "webm"}}, "layout"),
    Case("Choice", "on_edit", {"choice": {"on_edit": (Reset("seed"),)}}, "layout"),
    # An optional choice declares no default.
    Case("Choice", "required", {"choice": {"required": False, "default": None}}, "structure"),
    Case("Choice", "hidden", {"choice": {"hidden": True}}, "layout"),
    Case("Choice", "label", {"choice": {"label": "Format"}}, "layout"),
    Case("Choice", "tooltip", {"choice": {"tooltip": "Container"}}, "layout"),
    Case("Choice", "visible", {"choice": {"visible": False}}, "layout"),
    Case("Option", "label", {"option mp4": {"label": "MP4"}}, "layout"),
    Case("Use", "label", {"use": {"label": "Quality"}}, "layout"),
    Case("Use", "tooltip", {"use": {"tooltip": "Lower is better"}}, "layout"),
    Case("Use", "widget", {"use": {"widget": NumberWidget(min=0, max=51)}}, "layout"),
    # The asset kind a run resolves by is execution in a Use as in an Input.
    Case("Use", "widget", {"use": {"widget": AssetWidget(kind="model/vae")}}, "execution"),
    Case("Use", "default", {"use": {"default": 30}}, "execution"),
    Case("Use", "required", {"use": {"required": False}}, "execution"),
    # TypeChoice and Variant
    Case(
        "TypeChoice",
        "id",
        {"type choice": {"id": "pixels2"}},
        "structure",
        choices={"pixels2": "int", "format": "mp4"},
    ),
    Case(
        "TypeChoice",
        "variants (an unchosen variant added)",
        {
            "type choice": {
                "variants": (
                    Variant("int", INT, rows=(Input("scale", INT, default=1),)),
                    Variant("float", FLOAT),
                    Variant("text", STRING),
                )
            }
        },
        "structure",
    ),
    Case("TypeChoice", "required", {"type choice": {"required": False}}, "structure"),
    Case(
        "TypeChoice",
        "rows",
        {"type choice": {"rows": (Input("offset", INT, default=0),)}},
        "structure",
    ),
    Case("TypeChoice", "type_variable", {"type choice": {"type_variable": "T"}}, "structure"),
    Case("TypeChoice", "label", {"type choice": {"label": "Pixels"}}, "layout"),
    Case("TypeChoice", "tooltip", {"type choice": {"tooltip": "What flows"}}, "layout"),
    Case("TypeChoice", "visible", {"type choice": {"visible": False}}, "layout"),
    Case("Variant", "type (chosen)", {"variant": {"type": STRING}}, "structure"),
    Case(
        "Variant",
        "rows (chosen)",
        {"variant": {"rows": (Input("scale", INT, default=1), Input("bias", INT, default=0))}},
        "structure",
    ),
    Case("Variant", "tooltip", {"variant": {"tooltip": "Whole numbers"}}, "layout"),
    # Described
    Case("Described", "id", {"described": {"id": "shown"}}, "structure"),
    Case(
        "Described",
        "kinds",
        {"described": {"kinds": (Input("number", INT), Output("result", FLOAT))}},
        "structure",
    ),
    Case("Described", "min_entries", {"described": {"min_entries": 1}}, "structure"),
    Case("Described", "max_entries", {"described": {"max_entries": 9}}, "structure"),
    Case(
        "Described",
        "fixed_ids",
        {"described": {"fixed_ids": True}},
        "structure",
        stored={
            "exposed": json.dumps({"entries": [{"id": "number", "label": "N", "kind": "number"}]})
        },
    ),
    Case("Described", "label", {"described": {"label": "Exposed"}}, "layout"),
    # NodeSchema fields added with the body
    Case(
        "NodeSchema",
        "tables (unused by a structure formula)",
        {"node": {"tables": {"t": {"a": 1}}}},
        "layout",
    ),
    Case(
        "NodeSchema",
        "derived (unused by a structure formula)",
        {"node": {"derived": {"d": Formula("seed + 1")}}},
        "layout",
    ),
    Case(
        "NodeSchema",
        "validate",
        {"node": {"validate": (Rule(Formula("seed >= 0"), "The seed is negative"),)}},
        "layout",
    ),
    Case("NodeSchema", "title_suffix", {"node": {"title_suffix": Formula("str(seed)")}}, "layout"),
    Case(
        "NodeSchema",
        "menu",
        {"node": {"menu": (Button("reset_all", "Reset all", steps=(Reset("seed"),)),)}},
        "layout",
    ),
)

# Fields of the plan's tables that the declaration classes do not have yet.
NOT_BUILT = (
    "Input: order_only, read_only, disabled, disabled_reason, validate, types",
    "Output: known, types",
    "Widgets: StringWidget.tokens",
)


def run_case(case: Case, base: NodeSchema, flat: NodeSchema) -> dict[str, object]:
    result: dict[str, object] = {"kind": case.kind, "field": case.field, "class": case.expected}
    try:
        changed = declaration(case.over)
        changed_flat = flattened(changed, case.stored, case.choices)
        if case.stored is not None:
            # The stored state is part of the case: compare like with like.
            flat = flattened(base, case.stored)
    except Exception as exc:  # noqa: BLE001 - a case that cannot be built is reported
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    wire = schema_to_wire(changed) != schema_to_wire(base)
    decl = schema_signature(changed) != schema_signature(base)
    cache = schema_signature(changed_flat) != schema_signature(flat)
    shape = interface(changed_flat) != interface(flat)
    want_decl = case.expected != "layout"
    want_cache = {"execution": True, "layout": False, "structure": shape}[case.expected]
    result.update(wire=wire, declaration=decl, cache=cache, interface=shape)
    problems: list[str] = []
    if not wire:
        problems.append("the wire does not change")
    if decl != want_decl:
        problems.append(f"declaration signature {'moves' if decl else 'stays'}")
    if cache != want_cache:
        problems.append(f"cache key {'moves' if cache else 'stays'}")
    if problems:
        result["error"] = "; ".join(problems)
    return result


def main(report: Callable[[str], object] = print) -> int:
    base = declaration({})
    flat = flattened(base)
    results = [run_case(case, base, flat) for case in CASES]

    def mark(result: dict[str, object], key: str) -> str:
        return "moves" if result.get(key) else "-"

    report("| Row kind | Field | Class | Declaration | Cache key | Result |")
    report("| --- | --- | --- | --- | --- | --- |")
    for result in results:
        outcome = f"DIFFERS: {result['error']}" if "error" in result else "as its class"
        report(
            f"| {result['kind']} | {result['field']} | {result['class']} | "
            f"{mark(result, 'declaration')} | {mark(result, 'cache')} | {outcome} |"
        )
    failed = [result for result in results if "error" in result]
    report("")
    report(f"{len(results)} cases, {len(failed)} differ from their class.")
    report("Not built yet, so not probed:")
    for line in NOT_BUILT:
        report(f"- {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
