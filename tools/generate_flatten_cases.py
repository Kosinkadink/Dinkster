"""Write the flattening case file that the engine and the editor both load.

Usage: python tools/generate_flatten_cases.py [--check]

Each case is a node body on the schema wire, one node's stored state, and the
answer written out by hand below: the flattened inputs (id, type, required,
default) and outputs in order and the chosen keys, or a refusal, either when
the declaration loads or when the node is flattened. The answers are not
computed by the engine, so the engine's test and the editor's test both check
against what the node plan says. ``--check`` regenerates in memory and fails if
the committed file differs. After a change, update the hash in
``tests/test_flatten_cases.py`` and copy the file into Dinkster-Frontend
``packages/core/test/fixtures/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dinkster_schema.body import (
    MAX_DEPTH,
    BodyRow,
    Button,
    Call,
    Choice,
    Count,
    Described,
    Formula,
    Group,
    Named,
    Note,
    Numbered,
    Option,
    Randomize,
    Row,
    Section,
    SetMode,
    TypeChoice,
    Use,
    Variant,
)
from dinkster_schema.model import (
    ComboWidget,
    Input,
    NumberWidget,
    Output,
    StringWidget,
    TypeExpr,
)
from dinkster_schema.wire import SCHEMA_WIRE_VERSION, body_row_to_wire
from dinkster_values import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    CustomWidgetDescriptor,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "fixtures" / f"flatten_v{SCHEMA_WIRE_VERSION}.json"

INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)
LINK = {"link": True}

type Flat = list[object]


def row(
    row_id: str, type_id: str | None = CORE_STRING, required: bool = True, default: object = None
) -> Flat:
    """One expected flattened input: id, type id, required, default."""
    return [row_id, type_id, required, default]


def case(
    case_id: str,
    body: tuple[BodyRow, ...],
    *,
    stored: dict[str, object] | None = None,
    members: dict[str, list[str]] | None = None,
    choices: dict[str, str] | None = None,
    inputs: list[Flat] | None = None,
    outputs: list[str] | None = None,
    chose: dict[str, str] | None = None,
    refused: str | None = None,
    refused_at_load: str | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": case_id,
        "body": [body_row_to_wire(item, SCHEMA_WIRE_VERSION) for item in body],
        # A list of pairs: the stored order is the run's member order.
        "stored": [[key, value] for key, value in (stored or {}).items()],
        "members": members or {},
        "choices": choices or {},
    }
    if refused_at_load is not None:
        entry["outcome"] = "refused-at-load"
        entry["message"] = refused_at_load
    elif refused is not None:
        entry["outcome"] = "refused"
        entry["message"] = refused
    else:
        entry["outcome"] = "flattened"
        entry["inputs"] = inputs or []
        entry["outputs"] = outputs or []
        entry["chose"] = chose or {}
    return entry


def nested(depth: int) -> tuple[BodyRow, ...]:
    """Boxes ``depth`` levels deep, alternating sections and groups of records,
    with a Row of two inputs as the innermost box."""
    inner: tuple[BodyRow, ...] = (
        Row(rows=(Input("a", INT, default=0), Input("b", INT, default=0))),
    )
    for level in range(depth - 1, 0, -1):
        if level % 2:
            inner = (Section(f"s{level}", rows=inner),)
        else:
            inner = (Group(f"g{level}", member=inner, keys=Numbered("")),)
    return inner


def deep_ids(depth: int) -> list[Flat]:
    """The flattened ids of ``nested(depth)`` with member 1 of every group stored."""
    prefix = "".join(f"g{level}.1." for level in range(2, depth, 2))
    return [row(prefix + "a", CORE_INT, True, 0), row(prefix + "b", CORE_INT, True, 0)]


def deep_stored(depth: int) -> dict[str, object]:
    prefix = "".join(f"g{level}.1." for level in range(2, depth, 2))
    return {prefix + "a": 1}


SHARED_FORMAT = Choice(
    "format",
    options=(
        Option("mp4", rows=(Use("crf"),)),
        Option(
            "webm",
            rows=(
                Use("crf", default=31),
                Input("deadline", STRING, default="good"),
                Choice(
                    "audio",
                    options=(
                        Option("none"),
                        Option("opus", rows=(Input("bitrate", INT, default=96),)),
                    ),
                ),
            ),
        ),
    ),
    shared=(Input("crf", INT, default=23),),
)

LED = (
    Input("vae", INT, required=False),
    Choice(
        "mode",
        options=(Option("plain"), Option("decode", rows=(Input("tile", INT, default=64),))),
        leader=Formula('"decode" if linked(vae) else "plain"'),
    ),
)

COUNTED = (
    Input("n", INT, default=2),
    Group(
        "layers",
        member=(Input("w", INT, default=0),),
        keys=Numbered(""),
        grows=Count(Formula("n")),
        max_members=3,
    ),
)


# The count reads mode.n while option b is chosen: unknown, so min_members.
HIDDEN_OUTSIDE = (
    Choice("mode", options=(Option("a", rows=(Input("n", INT, default=1),)), Option("b"))),
    Group(
        "layers",
        value=Output("layer", INT),
        keys=Numbered(""),
        grows=Count(Formula("mode.n")),
        max_members=3,
    ),
)

HIDDEN_INSIDE = (
    Choice(
        "mode",
        options=(
            Option("a", rows=(Input("n", INT, default=1),)),
            Option(
                "b",
                rows=(
                    Group(
                        "layers",
                        value=Output("layer", INT),
                        keys=Numbered(""),
                        grows=Count(Formula("n")),
                        max_members=3,
                    ),
                ),
            ),
        ),
    ),
)


def counted_by(choice: Choice) -> tuple[BodyRow, ...]:
    """A group of outputs counted by ``mode.n``, beside ``choice``."""
    return (
        choice,
        Group(
            "layers",
            value=Output("layer", INT),
            keys=Numbered(""),
            grows=Count(Formula("mode.n")),
            max_members=4,
        ),
    )


# Nothing stored for mode.n: the count reads its default under the chosen option.
NESTED_DEFAULT = counted_by(
    Choice("mode", options=(Option("a", rows=(Input("n", INT, default=2),)), Option("b")))
)

# Nothing stored for mode.n: the count reads the default the option's Use gives.
USE_DEFAULT = counted_by(
    Choice(
        "mode",
        options=(Option("a", rows=(Use("n", default=3),)), Option("b", rows=(Use("n"),))),
        shared=(Input("n", INT, default=1),),
    )
)


# Option a holds a group; b is chosen, so its size and rows read unknown and
# each count falls back to its minimum.
GROUP_UNCHOSEN = (
    Choice(
        "c",
        options=(
            Option(
                "a", rows=(Group("g", member=(Input("s", INT, default=1),), keys=Numbered("")),)
            ),
            Option("b"),
        ),
    ),
    Group(
        "o",
        value=Output("v", INT),
        keys=Numbered(""),
        grows=Count(Formula("len(c.g)")),
        min_members=1,
        max_members=5,
    ),
    Group(
        "p",
        value=Output("v", INT),
        keys=Numbered(""),
        grows=Count(Formula("sum(c.g.s)")),
        min_members=1,
        max_members=5,
    ),
)

# The count reads mode.n while b is chosen: unknown, so min_members.
UNKNOWN_COUNT = (
    Choice("mode", options=(Option("a", rows=(Input("n", INT, default=1),)), Option("b"))),
    Group(
        "layers",
        value=Output("layer", INT),
        keys=Numbered(""),
        grows=Count(Formula("mode.n")),
        min_members=2,
        max_members=3,
    ),
)

# The leader reads mode.k while b is chosen: unknown, so the first option.
UNKNOWN_LEADER = (
    Choice("mode", options=(Option("a", rows=(Input("k", STRING, default="y"),)), Option("b"))),
    Choice(
        "led",
        options=(Option("x", rows=(Input("w", INT, default=0),)), Option("y")),
        leader=Formula("mode.k"),
    ),
)


# Member 1 stores nothing for s: sum(g.s) reads its default, 2, plus member 2's 3.
MEMBER_DEFAULT = (
    Group("g", member=(Input("s", INT, default=2), Input("t", INT, default=0)), keys=Numbered("")),
    Group(
        "o",
        value=Output("v", INT),
        keys=Numbered(""),
        grows=Count(Formula("sum(g.s)")),
        max_members=9,
    ),
)


def cases() -> list[dict[str, object]]:
    return [
        case(
            "group-of-values/ids-are-group-dot-key-in-run-order",
            (Group("images", value=Input("image", STRING), keys=Numbered("image", start=0)),),
            stored={"images.image2": "b", "images.image0": "a"},
            inputs=[row("images.image2"), row("images.image0")],
        ),
        case(
            "group-of-records/ids-are-group-dot-key-dot-row",
            (
                Group(
                    "loras",
                    member=(Input("strength", FLOAT, default=1.0), Input("name", STRING)),
                    keys=Numbered(""),
                ),
            ),
            stored={"loras.3.name": "x", "loras.1.name": "y"},
            inputs=[
                row("loras.3.name"),
                row("loras.3.strength", CORE_FLOAT, True, 1.0),
                row("loras.1.name"),
                row("loras.1.strength", CORE_FLOAT, True, 1.0),
            ],
        ),
        case(
            "group/key-the-naming-rule-cannot-make-is-refused",
            (Group("g", value=Input("v", INT), keys=Named(("a", "b"))),),
            stored={"g.c": 1},
            refused="naming rule cannot make",
        ),
        case(
            "group/more-members-than-the-maximum-is-refused",
            (Group("g", value=Input("v", INT), keys=Numbered(""), max_members=2),),
            stored={"g.1": 1, "g.2": 2, "g.3": 3},
            refused="more than 2 members",
        ),
        case(
            "group-minimum/optional-rows-run-with-fewer-members",
            (
                Group(
                    "any",
                    value=Input("any", STRING, required=False),
                    keys=Numbered("any_", start=1, pad=2),
                    min_members=5,
                ),
            ),
            stored={"any.any_05": "x", "any.any_02": "y"},
            inputs=[row("any.any_05", required=False), row("any.any_02", required=False)],
        ),
        case(
            "group-minimum/a-required-row-with-fewer-members-is-refused-naming-the-group",
            (
                Group(
                    "pairs",
                    member=(Input("x", INT),),
                    keys=Numbered("m_", start=1, pad=2),
                    min_members=2,
                ),
            ),
            stored={"pairs.m_01.x": 1},
            refused="pairs needs at least 2 members; the run has 1",
        ),
        case(
            "group-minimum/optional-after-minimum-counts-the-run-order",
            (
                Group(
                    "g",
                    member=(Input("x", INT),),
                    keys=Numbered("m_", start=1, pad=2),
                    min_members=2,
                    optional_after_minimum=True,
                ),
            ),
            stored={"g.m_03.x": 3, "g.m_01.x": 1, "g.m_07.x": 7},
            inputs=[
                row("g.m_03.x", CORE_INT),
                row("g.m_01.x", CORE_INT),
                row("g.m_07.x", CORE_INT, required=False),
            ],
        ),
        case(
            "choice/shared-row-under-the-chosen-option-with-its-override",
            (SHARED_FORMAT,),
            choices={"format": "webm", "format.audio": "none"},
            inputs=[
                row("format.crf", CORE_INT, True, 31),
                row("format.deadline", CORE_STRING, True, "good"),
            ],
            chose={"format": "webm", "format.audio": "none"},
        ),
        case(
            "choice/nested-choice-rows-join-the-path",
            (SHARED_FORMAT,),
            choices={"format": "webm", "format.audio": "opus"},
            stored={"format.audio.bitrate": 128},
            inputs=[
                row("format.audio.bitrate", CORE_INT, True, 96),
                row("format.crf", CORE_INT, True, 31),
                row("format.deadline", CORE_STRING, True, "good"),
            ],
            chose={"format": "webm", "format.audio": "opus"},
        ),
        case(
            "choice/the-other-option-keeps-the-shared-row-id",
            (SHARED_FORMAT,),
            choices={"format": "mp4"},
            inputs=[row("format.crf", CORE_INT, True, 23)],
            chose={"format": "mp4"},
        ),
        case(
            "choice/an-option-key-is-a-stored-value-and-may-hold-spaces-and-hyphens",
            (
                Choice(
                    "resize",
                    options=(
                        Option("keep resolution"),
                        Option("scale-to fit", rows=(Input("width", INT, default=512),)),
                    ),
                ),
            ),
            choices={"resize": "scale-to fit"},
            inputs=[row("resize.width", CORE_INT, True, 512)],
            chose={"resize": "scale-to fit"},
        ),
        case(
            "group/the-runs-member-list-orders-members-with-no-stored-input",
            (
                Group(
                    "g",
                    member=(
                        Choice(
                            "c",
                            options=(Option("x"), Option("y", rows=(Input("v", INT, default=1),))),
                        ),
                    ),
                    keys=Numbered(""),
                ),
            ),
            stored={"g.1.c.v": 5},
            members={"g": ["2", "1"]},
            choices={"g.1.c": "y", "g.2.c": "y"},
            inputs=[row("g.2.c.v", CORE_INT, True, 1), row("g.1.c.v", CORE_INT, True, 1)],
            chose={"g.2.c": "y", "g.1.c": "y"},
        ),
        case(
            "choice/a-required-choice-with-nothing-stored-is-refused",
            (SHARED_FORMAT,),
            refused="has no stored option",
        ),
        case(
            "choice/an-optional-choice-with-nothing-chosen-has-no-rows",
            (
                Choice(
                    "extra",
                    options=(Option("on", rows=(Input("x", INT, default=0),)),),
                    required=False,
                ),
            ),
        ),
        case(
            "choice-leader/a-wired-pin-picks-the-option",
            LED,
            stored={"vae": LINK},
            inputs=[row("mode.tile", CORE_INT, True, 64), row("vae", CORE_INT, False)],
            chose={"mode": "decode"},
        ),
        case(
            "choice-leader/no-wire-picks-the-other-option",
            LED,
            inputs=[row("vae", CORE_INT, False)],
            chose={"mode": "plain"},
        ),
        case(
            "choice-leader/a-stored-key-that-disagrees-is-refused",
            LED,
            stored={"vae": LINK},
            choices={"mode": "plain"},
            refused="led by a formula",
        ),
        case(
            "group-count/members-are-the-first-keys-of-the-naming-rule",
            COUNTED,
            stored={"n": 2},
            inputs=[
                row("layers.1.w", CORE_INT, True, 0),
                row("layers.2.w", CORE_INT, True, 0),
                row("n", CORE_INT, True, 2),
            ],
        ),
        case(
            "group-count/a-count-above-the-maximum-uses-the-maximum",
            COUNTED,
            stored={"n": 5},
            inputs=[
                row("layers.1.w", CORE_INT, True, 0),
                row("layers.2.w", CORE_INT, True, 0),
                row("layers.3.w", CORE_INT, True, 0),
                row("n", CORE_INT, True, 2),
            ],
        ),
        case(
            "group-count/a-row-under-an-option-not-chosen-is-unknown-outside-the-choice",
            HIDDEN_OUTSIDE,
            stored={"mode.n": 2},
            choices={"mode": "b"},
            chose={"mode": "b"},
        ),
        case(
            "group-count/a-row-of-another-option-is-unknown-inside-the-choice",
            HIDDEN_INSIDE,
            stored={"mode.n": 2},
            choices={"mode": "b"},
            chose={"mode": "b"},
        ),
        case(
            "group-count/a-row-under-the-chosen-option-counts-by-its-default",
            NESTED_DEFAULT,
            choices={"mode": "a"},
            inputs=[row("mode.n", CORE_INT, True, 2)],
            outputs=["layers.1", "layers.2"],
            chose={"mode": "a"},
        ),
        case(
            "group-count/a-shared-row-counts-by-the-default-its-use-gives",
            USE_DEFAULT,
            choices={"mode": "a"},
            inputs=[row("mode.n", CORE_INT, True, 3)],
            outputs=["layers.1", "layers.2", "layers.3"],
            chose={"mode": "a"},
        ),
        case(
            "group-count/the-same-shared-row-keeps-its-own-default-under-another-option",
            USE_DEFAULT,
            choices={"mode": "b"},
            inputs=[row("mode.n", CORE_INT, True, 1)],
            outputs=["layers.1"],
            chose={"mode": "b"},
        ),
        case(
            "group-count/a-group-under-an-option-not-chosen-has-no-size",
            GROUP_UNCHOSEN,
            stored={"c.g.1.s": 4},
            members={"c.g": ["1"]},
            choices={"c": "b"},
            outputs=["o.1", "p.1"],
            chose={"c": "b"},
        ),
        case(
            "group-count/a-count-that-reads-unknown-gives-min-members",
            UNKNOWN_COUNT,
            stored={"mode.n": 3},
            choices={"mode": "b"},
            outputs=["layers.1", "layers.2"],
            chose={"mode": "b"},
        ),
        case(
            "choice-leader/a-leader-that-reads-unknown-gives-the-first-option",
            UNKNOWN_LEADER,
            stored={"mode.k": "y"},
            choices={"mode": "b"},
            inputs=[row("led.w", CORE_INT, True, 0)],
            chose={"mode": "b", "led": "x"},
        ),
        case(
            "group-count/a-member-row-with-nothing-stored-counts-by-its-default",
            MEMBER_DEFAULT,
            stored={"g.1.t": 5, "g.2.s": 3},
            inputs=[
                row("g.1.s", CORE_INT, True, 2),
                row("g.1.t", CORE_INT, True, 0),
                row("g.2.s", CORE_INT, True, 2),
                row("g.2.t", CORE_INT, True, 0),
            ],
            outputs=["o.1", "o.2", "o.3", "o.4", "o.5"],
        ),
        case(
            "group-outputs/an-output-in-a-member-exists-with-its-member",
            (
                Group(
                    "parts",
                    member=(Input("text", STRING), Output("shout", STRING)),
                    keys=Numbered(""),
                ),
            ),
            stored={"parts.2.text": "a"},
            inputs=[row("parts.2.text")],
            outputs=["parts.2.shout"],
        ),
        case(
            "group-outputs/a-group-of-outputs-takes-its-stored-membership",
            (Group("parts", value=Output("part", STRING), keys=Numbered("")),),
            members={"parts": ["2", "1"]},
            outputs=["parts.2", "parts.1"],
        ),
        case(
            "described/entries-become-rows-under-the-described-id",
            (Described("exposed", kinds=(Input("number", INT),), max_entries=4),),
            stored={
                "exposed": json.dumps({"entries": [{"id": "a", "label": "A", "kind": "number"}]}),
                "exposed.a": 5,
            },
            inputs=[row("exposed", CORE_STRING, False), row("exposed.a", CORE_INT)],
        ),
        case(
            "type-choice/the-chosen-variant-types-the-pin-and-adds-its-rows",
            (
                TypeChoice(
                    "pixels",
                    variants=(
                        Variant("int", INT, rows=(Input("scale", INT, default=1),)),
                        Variant("float", FLOAT),
                    ),
                ),
            ),
            choices={"pixels": "int"},
            inputs=[row("pixels", CORE_INT), row("pixels.scale", CORE_INT, True, 1)],
            chose={"pixels": "int"},
        ),
        case(
            "layout/sections-and-rows-open-no-scope",
            (
                Input("z", INT, default=0),
                Section(
                    "size",
                    rows=(
                        Row(
                            rows=(
                                Input("width", INT, default=8),
                                Input("height", INT, default=8),
                            )
                        ),
                    ),
                ),
            ),
            inputs=[
                row("height", CORE_INT, True, 8),
                row("width", CORE_INT, True, 8),
                row("z", CORE_INT, True, 0),
            ],
        ),
        case(
            f"depth/{MAX_DEPTH}-boxes-of-sections-groups-and-a-row-load",
            nested(MAX_DEPTH),
            stored=deep_stored(MAX_DEPTH),
            inputs=deep_ids(MAX_DEPTH),
        ),
        case(
            f"depth/{MAX_DEPTH + 1}-boxes-are-refused-when-the-pack-loads",
            nested(MAX_DEPTH + 1),
            refused_at_load=f"nests deeper than {MAX_DEPTH}",
        ),
        case(
            "ids/options-of-one-choice-may-each-hold-a-section-with-one-id",
            (
                Choice(
                    "format",
                    options=(
                        Option(
                            "png",
                            rows=(Section("advanced", rows=(Input("level", INT, default=6),)),),
                        ),
                        Option(
                            "exr",
                            rows=(Section("advanced", rows=(Input("half", INT, default=0),)),),
                        ),
                    ),
                ),
            ),
            choices={"format": "exr"},
            inputs=[row("format.half", CORE_INT, True, 0)],
            chose={"format": "exr"},
        ),
        case(
            "ids/options-of-one-choice-may-each-hold-a-note-with-one-id",
            (
                Choice(
                    "format",
                    options=(
                        Option("png", rows=(Note("about", "Lossless"),)),
                        Option(
                            "jpg", rows=(Note("about", "Lossy"), Input("quality", INT, default=90))
                        ),
                    ),
                ),
            ),
            choices={"format": "jpg"},
            inputs=[row("format.quality", CORE_INT, True, 90)],
            chose={"format": "jpg"},
        ),
        case(
            "shared/a-section-in-shared-is-refused-when-the-pack-loads",
            (
                Choice(
                    "format",
                    options=(Option("png", rows=(Use("level"),)),),
                    shared=(Section("level", rows=(Input("level", INT, default=6),)),),
                ),
            ),
            refused_at_load="in shared is layout",
        ),
        case(
            "shared/a-note-in-shared-is-refused-when-the-pack-loads",
            (
                Choice(
                    "format",
                    options=(Option("png", rows=(Use("about"),)),),
                    shared=(Note("about", "Lossless"),),
                ),
            ),
            refused_at_load="in shared holds no value",
        ),
        case(
            "counted/reading-across-a-counted-group-is-refused-when-the-pack-loads",
            (
                Input("n", INT, default=3),
                Group(
                    "g",
                    member=(Input("s", INT, default=2),),
                    keys=Numbered(""),
                    grows=Count(Formula("n")),
                    max_members=5,
                ),
                Group(
                    "o",
                    value=Output("v", INT),
                    keys=Numbered(""),
                    grows=Count(Formula("sum(g.s)")),
                    max_members=9,
                ),
            ),
            refused_at_load="is not built yet",
        ),
        case(
            "ids/one-option-may-not-hold-two-sections-with-one-id",
            (
                Choice(
                    "format",
                    options=(
                        Option(
                            "png",
                            rows=(
                                Section("advanced", rows=(Input("level", INT, default=6),)),
                                Section("advanced", rows=(Input("fast", INT, default=0),)),
                            ),
                        ),
                    ),
                ),
            ),
            refused_at_load="declared twice",
        ),
        case(
            "choice/a-use-may-not-make-a-choice-with-a-default-optional",
            (
                Choice(
                    "format",
                    options=(Option("webm", rows=(Use("audio", required=False),)),),
                    shared=(
                        Choice("audio", options=(Option("opus"), Option("none")), default="opus"),
                    ),
                ),
            ),
            refused_at_load="an optional choice cannot declare a default",
        ),
        case(
            "ids/a-group-and-an-input-of-one-scope-may-not-share-an-id",
            (
                Input("loras", INT, default=0),
                Group("loras", value=Input("v", INT), keys=Numbered("")),
            ),
            refused_at_load="loras",
        ),
        case(
            "member-label/key-in-a-numbered-group-is-refused",
            (
                Group(
                    "g",
                    value=Input("v", INT),
                    keys=Numbered(""),
                    member_label=Formula("key()"),
                ),
            ),
            refused_at_load="key()",
        ),
    ]


APP_WORKFLOW = (
    Input("app", STRING),
    Input("overrides", STRING, default="{}", widget=StringWidget(multiline=True)),
    Choice(
        "source",
        leader=Formula('"wired" if linked(app) else "picked"'),
        options=(
            Option(
                "picked",
                rows=(
                    Button(
                        "load_inputs",
                        "Load inputs",
                        steps=(
                            Call(
                                "app_pack.exposure", args=("app",), writes=("exposed", "overrides")
                            ),
                        ),
                    ),
                    Described(
                        "exposed",
                        max_entries=32,
                        kinds=(
                            Input("number", INT),
                            Input("decimal", FLOAT),
                            Input("toggle", TypeExpr.concrete(CORE_BOOLEAN)),
                            Input("text", STRING),
                            Input("choice", TypeExpr.concrete(CORE_COMBO)),
                            Input("color", STRING),
                            Input("wire", TypeExpr.variable("A")),
                            Output("result", TypeExpr.variable("A")),
                        ),
                    ),
                ),
            ),
            Option(
                "wired",
                rows=(
                    Group(
                        "generic_in",
                        grows=Count(Formula("4")),
                        keys=Numbered("input_", start=1),
                        member=(Input("value", TypeExpr.variable("A"), required=False),),
                    ),
                    Group(
                        "generic_out",
                        grows=Count(Formula("4")),
                        keys=Numbered("output_", start=1),
                        member=(Output("value", TypeExpr.variable("A")),),
                    ),
                ),
            ),
        ),
    ),
    Output("exposed_info", TypeExpr.concrete("DICT"), label="exposed"),
)

_AUDIO_OUT = TypeExpr.concrete("AUDIO_ENCODER_OUTPUT")
_MASK = TypeExpr.concrete("MASK")
INFINITE_TALK = (
    Choice(
        "mode",
        default="single_speaker",
        options=(
            Option("single_speaker"),
            Option(
                "two_speakers",
                rows=(
                    Input("audio_encoder_output_2", _AUDIO_OUT, required=False),
                    Input("mask_1", _MASK, required=False),
                    Input("mask_2", _MASK, required=False),
                ),
            ),
        ),
    ),
    *(
        Input(name, TypeExpr.concrete(type_id))
        for name, type_id in (
            ("model", "MODEL"),
            ("model_patch", "MODEL_PATCH"),
            ("positive", "CONDITIONING"),
            ("negative", "CONDITIONING"),
            ("vae", "VAE"),
        )
    ),
    Input("width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)),
    Input("height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)),
    Input("length", INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)),
    Input("clip_vision_output", TypeExpr.concrete("CLIP_VISION_OUTPUT"), required=False),
    Input("start_image", TypeExpr.concrete("IMAGE"), required=False),
    Input("audio_encoder_output_1", _AUDIO_OUT),
    Input("audio_scale", FLOAT, default=1.0, widget=NumberWidget(min=-10.0, max=10.0, step=0.01)),
    Input("previous_frames", TypeExpr.concrete("IMAGE"), required=False),
    Section(
        "advanced",
        label="Advanced",
        rows=(
            Input("motion_frame_count", INT, default=9, widget=NumberWidget(min=1, max=33, step=1)),
        ),
    ),
    Output("model", TypeExpr.concrete("MODEL")),
    Output("positive", TypeExpr.concrete("CONDITIONING")),
    Output("negative", TypeExpr.concrete("CONDITIONING")),
    Output("latent", TypeExpr.concrete("LATENT")),
    Output("trim_image", INT),
)


def t(name: str) -> TypeExpr:
    return TypeExpr.concrete(name)


def example_cases() -> list[dict[str, object]]:
    """The section 11 gallery examples that use only what the tree builds,
    each with one stored state. Steps written as tuples in the gallery are
    the step classes here; fields the tree does not build are left out where
    they are layout, and listed on the pull request."""
    image, mask = t("IMAGE"), t("MASK")
    boolean, combo = t(CORE_BOOLEAN), t(CORE_COMBO)
    return [
        case(
            "example/3-image-batch-splitter",
            (
                Input("images", image),
                Input(
                    "split_count",
                    INT,
                    default=4,
                    pin=False,
                    widget=NumberWidget(min=0, max=50, step=1),
                ),
                Group(
                    "images_out",
                    grows=Count(Formula("split_count")),
                    keys=Numbered("image_", start=1),
                    max_members=50,
                    member=(Output("image", image, label=Formula('"image " + str(index() + 1)')),),
                ),
                Output("remained", image),
            ),
            stored={"split_count": 2},
            inputs=[row("images", "IMAGE"), row("split_count", CORE_INT, True, 4)],
            outputs=["images_out.image_1.image", "images_out.image_2.image", "remained"],
        ),
        case(
            "example/11-painter",
            (
                Input("image", image, widget=CustomWidgetDescriptor("alekpet.painter")),
                Input("images", image, required=False),
                Output("image", image),
                Output("mask", mask),
            ),
            inputs=[row("image", "IMAGE"), row("images", "IMAGE", False)],
            outputs=["image", "mask"],
        ),
        case(
            "example/12-timestep-keyframe",
            (
                Input(
                    "start_percent",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                Input("prev_timestep_kf", t("TIMESTEP_KEYFRAME"), required=False),
                Input(
                    "strength",
                    FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.001),
                ),
                Input("cn_weights", t("CONTROL_NET_WEIGHTS"), required=False),
                Input("latent_keyframe", t("LATENT_KEYFRAME"), required=False),
                Input(
                    "null_latent_kf_strength",
                    FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.001),
                ),
                Input("inherit_missing", boolean, required=False, default=True),
                Input(
                    "guarantee_steps",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=0, max=9007199254740991),
                ),
                Input("mask_optional", mask, required=False, label="mask"),
                Output("TIMESTEP_KF", t("TIMESTEP_KEYFRAME")),
            ),
            inputs=[
                row("cn_weights", "CONTROL_NET_WEIGHTS", False),
                row("guarantee_steps", CORE_INT, False, 1),
                row("inherit_missing", CORE_BOOLEAN, False, True),
                row("latent_keyframe", "LATENT_KEYFRAME", False),
                row("mask_optional", "MASK", False),
                row("null_latent_kf_strength", CORE_FLOAT, False, 0.0),
                row("prev_timestep_kf", "TIMESTEP_KEYFRAME", False),
                row("start_percent", CORE_FLOAT, True, 0.0),
                row("strength", CORE_FLOAT, False, 1.0),
            ],
            outputs=["TIMESTEP_KF"],
        ),
        case(
            "example/14-kling-text-to-video",
            (
                Input("prompt", STRING, widget=StringWidget(multiline=True)),
                Input("negative_prompt", STRING, widget=StringWidget(multiline=True)),
                Input("cfg_scale", FLOAT, default=1.0, widget=NumberWidget(min=0.0, max=1.0)),
                Input(
                    "aspect_ratio",
                    combo,
                    default="16:9",
                    widget=ComboWidget(options=("16:9", "9:16", "1:1")),
                ),
                Input(
                    "mode",
                    combo,
                    default="pro mode / 5s duration / kling-v2-5-turbo",
                    widget=ComboWidget(
                        options=(
                            "pro mode / 5s duration / kling-v2-5-turbo",
                            "pro mode / 10s duration / kling-v2-5-turbo",
                        )
                    ),
                ),
                Output("video", t("VIDEO")),
                Output("video_id", STRING),
                Output("duration", STRING),
            ),
            inputs=[
                row("aspect_ratio", CORE_COMBO, True, "16:9"),
                row("cfg_scale", CORE_FLOAT, True, 1.0),
                row("mode", CORE_COMBO, True, "pro mode / 5s duration / kling-v2-5-turbo"),
                row("negative_prompt"),
                row("prompt"),
            ],
            outputs=["duration", "video", "video_id"],
        ),
        case(
            "example/15-replace-text-regex",
            (
                Input("string", STRING, widget=StringWidget(multiline=True)),
                Input("regex_pattern", STRING, widget=StringWidget(multiline=True)),
                Input("replace", STRING, widget=StringWidget(multiline=True)),
                Section(
                    "advanced",
                    label="Advanced",
                    collapsed=True,
                    rows=(
                        Input("case_insensitive", boolean, required=False, default=True),
                        Input("multiline", boolean, required=False, default=False),
                        Input("dotall", boolean, required=False, default=False),
                        Input(
                            "count",
                            INT,
                            required=False,
                            default=0,
                            widget=NumberWidget(min=0, max=100),
                        ),
                    ),
                ),
                Output("STRING", STRING),
            ),
            inputs=[
                row("case_insensitive", CORE_BOOLEAN, False, True),
                row("count", CORE_INT, False, 0),
                row("dotall", CORE_BOOLEAN, False, False),
                row("multiline", CORE_BOOLEAN, False, False),
                row("regex_pattern"),
                row("replace"),
                row("string"),
            ],
            outputs=["STRING"],
        ),
        case(
            "example/5-text-switch-case",
            (
                Input(
                    "switch_cases",
                    STRING,
                    default="",
                    widget=StringWidget(
                        multiline=True,
                        dynamic_prompts=False,
                        placeholder="case_1:output_1\ncase_2:output_2",
                    ),
                ),
                Input("default_value", STRING, default=""),
                Input("delimiter", STRING, default=":"),
                Input("condition", STRING, default=""),
                Output("STRING", STRING),
            ),
            inputs=[
                row("condition", CORE_STRING, True, ""),
                row("default_value", CORE_STRING, True, ""),
                row("delimiter", CORE_STRING, True, ":"),
                row("switch_cases", CORE_STRING, True, ""),
            ],
            outputs=["STRING"],
        ),
        case(
            "example/6-wanvideo-decode-tiles",
            (
                Input("vae", t("WANVAE")),
                Input("samples", t("LATENT")),
                Input("enable_vae_tiling", boolean, default=False),
                *(
                    Input(
                        tile_id,
                        INT,
                        default=default,
                        widget=NumberWidget(min=low, max=high, step=8),
                    )
                    for tile_id, default, low, high in (
                        ("tile_x", 272, 40, 2048),
                        ("tile_y", 272, 40, 2048),
                        ("tile_stride_x", 144, 32, 2040),
                        ("tile_stride_y", 128, 32, 2040),
                    )
                ),
                Section(
                    "advanced",
                    label="Advanced",
                    rows=(
                        Input(
                            "normalization",
                            combo,
                            required=False,
                            default="default",
                            widget=ComboWidget(options=("default", "minmax", "none")),
                        ),
                    ),
                ),
                Output("images", image),
            ),
            inputs=[
                row("enable_vae_tiling", CORE_BOOLEAN, True, False),
                row("normalization", CORE_COMBO, False, "default"),
                row("samples", "LATENT"),
                row("tile_stride_x", CORE_INT, True, 144),
                row("tile_stride_y", CORE_INT, True, 128),
                row("tile_x", CORE_INT, True, 272),
                row("tile_y", CORE_INT, True, 272),
                row("vae", "WANVAE"),
            ],
            outputs=["images"],
        ),
        case(
            "example/7-app-workflow-picked",
            APP_WORKFLOW,
            stored={
                "app": "upscale.app.json",
                "source.exposed": json.dumps(
                    {
                        "entries": [
                            {"id": "steps", "label": "Steps", "kind": "number"},
                            {"id": "upscaled", "label": "Upscaled", "kind": "result"},
                        ]
                    }
                ),
                "source.exposed.steps": 20,
            },
            inputs=[
                row("app"),
                row("overrides", CORE_STRING, True, "{}"),
                row("source.exposed", CORE_STRING, False),
                row("source.exposed.steps", CORE_INT),
            ],
            outputs=["exposed_info", "source.exposed.upscaled"],
            chose={"source": "picked"},
        ),
        case(
            "example/7-app-workflow-wired",
            APP_WORKFLOW,
            stored={"app": {"link": True}},
            inputs=[
                row("app"),
                row("overrides", CORE_STRING, True, "{}"),
                *(row(f"source.generic_in.input_{n}.value", None, False) for n in range(1, 5)),
            ],
            outputs=[
                "exposed_info",
                *(f"source.generic_out.output_{n}.value" for n in range(1, 5)),
            ],
            chose={"source": "wired"},
        ),
        case(
            "example/10-prompt-relay-timeline",
            (
                Input("model", t("MODEL")),
                Input("clip", t("CLIP")),
                Input("latent", t("LATENT")),
                Input("global_prompt", STRING, default="", widget=StringWidget(multiline=True)),
                Input(
                    "max_frames",
                    INT,
                    default=129,
                    widget=NumberWidget(min=1, max=10000, step=1),
                ),
                Input(
                    "timeline_data",
                    STRING,
                    default="",
                    widget=CustomWidgetDescriptor(
                        "promptrelay.timeline",
                        writes={
                            "prompts": "local_prompts",
                            "lengths": "segment_lengths",
                        },
                        reads={"frames": "max_frames", "fps": "fps", "units": "time_units"},
                    ),
                ),
                Input("local_prompts", STRING, default="", hidden=True),
                Input("segment_lengths", STRING, default="", hidden=True),
                Input(
                    "epsilon",
                    FLOAT,
                    default=1e-3,
                    widget=NumberWidget(min=1e-6, max=0.99, step=1e-4),
                ),
                Input(
                    "fps",
                    FLOAT,
                    required=False,
                    default=24.0,
                    widget=NumberWidget(min=0.1, max=240.0, step=0.1),
                ),
                Input(
                    "time_units",
                    combo,
                    required=False,
                    default="frames",
                    widget=ComboWidget(options=("frames", "seconds")),
                ),
                Input("relay_options", t("RELAY_OPTIONS"), required=False),
                Output("model", t("MODEL")),
                Output("positive", t("CONDITIONING")),
            ),
            inputs=[
                row("clip", "CLIP"),
                row("epsilon", CORE_FLOAT, True, 1e-3),
                row("fps", CORE_FLOAT, False, 24.0),
                row("global_prompt", CORE_STRING, True, ""),
                row("latent", "LATENT"),
                row("local_prompts", CORE_STRING, True, ""),
                row("max_frames", CORE_INT, True, 129),
                row("model", "MODEL"),
                row("relay_options", "RELAY_OPTIONS", False),
                row("segment_lengths", CORE_STRING, True, ""),
                row("time_units", CORE_COMBO, False, "frames"),
                row("timeline_data", CORE_STRING, True, ""),
            ],
            outputs=["model", "positive"],
        ),
        case(
            "example/17-wan-infinite-talk-two-speakers",
            INFINITE_TALK,
            choices={"mode": "two_speakers"},
            inputs=[
                row("audio_encoder_output_1", "AUDIO_ENCODER_OUTPUT"),
                row("audio_scale", CORE_FLOAT, True, 1.0),
                row("clip_vision_output", "CLIP_VISION_OUTPUT", False),
                row("height", CORE_INT, True, 480),
                row("length", CORE_INT, True, 81),
                row("mode.audio_encoder_output_2", "AUDIO_ENCODER_OUTPUT", False),
                row("mode.mask_1", "MASK", False),
                row("mode.mask_2", "MASK", False),
                row("model", "MODEL"),
                row("model_patch", "MODEL_PATCH"),
                row("motion_frame_count", CORE_INT, True, 9),
                row("negative", "CONDITIONING"),
                row("positive", "CONDITIONING"),
                row("previous_frames", "IMAGE", False),
                row("start_image", "IMAGE", False),
                row("vae", "VAE"),
                row("width", CORE_INT, True, 832),
            ],
            outputs=["latent", "model", "negative", "positive", "trim_image"],
            chose={"mode": "two_speakers"},
        ),
        case(
            "example/21-minimax-reference-prompt",
            (
                Input("clip", t("CLIP")),
                Input("vae", t("VAE"), required=False),
                Input("audio_vae", t("VAE"), required=False),
                Input(
                    "prompt",
                    STRING,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                Input("width", INT, default=1344, widget=NumberWidget(min=32, max=16384, step=32)),
                Input("height", INT, default=768, widget=NumberWidget(min=32, max=16384, step=32)),
                Input("length", INT, default=124, widget=NumberWidget(min=5, max=3600, step=17)),
                Input(
                    "ref_image_size",
                    combo,
                    default="match",
                    widget=ComboWidget(options=("match", "max")),
                ),
                Group(
                    "ref_images",
                    keys=Numbered("ref_image_", start=1),
                    max_members=9,
                    member_label=Formula('"Picture " + str(index() + 1)'),
                    member=(Input("image", image, required=False),),
                ),
                Group(
                    "ref_videos",
                    keys=Numbered("ref_video_", start=1),
                    max_members=3,
                    member_label=Formula('"Video " + str(index() + 1)'),
                    member=(
                        Input("frames", image, required=False),
                        Input("audio", t("AUDIO"), required=False),
                    ),
                ),
                Group(
                    "ref_audios",
                    keys=Numbered("ref_audio_", start=1),
                    max_members=3,
                    member_label=Formula('"Audio " + str(index() + 1)'),
                    member=(Input("audio", t("AUDIO"), required=False),),
                ),
                Output("positive", t("CONDITIONING")),
                Output("LATENT", t("LATENT")),
            ),
            stored={"ref_images.ref_image_2.image": {"link": True}},
            members={"ref_images": ["ref_image_2", "ref_image_1"], "ref_videos": ["ref_video_1"]},
            inputs=[
                row("audio_vae", "VAE", False),
                row("clip", "CLIP"),
                row("height", CORE_INT, True, 768),
                row("length", CORE_INT, True, 124),
                row("prompt"),
                row("ref_image_size", CORE_COMBO, True, "match"),
                row("ref_images.ref_image_2.image", "IMAGE", False),
                row("ref_images.ref_image_1.image", "IMAGE", False),
                row("ref_videos.ref_video_1.audio", "AUDIO", False),
                row("ref_videos.ref_video_1.frames", "IMAGE", False),
                row("vae", "VAE", False),
                row("width", CORE_INT, True, 1344),
            ],
            outputs=["LATENT", "positive"],
        ),
        case(
            "example/23-seed-with-reuse",
            (
                Input(
                    "seed",
                    INT,
                    default=0,
                    widget=NumberWidget(min=-1125899906842624, max=1125899906842624),
                    after_run="randomize",
                ),
                Button(
                    "new_fixed_random",
                    "New fixed random",
                    steps=(SetMode("seed", "fixed"), Randomize("seed")),
                ),
                Output("SEED", INT),
            ),
            inputs=[row("seed", CORE_INT, True, 0)],
            outputs=["SEED"],
        ),
    ]


def render() -> str:
    data = {"schemaVersion": SCHEMA_WIRE_VERSION, "cases": [*cases(), *example_cases()]}
    return json.dumps(data, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    text = render()
    if parser.parse_args().check:
        if OUTPUT.read_text("ascii") != text:
            print(f"{OUTPUT} differs from the generator", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text, "ascii")
    return 0


if __name__ == "__main__":
    sys.exit(main())
