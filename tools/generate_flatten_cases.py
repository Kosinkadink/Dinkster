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
    Choice,
    Count,
    Described,
    Formula,
    Group,
    Named,
    Note,
    Numbered,
    Option,
    Row,
    Section,
    TypeChoice,
    Use,
    Variant,
)
from dinkster_schema.model import Input, Output, TypeExpr
from dinkster_schema.wire import SCHEMA_WIRE_VERSION, body_row_to_wire
from dinkster_values import CORE_FLOAT, CORE_INT, CORE_STRING

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "fixtures" / f"flatten_v{SCHEMA_WIRE_VERSION}.json"

INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)
LINK = {"link": True}

type Flat = list[object]


def row(
    row_id: str, type_id: str = CORE_STRING, required: bool = True, default: object = None
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


def render() -> str:
    data = {"schemaVersion": SCHEMA_WIRE_VERSION, "cases": cases()}
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
