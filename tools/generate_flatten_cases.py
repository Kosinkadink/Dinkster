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
    Numbered,
    Option,
    Row,
    Section,
    TypeChoice,
    Use,
    Variant,
)
from dinkster_schema.model import InputSpec, OutputSpec, TypeExpr
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
        Row(rows=(InputSpec("a", INT, default=0), InputSpec("b", INT, default=0))),
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
                InputSpec("deadline", STRING, default="good"),
                Choice(
                    "audio",
                    options=(
                        Option("none"),
                        Option("opus", rows=(InputSpec("bitrate", INT, default=96),)),
                    ),
                ),
            ),
        ),
    ),
    shared=(InputSpec("crf", INT, default=23),),
)

LED = (
    InputSpec("vae", INT, required=False),
    Choice(
        "mode",
        options=(Option("plain"), Option("decode", rows=(InputSpec("tile", INT, default=64),))),
        leader=Formula('"decode" if linked(vae) else "plain"'),
    ),
)

COUNTED = (
    InputSpec("n", INT, default=2),
    Group(
        "layers",
        member=(InputSpec("w", INT, default=0),),
        keys=Numbered(""),
        grows=Count(Formula("n")),
        max_members=3,
    ),
)


def cases() -> list[dict[str, object]]:
    return [
        case(
            "group-of-values/ids-are-group-dot-key-in-run-order",
            (Group("images", value=InputSpec("image", STRING), keys=Numbered("image", start=0)),),
            stored={"images.image2": "b", "images.image0": "a"},
            inputs=[row("images.image2"), row("images.image0")],
        ),
        case(
            "group-of-records/ids-are-group-dot-key-dot-row",
            (
                Group(
                    "loras",
                    member=(InputSpec("strength", FLOAT, default=1.0), InputSpec("name", STRING)),
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
            (Group("g", value=InputSpec("v", INT), keys=Named(("a", "b"))),),
            stored={"g.c": 1},
            refused="naming rule cannot make",
        ),
        case(
            "group/more-members-than-the-maximum-is-refused",
            (Group("g", value=InputSpec("v", INT), keys=Numbered(""), max_members=2),),
            stored={"g.1": 1, "g.2": 2, "g.3": 3},
            refused="more than 2 members",
        ),
        case(
            "group-minimum/optional-rows-run-with-fewer-members",
            (
                Group(
                    "any",
                    value=InputSpec("any", STRING, required=False),
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
                    member=(InputSpec("x", INT),),
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
                    member=(InputSpec("x", INT),),
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
            "choice/a-required-choice-with-nothing-stored-is-refused",
            (SHARED_FORMAT,),
            refused="has no stored option",
        ),
        case(
            "choice/an-optional-choice-with-nothing-chosen-has-no-rows",
            (
                Choice(
                    "extra",
                    options=(Option("on", rows=(InputSpec("x", INT, default=0),)),),
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
            "group-outputs/an-output-in-a-member-exists-with-its-member",
            (
                Group(
                    "parts",
                    member=(InputSpec("text", STRING), OutputSpec("shout", STRING)),
                    keys=Numbered(""),
                ),
            ),
            stored={"parts.2.text": "a"},
            inputs=[row("parts.2.text")],
            outputs=["parts.2.shout"],
        ),
        case(
            "group-outputs/a-group-of-outputs-takes-its-stored-membership",
            (Group("parts", value=OutputSpec("part", STRING), keys=Numbered("")),),
            members={"parts": ["2", "1"]},
            outputs=["parts.2", "parts.1"],
        ),
        case(
            "described/entries-become-rows-under-the-described-id",
            (Described("exposed", kinds=(InputSpec("number", INT),), max_entries=4),),
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
                        Variant("int", INT, rows=(InputSpec("scale", INT, default=1),)),
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
                InputSpec("z", INT, default=0),
                Section(
                    "size",
                    rows=(
                        Row(
                            rows=(
                                InputSpec("width", INT, default=8),
                                InputSpec("height", INT, default=8),
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
            "ids/a-group-and-an-input-of-one-scope-may-not-share-an-id",
            (
                InputSpec("loras", INT, default=0),
                Group("loras", value=InputSpec("v", INT), keys=Numbered("")),
            ),
            refused_at_load="loras",
        ),
        case(
            "member-label/key-in-a-numbered-group-is-refused",
            (
                Group(
                    "g",
                    value=InputSpec("v", INT),
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
