"""The engine evaluator gives every answer in the formula vector file.

The editor loads the same file (Dinkster-Frontend
``packages/core/test/fixtures/formula_v1.json``), so a formula a pack declares
gives the same answer in the browser and when a run is submitted.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from dinkster_schema.formula import (
    UNKNOWN,
    Failed,
    FormulaRefused,
    FormulaScope,
    FormulaValues,
    parse_formula,
)

VECTORS = Path(__file__).parent / "fixtures" / "formula_v1.json"
VECTORS_SHA256 = "a50e41758bc77c607b6e01c30511264d325fb8e2a5a6a2536dd288b4a8a69ca6"


def _decode(data: object) -> object:
    ((tag, body),) = cast(Mapping[str, object], data).items()
    if tag == "none":
        return None
    if tag == "bool":
        return body
    if tag == "int":
        text = cast(str, body)
        return -int(text[1:], 16) if text.startswith("-") else int(text, 16)
    if tag == "float":
        return struct.unpack(">d", bytes.fromhex(cast(str, body)))[0]
    if tag == "text":
        return body
    if tag == "list":
        return [_decode(item) for item in cast(list[object], body)]
    if tag == "table":
        return {_decode(key): _decode(value) for key, value in cast(list[list[object]], body)}
    raise AssertionError(f"bad encoding {data!r}")


def _same(actual: object, expected: object) -> bool:
    """Equal in value and in type, with floats compared bit for bit."""
    if isinstance(expected, list):
        items = cast(list[object], expected)
        return (
            isinstance(actual, tuple)
            and len(actual) == len(items)
            and all(
                _same(a, e) for a, e in zip(cast(tuple[object, ...], actual), items, strict=True)
            )
        )
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, float):
        return struct.pack(">d", cast(float, actual)) == struct.pack(">d", expected)
    return actual == expected


def _run(case: Mapping[str, object]) -> str | None:
    scope_data = cast(Mapping[str, object], case.get("scope", {}))
    scope = FormulaScope(
        names=cast(dict[str, str], scope_data.get("names", {})),
        inputs=frozenset(cast(list[str], scope_data.get("inputs", []))),
        assets=frozenset(cast(list[str], scope_data.get("assets", []))),
        tables={
            name: _decode(table)
            for name, table in cast(Mapping[str, object], scope_data.get("tables", {})).items()
        },
        derived=cast(dict[str, str], scope_data.get("derived", {})),
        in_member=bool(scope_data.get("in_member", False)),
    )
    values_data = cast(Mapping[str, object], case.get("values", {}))
    values = FormulaValues(
        values={
            name: UNKNOWN if value == "unknown" else _decode(value)
            for name, value in cast(Mapping[str, object], values_data.get("values", {})).items()
        },
        linked=cast(dict[str, bool], values_data.get("linked", {})),
        linked_types=cast(dict[str, str | None], values_data.get("linked_types", {})),
        group_sizes=cast(dict[str, int], values_data.get("group_sizes", {})),
        member_index=cast(int | None, values_data.get("member_index")),
        member_key=cast(str | None, values_data.get("member_key")),
        facts={
            (cast(str, fact), cast(str, name)): _decode(value)
            for fact, name, value in cast(list[list[object]], values_data.get("facts", []))
        },
    )
    expected = case["outcome"]
    try:
        formula = parse_formula(cast(str, case["formula"]), scope)
    except FormulaRefused as exc:
        return None if expected == "refused" else f"refused ({exc}), expected {expected}"
    if expected == "refused":
        return "accepted, expected refused"
    result = formula.evaluate(values)
    if result is UNKNOWN:
        return None if expected == "unknown" else f"unknown, expected {expected}"
    if isinstance(result, Failed):
        return None if expected == "failed" else f"failed ({result.reason}), expected {expected}"
    if expected != "value":
        return f"value {result!r}, expected {expected}"
    wanted = _decode(case["value"])
    return None if _same(result, wanted) else f"value {result!r}, expected {wanted!r}"


def test_vector_file_is_the_one_both_sides_check() -> None:
    assert hashlib.sha256(VECTORS.read_bytes()).hexdigest() == VECTORS_SHA256


def test_engine_gives_every_vector_answer() -> None:
    cases = cast(list[Mapping[str, object]], json.loads(VECTORS.read_text("ascii"))["cases"])
    assert len(cases) > 5000
    mismatches = [
        f"{case['id']} {case['formula']!r}: {problem}"
        for case in cases
        if (problem := _run(case)) is not None
    ]
    assert not mismatches, "\n".join(mismatches[:50])
