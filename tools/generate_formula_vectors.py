"""Write the formula language vector file that the engine and the editor both load.

Usage: python tools/generate_formula_vectors.py [--check]

Cases come from ``tools/formula_cases/*.json`` plus a seeded generated set. The
engine evaluator (``dinkster_schema.formula``) gives each case its outcome. A
case that names an ``expect`` must get exactly that answer. Every case plain
Python can express is also run through Python's own ``eval`` with the
language's functions written out below, and the file is not written if the
two disagree. ``--check`` regenerates in memory and fails if the committed
file differs. After a change, update the hash in ``tests/test_formula.py`` and
copy the file into Dinkster-Frontend ``packages/core/test/fixtures/``.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import math
import random
import struct
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from dinkster_schema.formula import (
    FORMULA_LANGUAGE_VERSION,
    UNKNOWN,
    Failed,
    FormulaRefused,
    FormulaScope,
    FormulaValues,
    parse_formula,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tools" / "formula_cases"
OUTPUT = ROOT / "tests" / "fixtures" / f"formula_v{FORMULA_LANGUAGE_VERSION}.json"
SEED = 20261121
GENERATED = 4000


# --- encoding ---------------------------------------------------------------


def encode(value: object) -> object:
    if value is None:
        return {"none": True}
    if type(value) is bool:
        return {"bool": value}
    if type(value) is int:
        number = cast(int, value)
        return {"int": ("-" if number < 0 else "") + hex(abs(number))}
    if type(value) is float:
        return {"float": struct.pack(">d", value).hex()}
    if type(value) is str:
        return {"text": value}
    if isinstance(value, (list, tuple)):
        return {"list": [encode(item) for item in cast(Sequence[object], value)]}
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {"table": [[encode(key), encode(item)] for key, item in mapping.items()]}
    raise TypeError(f"cannot encode {value!r}")


def decode(data: object) -> object:
    item = cast(Mapping[str, object], data)
    ((tag, body),) = item.items()
    if tag == "none":
        return None
    if tag == "bool":
        return bool(body)
    if tag == "int":
        text = cast(str, body)
        return -int(text[1:], 16) if text.startswith("-") else int(text, 16)
    if tag == "float":
        return cast(float, struct.unpack(">d", bytes.fromhex(cast(str, body)))[0])
    if tag == "text":
        return cast(str, body)
    if tag == "list":
        return [decode(part) for part in cast(list[object], body)]
    if tag == "table":
        return {decode(key): decode(value) for key, value in cast(list[list[object]], body)}
    raise ValueError(f"bad value encoding {data!r}")


def scope_of(case: Mapping[str, object]) -> FormulaScope:
    raw = cast(Mapping[str, object], case.get("scope", {}))
    return FormulaScope(
        names=cast(dict[str, str], raw.get("names", {})),
        inputs=frozenset(cast(list[str], raw.get("inputs", []))),
        assets=frozenset(cast(list[str], raw.get("assets", []))),
        tables={
            name: decode(table)
            for name, table in cast(Mapping[str, object], raw.get("tables", {})).items()
        },
        derived=cast(dict[str, str], raw.get("derived", {})),
        in_member=bool(raw.get("in_member", False)),
    )


def values_of(case: Mapping[str, object]) -> FormulaValues:
    raw = cast(Mapping[str, object], case.get("values", {}))
    values = {
        name: UNKNOWN if value == "unknown" else decode(value)
        for name, value in cast(Mapping[str, object], raw.get("values", {})).items()
    }
    return FormulaValues(
        values=values,
        linked=cast(dict[str, bool], raw.get("linked", {})),
        linked_types=cast(dict[str, str | None], raw.get("linked_types", {})),
        group_sizes=cast(dict[str, int], raw.get("group_sizes", {})),
        member_index=cast(int | None, raw.get("member_index")),
        member_key=cast(str | None, raw.get("member_key")),
        facts={
            (fact, name): decode(value)
            for fact, name, value in cast(list[list[object]], raw.get("facts", []))
            if isinstance(fact, str) and isinstance(name, str)
        },
    )


def answer(case: Mapping[str, object]) -> dict[str, object]:
    try:
        formula = parse_formula(cast(str, case["formula"]), scope_of(case))
    except FormulaRefused:
        return {"outcome": "refused"}
    result = formula.evaluate(values_of(case))
    if result is UNKNOWN:
        return {"outcome": "unknown"}
    if isinstance(result, Failed):
        return {"outcome": "failed"}
    return {"outcome": "value", "value": encode(result)}


# --- the reference in plain Python -----------------------------------------
# Written from the language's definition, not from the evaluator, so a mistake
# in one shows up as a difference.


def _ref_sum(items: Sequence[object]) -> object:
    total: object = 0
    for item in items:
        total = total + item  # type: ignore[operator]
    return total


def _ref_clamp(x: object, low: object, high: object) -> object:
    if x < low:  # type: ignore[operator]
        return low
    if x > high:  # type: ignore[operator]
        return high
    return x


def _ref_lines(text: str) -> list[str]:
    parts = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return [part.strip(" \t\n\r") for part in parts if part.strip(" \t\n\r")]


def _ref_number(text: str) -> object:
    import re

    if not re.fullmatch(r"[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?", text):
        raise ValueError(text)
    return int(text) if re.fullmatch(r"[+-]?[0-9]+", text) else float(text)


def _ref_str(value: object) -> str:
    if type(value) not in (int, bool, str):
        raise TypeError(value)
    return str(value)


def _ref_split(text: str, separator: str) -> list[str]:
    if not separator:
        raise ValueError("empty separator")
    return text.split(separator)


REFERENCE: Mapping[str, Callable[..., object]] = {
    "abs": abs,
    "sign": lambda x: (x > 0) - (x < 0),
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "min": min,
    "max": max,
    "clamp": _ref_clamp,
    "sum": _ref_sum,
    "sqrt": math.sqrt,
    "mod": lambda a, b: a % b,
    "lerp": lambda a, b, t: a + (b - a) * t,
    "int": int,
    "float": float,
    "len": len,
    "lower": lambda t: "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in t),
    "upper": lambda t: "".join(chr(ord(c) - 32) if "a" <= c <= "z" else c for c in t),
    "strip": lambda t: t.strip(" \t\n\r"),
    "startswith": lambda t, p: t.startswith(p),
    "endswith": lambda t, p: t.endswith(p),
    "replace": lambda t, old, new: t.replace(old, new),
    "split": _ref_split,
    "lines": _ref_lines,
    "join": lambda sep, items: sep.join(items),
    "str": _ref_str,
    "fixed": lambda x, digits: format(float(x), f".{digits}f"),
    "number": _ref_number,
    "any": any,
    "all": all,
    "count": lambda items, x: items.count(x),
    "position": lambda items, x: items.index(x),
    "range": lambda n: list(builtins.range(n)),
    "get": lambda table, key, default: table.get(key, default),
}

_FACTS = {
    "linked",
    "linked_type",
    "index",
    "key",
    "known",
    "width",
    "height",
    "frames",
    "fps",
    "duration",
    "kind",
    "filename",
}


class _TuplesAsLists(ast.NodeTransformer):
    def visit_Tuple(self, node: ast.Tuple) -> ast.AST:
        self.generic_visit(node)
        return ast.copy_location(ast.List(elts=node.elts, ctx=ast.Load()), node)


def python_answer(case: Mapping[str, object]) -> object | None:
    """Plain Python's answer, or ``None`` when Python cannot express the case."""
    scope = scope_of(case)
    if scope.derived or any("." in name for name in scope.names):
        return None
    tree = ast.parse(cast(str, case["formula"]).strip(" \t\n\r\f"), mode="eval")  # already accepted
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FACTS:
                return None
        if isinstance(node, ast.Name) and scope.names.get(node.id) in ("group", "opaque"):
            return None
    values = values_of(case)
    namespace: dict[str, object] = {"__builtins__": {}}
    namespace.update(REFERENCE)
    namespace.update(scope.tables)
    for name, kind in scope.names.items():
        value = values.values.get(name, UNKNOWN)
        if value is UNKNOWN:
            return None
        if kind == "float":
            value = float(cast(float, value))
        elif kind == "float[]":
            value = [float(item) for item in cast(list[float], value)]
        namespace[name] = value
    code = compile(ast.fix_missing_locations(_TuplesAsLists().visit(tree)), "<formula>", "eval")
    try:
        result = eval(code, namespace)  # noqa: S307 - the formula passed the language check
    except Exception:  # noqa: BLE001
        return _RAISED
    try:
        return encode(result)
    except TypeError:
        return None


_RAISED = object()


# --- generated cases --------------------------------------------------------


def generated_cases(rng: random.Random, count: int) -> list[dict[str, object]]:
    """Formulas over three numbers, a text and a list, mixing every kind of operation."""
    cases: list[dict[str, object]] = []
    numbers = ("a", "b", "c")
    binary = ("+", "-", "*", "/", "//", "%", "**", "<<", ">>", "|", "^", "&")
    compare = ("==", "!=", "<", "<=", ">", ">=")
    unary_functions = ("abs", "sign", "floor", "ceil", "round", "int", "float", "sqrt")

    def number_literal() -> str:
        choice = rng.random()
        if choice < 0.4:
            return str(rng.randint(-20, 20))
        if choice < 0.5:
            return str(rng.choice((2**53, 2**53 + 1, 10**20, -(2**63), 3**40)))
        if choice < 0.85:
            return repr(round(rng.uniform(-50, 50), rng.randint(0, 6)))
        return rng.choice(("0.1", "0.5", "2.5", "1e-3", "1e300", "-0.0", "True", "False"))

    def numeric(depth: int) -> str:
        if depth <= 0 or rng.random() < 0.25:
            return rng.choice(numbers) if rng.random() < 0.6 else number_literal()
        roll = rng.random()
        if roll < 0.45:
            return f"({numeric(depth - 1)} {rng.choice(binary)} {numeric(depth - 1)})"
        if roll < 0.6:
            return f"{rng.choice(unary_functions)}({numeric(depth - 1)})"
        if roll < 0.68:
            return f"round({numeric(depth - 1)}, {rng.randint(-3, 6)})"
        if roll < 0.76:
            name = rng.choice(("min", "max"))
            return f"{name}({numeric(depth - 1)}, {numeric(depth - 1)}, {numeric(depth - 1)})"
        if roll < 0.82:
            return f"clamp({numeric(depth - 1)}, {numeric(depth - 1)}, {numeric(depth - 1)})"
        if roll < 0.88:
            return (
                f"({numeric(depth - 1)} if {numeric(depth - 1)} {rng.choice(compare)} "
                f"{numeric(depth - 1)} else {numeric(depth - 1)})"
            )
        if roll < 0.93:
            return f"({numeric(depth - 1)} {rng.choice(('and', 'or'))} {numeric(depth - 1)})"
        if roll < 0.97:
            return f"sum([{numeric(depth - 1)}, {numeric(depth - 1)}, {numeric(depth - 1)}])"
        return f"lerp({numeric(depth - 1)}, {numeric(depth - 1)}, {numeric(depth - 1)})"

    def textual(depth: int) -> str:
        roll = rng.random()
        if depth <= 0 or roll < 0.2:
            return rng.choice(("t", '"Wan"', '" a b "', '""', '"x,y,,z"', '"\\u00e9"'))
        if roll < 0.35:
            return f"({textual(depth - 1)} + {textual(depth - 1)})"
        if roll < 0.5:
            return f"{rng.choice(('lower', 'upper', 'strip'))}({textual(depth - 1)})"
        if roll < 0.6:
            return f"str({numeric(depth - 1)})"
        if roll < 0.7:
            return f"fixed({numeric(depth - 1)}, {rng.randint(0, 12)})"
        if roll < 0.8:
            return f'join("-", split({textual(depth - 1)}, ","))'
        if roll < 0.9:
            return f'replace({textual(depth - 1)}, "a", {textual(depth - 1)})'
        return f"{textual(depth - 1)}[{rng.randint(-3, 3)}]"

    def listy() -> str:
        roll = rng.random()
        if roll < 0.4:
            return f"[x * {number_literal()} for x in items if x > {number_literal()}]"
        if roll < 0.6:
            return f"len([x for x in range({rng.randint(-2, 12)}) if x % 2 == 0])"
        if roll < 0.8:
            return f"items[{rng.randint(-4, 4)}]"
        return f"{rng.choice(('any', 'all', 'sum', 'min', 'max', 'len'))}(items)"

    for index in range(count):
        roll = rng.random()
        if roll < 0.7:
            formula = numeric(rng.randint(1, 4))
        elif roll < 0.88:
            formula = textual(rng.randint(1, 3))
            if rng.random() < 0.4:
                formula = (
                    f"len({formula})"
                    if rng.random() < 0.5
                    else (f"{formula} {rng.choice(('==', '!=', 'in', 'not in'))} {textual(1)}")
                )
        else:
            formula = listy()
        item_list = [
            rng.choice((rng.randint(-5, 5), round(rng.uniform(-5, 5), 2)))
            for _ in range(rng.randint(0, 5))
        ]
        cases.append(
            {
                "id": f"generated-{index:05d}",
                "formula": formula,
                "scope": {
                    "names": {
                        "a": "int",
                        "b": "float",
                        "c": "int",
                        "t": "text",
                        "items": "float[]",
                    },
                },
                "values": {
                    "values": {
                        "a": encode(rng.randint(-1000, 1000)),
                        "b": encode(float(round(rng.uniform(-100, 100), rng.randint(0, 4)))),
                        "c": encode(rng.choice((0, 1, -1, 2, 7, 2**31, -(2**53)))),
                        "t": encode(
                            rng.choice(("", "Wan", "a,b", " Mixed Case ", "\u00e9t\u00e9"))
                        ),
                        "items": encode([float(item) for item in item_list]),
                    }
                },
            }
        )
    return cases


# --- the file ---------------------------------------------------------------


def build() -> tuple[str, dict[str, int]]:
    sys.set_int_max_str_digits(0)
    warnings.simplefilter("ignore", SyntaxWarning)
    cases: list[dict[str, object]] = []
    for path in sorted(CASES.glob("*.json")):
        source = json.loads(path.read_text(encoding="utf-8"))
        for case in source:
            case = dict(case)
            case["id"] = f"{path.stem}/{case['id']}"
            cases.append(case)
    cases.extend(generated_cases(random.Random(SEED), GENERATED))
    seen: set[str] = set()
    problems: list[str] = []
    counts: dict[str, int] = {}
    written: list[dict[str, object]] = []
    for case in cases:
        identifier = cast(str, case["id"])
        if identifier in seen:
            problems.append(f"{identifier}: duplicate id")
        seen.add(identifier)
        result = answer(case)
        expected = case.get("expect")
        if expected is not None and expected != result:
            problems.append(f"{identifier}: expected {expected}, evaluator gave {result}")
        # A failed case is not compared: the language bounds work Python does not,
        # and Python would spend minutes on some of them.
        if result["outcome"] == "value":
            python = python_answer(case)
            if python is _RAISED:
                problems.append(f"{identifier}: Python raises, evaluator gave {result}")
            elif python is not None:
                if result["value"] != python:
                    problems.append(f"{identifier}: Python gives {python}, evaluator gave {result}")
                else:
                    counts["python-agrees"] = counts.get("python-agrees", 0) + 1
        counts[str(result["outcome"])] = counts.get(str(result["outcome"]), 0) + 1
        record = {key: case[key] for key in ("id", "formula", "scope", "values") if key in case}
        record.update(result)
        written.append(record)
    if problems:
        raise SystemExit("refusing to write the vector file:\n" + "\n".join(problems))
    lines = [json.dumps(record, ensure_ascii=True, sort_keys=True) for record in written]
    body = (
        '{"language": '
        + str(FORMULA_LANGUAGE_VERSION)
        + ', "cases": [\n'
        + ",\n".join(lines)
        + "\n]}\n"
    )
    return body, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the formula language vector file.")
    parser.add_argument("--check", action="store_true", help="fail if the committed file differs")
    args = parser.parse_args()
    body, counts = build()
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()
    if args.check:
        if OUTPUT.read_text(encoding="ascii") != body:
            print(f"{OUTPUT} is out of date; rerun without --check")
            return 1
    else:
        OUTPUT.write_text(body, encoding="ascii")
    print(json.dumps(counts, sort_keys=True))
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
