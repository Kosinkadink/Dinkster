"""Send formulas through the engine's and the editor's evaluators and report any difference.

Usage: python tools/compare_formula_evaluators.py [--frontend PATH] [--seed N] [--count N]

The editor's evaluator is ``packages/core/src/mirror/formula.ts`` in a
Dinkster-Frontend checkout (default: the sibling directory), transpiled with
that checkout's TypeScript and run in Node. The inputs are every case of the
vector file, formulas from the vector generator with a fresh seed, the vector
formulas with one character inserted, deleted or doubled, and random token
strings that probe the whole-text check. A case differs when the outcome
(refused, unknown, failed, value) or the value differs. Exit status 1 when
any case differs.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_formula_vectors import OUTPUT, answer, generated_cases  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

_BUILD = r"""
import { readFileSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { join } from 'node:path'
const [source, target] = process.argv.slice(2)
const ts = createRequire(join(source, '..', '..', '..', 'package.json'))('typescript')
for (const name of ['expression', 'formula']) {
  const text = readFileSync(join(source, `${name}.ts`), 'utf8')
  const out = ts.transpileModule(text, { compilerOptions: { target: 'ES2022', module: 'ESNext' } })
  writeFileSync(join(target, `${name}.js`), out.outputText)
}
"""

_RUN = r"""
import { readFileSync, writeFileSync } from 'node:fs'
import { Formula, FormulaRefused, FormulaTable, UNKNOWN } from './formula.js'

function decode(data) {
  const [[tag, body]] = Object.entries(data)
  if (tag === 'none') return null
  if (tag === 'bool') return body
  if (tag === 'int') return body.startsWith('-') ? -BigInt(body.slice(1)) : BigInt(body)
  if (tag === 'float') {
    const view = new DataView(new ArrayBuffer(8))
    view.setBigUint64(0, BigInt(`0x${body}`))
    return view.getFloat64(0)
  }
  if (tag === 'text') return body
  if (tag === 'list') return Object.freeze(body.map(decode))
  if (tag === 'table') {
    return new FormulaTable(body.map(([key, value]) => [decode(key), decode(value)]))
  }
  throw new Error(`bad encoding ${JSON.stringify(data)}`)
}

function encode(value) {
  if (value === null) return { none: true }
  if (typeof value === 'boolean') return { bool: value }
  if (typeof value === 'bigint') {
    return { int: value < 0n ? `-0x${(-value).toString(16)}` : `0x${value.toString(16)}` }
  }
  if (typeof value === 'number') {
    const view = new DataView(new ArrayBuffer(8))
    view.setFloat64(0, value)
    return { float: view.getBigUint64(0).toString(16).padStart(16, '0') }
  }
  if (typeof value === 'string') return { text: value }
  return { list: value.map(encode) }
}

function run(testCase) {
  const s = testCase.scope ?? {}
  const scope = {
    names: new Map(Object.entries(s.names ?? {})),
    inputs: new Set(s.inputs ?? []),
    assets: new Set(s.assets ?? []),
    tables: new Map(Object.entries(s.tables ?? {}).map(([name, table]) => [name, decode(table)])),
    derived: new Map(Object.entries(s.derived ?? {})),
    inMember: s.in_member ?? false,
  }
  const v = testCase.values ?? {}
  const values = {
    values: new Map(
      Object.entries(v.values ?? {}).map(([n, x]) => [n, x === 'unknown' ? UNKNOWN : decode(x)]),
    ),
    linked: new Map(Object.entries(v.linked ?? {})),
    linkedTypes: new Map(Object.entries(v.linked_types ?? {})),
    groupSizes: new Map(Object.entries(v.group_sizes ?? {})),
    memberIndex: v.member_index ?? null,
    memberKey: v.member_key ?? null,
    facts: new Map((v.facts ?? []).map(([fact, name, x]) => [`${fact}(${name})`, decode(x)])),
  }
  let formula
  try {
    formula = Formula.parse(testCase.formula, scope)
  } catch (error) {
    if (error instanceof FormulaRefused) return { outcome: 'refused' }
    return { outcome: `threw ${error}` }
  }
  const result = formula.evaluate(values)
  if (result.kind !== 'value') return { outcome: result.kind }
  return { outcome: 'value', value: encode(result.value) }
}

const [input, output] = process.argv.slice(2)
const lines = readFileSync(input, 'utf8').split('\n').filter((line) => line !== '')
writeFileSync(output, lines.map((line) => JSON.stringify(run(JSON.parse(line)))).join('\n') + '\n')
"""

_SCOPE: Mapping[str, object] = {
    "names": {
        "a": "int",
        "b": "float",
        "t": "text",
        "on": "bool",
        "items": "float[]",
        "pick": "choice",
    },
    "inputs": ["video"],
    "tables": {
        "PRESETS": {
            "table": [[{"text": "sd"}, {"int": "0x200"}], [{"int": "0x1"}, {"text": "one"}]]
        }
    },
    "derived": {"half": "a / 2"},
}
_VALUES: Mapping[str, object] = {
    "values": {
        "a": {"int": "0x7"},
        "b": {"float": "4004000000000000"},
        "t": {"text": "Wan 2,1"},
        "on": {"bool": True},
        "items": {"list": [{"float": "3ff0000000000000"}, {"float": "c000000000000000"}]},
        "pick": {"text": "sd"},
    },
    "linked": {"video": True},
    "facts": [["fps", "video", {"float": "4038000000000000"}]],
}
_TOKENS = (
    "a",
    "b",
    "t",
    "on",
    "items",
    "pick",
    "PRESETS",
    "half",
    "video",
    "x",
    "unknown_name",
    "0",
    "1",
    "7",
    "00",
    "012",
    "0x1F",
    "0X",
    "0o7",
    "0b1",
    "1j",
    "1_000",
    "1__0",
    "1_",
    ".5",
    "1.",
    "1e5",
    "1E-2",
    "2.5e",
    "1.5",
    "3",
    "-",
    "+",
    "*",
    "/",
    "//",
    "%",
    "**",
    "<<",
    ">>",
    "|",
    "^",
    "&",
    "~",
    "@",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "=",
    ":=",
    ":",
    ";",
    ",",
    ".",
    "...",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "and",
    "or",
    "not",
    "in",
    "is",
    "if",
    "else",
    "for",
    "lambda",
    "None",
    "True",
    "False",
    "'x'",
    '"y"',
    "''",
    "'a\\nb'",
    "'\\q'",
    "b'x'",
    "f'x'",
    "'unclosed",
    '"""z"""',
    "#",
    "\\",
    "$",
    "?",
    "!",
    "len",
    "min",
    "max",
    "round",
    "str",
    "int",
    "known",
    "linked",
    "fps",
    "range",
    "get",
    "sum",
    "lines",
    "fixed",
    "clamp",
    "abs",
)
_GAPS = ("", " ", " ", " ", "\t", "\n", "\r\n", "\f")


def token_cases(rng: random.Random, count: int) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for index in range(count):
        parts = [rng.choice(_TOKENS) for _ in range(rng.randint(1, 9))]
        formula = "".join(part + rng.choice(_GAPS) for part in parts)
        cases.append(
            {"id": f"tokens-{index:05d}", "formula": formula, "scope": _SCOPE, "values": _VALUES}
        )
    return cases


def mutated_cases(
    rng: random.Random, cases: list[dict[str, object]], count: int
) -> list[dict[str, object]]:
    alphabet = " ()[],.:'\"+-*/%<>=!_0123456789abexn\\\t\n"
    mutated: list[dict[str, object]] = []
    for index in range(count):
        case = dict(rng.choice(cases))
        text = cast(str, case["formula"])
        at = rng.randint(0, len(text))
        roll = rng.random()
        if roll < 0.4:
            text = text[:at] + rng.choice(alphabet) + text[at:]
        elif roll < 0.7:
            text = text[:at] + text[at + 1 :]
        else:
            text = text[:at] + text[at : at + 1] * 2 + text[at + 1 :]
        case["formula"] = text
        case["id"] = f"mutated-{index:05d}-{case['id']}"
        mutated.append(case)
    return mutated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the engine and editor formula evaluators."
    )
    parser.add_argument("--frontend", type=Path, default=ROOT.parent / "Dinkster-Frontend")
    parser.add_argument("--seed", type=int, default=random.SystemRandom().randrange(2**32))
    parser.add_argument("--count", type=int, default=5000, help="cases of each generated kind")
    args = parser.parse_args()
    source = (args.frontend / "packages" / "core" / "src" / "mirror").resolve()
    if not (source / "formula.ts").is_file():
        parser.error(f"no editor evaluator at {source / 'formula.ts'}")
    sys.set_int_max_str_digits(0)
    rng = random.Random(args.seed)
    vectors = cast(list[dict[str, object]], json.loads(OUTPUT.read_text("ascii"))["cases"])
    cases = [
        *vectors,
        *generated_cases(rng, args.count),
        *mutated_cases(rng, vectors, args.count),
        *token_cases(rng, args.count),
    ]
    with tempfile.TemporaryDirectory() as work:
        folder = Path(work)
        (folder / "package.json").write_text('{"type": "module"}')
        (folder / "build.mjs").write_text(_BUILD)
        (folder / "run.mjs").write_text(_RUN)
        subprocess.run(["node", str(folder / "build.mjs"), str(source), work], check=True)
        inputs = folder / "cases.jsonl"
        inputs.write_text("".join(json.dumps(case) + "\n" for case in cases))
        subprocess.run(
            ["node", str(folder / "run.mjs"), str(inputs), str(folder / "out.jsonl")], check=True
        )
        editor = [json.loads(line) for line in (folder / "out.jsonl").read_text().splitlines()]
    differences = 0
    for case, theirs in zip(cases, editor, strict=True):
        ours = answer(case)
        if ours != theirs:
            differences += 1
            if differences <= 50:
                print(f"{case['id']} {case['formula']!r}: engine {ours}, editor {theirs}")
    print(f"seed {args.seed}: {len(cases)} cases, {differences} differ")
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
