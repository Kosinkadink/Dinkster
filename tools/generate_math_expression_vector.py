"""Generate the deterministic scalar math expression mirror parity vector.

The committed vector (tests/fixtures/math_expression_v1.json) is the golden
corpus for the dinkster.math.expression frontend mirror. ComfyUI-compatible cases
come from executing the pinned ComfyMathExpression source. The remaining cases
cover safe grammar extensions shared by the backend and frontend mirror. Each
case carries a mirror class:

- ``exact``: every operation in the expression is correctly rounded under
  IEEE-754 binary64 (arithmetic, sqrt, comparisons, rounding, integer
  semantics), so any conforming evaluator reproduces the recorded outputs
  bit for bit on every platform. ``float_bits`` records the normative
  big-endian bit pattern of each float output.
- ``bounded``: the expression uses transcendentals, whose libm and JS-engine
  implementations differ by ulps across platforms. Recorded float outputs
  (and their ``float_bits``) are the generator platform's values and are
  informative; conforming evaluators must agree within the recorded
  ``mirror_relative_tolerance``.

``int_strings`` records exact decimal digits whenever an integer output's
magnitude exceeds 2**53: consumers parsing this file with binary64 JSON numbers
(JavaScript) must read that output from the string.

Regeneration may change bounded-case float outputs (within tolerance) when
the platform libm differs; exact-case data must never change. The
regeneration test compares through ``regeneration_stable_view`` so a
platform-local rebuild of bounded floats is not a corpus change.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
import subprocess
import sys
import types
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TypedDict, cast

from dinkster_nodes_foundation.expression import EXPRESSION_GRAMMAR_VERSION, evaluate_expression

OUTPUT_PATH = Path(__file__).parents[1] / "tests" / "fixtures" / "math_expression_v1.json"
COMFYUI_REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
COMFYUI_REFERENCE_PATH = "comfy_extras/nodes_math.py"
COMFYUI_SOURCE_SHA256 = "b117db3e5eadf7e1689a7a2edaad9436bb6282ab3d57865ec2cd78004de9fb88"
SIMPLEEVAL_REFERENCE_VERSION = "1.0.3"
MIRROR_RELATIVE_TOLERANCE = 1e-12
MAX_JSON_SAFE_INTEGER = 2**53
ReferenceExecute = Callable[[str, Mapping[str, object]], object]


class CaseSpec(TypedDict):
    id: str
    compatibility: str
    mirror_class: str
    expression: str
    inputs: dict[str, object]


CASES: tuple[CaseSpec, ...] = (
    {
        "id": "operator-precedence",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "exact",
        "expression": "a + b * 2",
        "inputs": {"a": 3, "b": 2.5},
    },
    {
        "id": "division-and-int-output",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "exact",
        "expression": "a / b",
        "inputs": {"a": 7, "b": 2},
    },
    {
        "id": "core-functions",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "bounded",
        "expression": "round(sqrt(a) + log2(b), 3)",
        "inputs": {"a": 9, "b": 8},
    },
    {
        "id": "conditional",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "exact",
        "expression": "a if a > b else b",
        "inputs": {"a": -4, "b": -1},
    },
    {
        "id": "ordered-values",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "exact",
        "expression": "sum(values)",
        "inputs": {"a": 1, "b": 2.5, "c": True},
    },
    {
        "id": "highest-named-operand",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "exact",
        "expression": "a + z",
        "inputs": {"a": 1, "z": 2.5},
    },
    {
        "id": "floor-division-and-modulo-negatives",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "exact",
        "expression": "(a // b) * 1000 + a % b",
        "inputs": {"a": -7, "b": 3},
    },
    {
        "id": "float-modulo-follows-divisor-sign",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "exact",
        "expression": "a % b",
        "inputs": {"a": -7.5, "b": 3.0},
    },
    {
        "id": "bankers-rounding",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "exact",
        "expression": "round(a) + round(b) * 10",
        "inputs": {"a": 2.5, "b": 3.5},
    },
    {
        "id": "big-integer-power",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "exact",
        "expression": "a ** b + 1",
        "inputs": {"a": 2, "b": 80},
    },
    {
        "id": "bitwise-and-shifts",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "exact",
        "expression": "(a << 5) ^ b | (a & b)",
        "inputs": {"a": 13, "b": 9},
    },
    {
        "id": "sqrt-irrational",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "exact",
        "expression": "sqrt(a)",
        "inputs": {"a": 2.0},
    },
    {
        "id": "negative-zero-product",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "exact",
        "expression": "a * b",
        "inputs": {"a": -1.0, "b": 0.0},
    },
    {
        "id": "lerp-and-minmax",
        "compatibility": "dinkster-extension",
        "mirror_class": "exact",
        "expression": "lerp(min(a, b), max(a, b), 0.25)",
        "inputs": {"a": 0.1, "b": 0.7},
    },
    {
        "id": "sin-cos-product",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "bounded",
        "expression": "sin(a) * cos(b)",
        "inputs": {"a": 1.2, "b": 0.4},
    },
    {
        "id": "exp-log-round-trip",
        "compatibility": "dinkster-extension",
        "mirror_class": "bounded",
        "expression": "log(exp(a)) + log(b, 10)",
        "inputs": {"a": 0.75, "b": 1000},
    },
    {
        "id": "float-power",
        "compatibility": "ComfyMathExpression",
        "mirror_class": "bounded",
        "expression": "pow(a, b)",
        "inputs": {"a": 2.0, "b": 0.5},
    },
    {
        "id": "atan2-and-tan",
        "compatibility": "dinkster-extension",
        "mirror_class": "bounded",
        "expression": "atan2(a, b) + tan(a / 4)",
        "inputs": {"a": 1.0, "b": 2.0},
    },
)


def _float_bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def _reference_result(value: object) -> tuple[float, int, bool]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or type(value[0]) is not float
        or type(value[1]) is not int
        or type(value[2]) is not bool
    ):
        raise ValueError(f"unexpected ComfyMathExpression output: {value!r}")
    return cast("tuple[float, int, bool]", value)


def _assert_reference_parity(
    spec: CaseSpec,
    dinkster: tuple[float, int, bool],
    reference: tuple[float, int, bool],
) -> None:
    if dinkster[1:] != reference[1:]:
        raise ValueError(f"{spec['id']}: integer or boolean output differs from ComfyUI")
    if spec["mirror_class"] == "exact":
        if _float_bits(dinkster[0]) != _float_bits(reference[0]):
            raise ValueError(f"{spec['id']}: float output differs from ComfyUI")
        return
    bound = MIRROR_RELATIVE_TOLERANCE * max(abs(reference[0]), sys.float_info.min)
    if abs(dinkster[0] - reference[0]) > bound:
        raise ValueError(f"{spec['id']}: float output exceeds the ComfyUI tolerance")


def build_vector(reference_execute: ReferenceExecute | None = None) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for spec in CASES:
        results = evaluate_expression(spec["expression"], spec["inputs"])
        if reference_execute is not None and spec["compatibility"] == "ComfyMathExpression":
            reference = _reference_result(reference_execute(spec["expression"], spec["inputs"]))
            _assert_reference_parity(spec, results, reference)
            results = reference
        float_result, int_result, _ = results
        case: dict[str, object] = {
            **spec,
            "outputs": dict(zip(("float", "int", "boolean"), results, strict=True)),
            "float_bits": _float_bits(float_result),
        }
        if abs(int_result) > MAX_JSON_SAFE_INTEGER:
            case["int_strings"] = str(int_result)
        cases.append(case)
    return {
        "format_version": 2,
        "grammar_version": EXPRESSION_GRAMMAR_VERSION,
        "comfyui_behavioral_reference_commit": COMFYUI_REFERENCE_COMMIT,
        "comfyui_reference": {
            "commit": COMFYUI_REFERENCE_COMMIT,
            "path": COMFYUI_REFERENCE_PATH,
            "source_sha256": COMFYUI_SOURCE_SHA256,
            "executed_symbol": "MathExpressionNode.execute",
            "simpleeval_version": SIMPLEEVAL_REFERENCE_VERSION,
            "generator": "tools/generate_math_expression_vector.py",
        },
        "mirror_relative_tolerance": MIRROR_RELATIVE_TOLERANCE,
        "cases": cases,
    }


def regeneration_stable_view(vector: dict[str, object]) -> dict[str, object]:
    """The vector with bounded-case float data blanked.

    Bounded floats are the generator platform's libm values, so rebuilding on
    another platform may legally move them within tolerance. Everything else
    (case set, inputs, exact outputs, int/boolean outputs, bit patterns of
    exact floats) must be regeneration-stable."""
    view = copy.deepcopy(vector)
    for case in cast("list[dict[str, object]]", view["cases"]):
        if case["mirror_class"] != "bounded":
            continue
        outputs = cast("dict[str, object]", case["outputs"])
        outputs["float"] = None
        case["float_bits"] = None
    return view


def load_reference(comfyui: Path) -> ReferenceExecute:
    resolved = subprocess.run(
        ["git", "-C", str(comfyui), "rev-parse", f"{COMFYUI_REFERENCE_COMMIT}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != COMFYUI_REFERENCE_COMMIT:
        raise SystemExit(
            f"reference commit resolution mismatch: {resolved} != {COMFYUI_REFERENCE_COMMIT}"
        )
    source = subprocess.run(
        [
            "git",
            "-C",
            str(comfyui),
            "show",
            f"{COMFYUI_REFERENCE_COMMIT}:{COMFYUI_REFERENCE_PATH}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != COMFYUI_SOURCE_SHA256:
        raise SystemExit(f"reference source sha256 mismatch: {digest} != {COMFYUI_SOURCE_SHA256}")
    try:
        installed_simpleeval = version("simpleeval")
    except PackageNotFoundError:
        raise SystemExit(
            "simpleeval is required; run with uv run --with simpleeval==1.0.3"
        ) from None
    if installed_simpleeval != SIMPLEEVAL_REFERENCE_VERSION:
        raise SystemExit(
            f"simpleeval {installed_simpleeval} is installed; "
            f"expected {SIMPLEEVAL_REFERENCE_VERSION}"
        )

    def node_output(*values: object) -> tuple[object, ...]:
        return values

    class ReferenceAutogrow:
        Type = dict[str, object]

    io = types.SimpleNamespace(
        Autogrow=ReferenceAutogrow,
        ComfyNode=object,
        NodeOutput=node_output,
        Schema=object,
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.__dict__["ComfyExtension"] = object
    latest.__dict__["io"] = io
    package = types.ModuleType("comfy_api")
    package.__path__ = []
    package.__dict__["latest"] = latest
    previous = {name: sys.modules.get(name) for name in ("comfy_api", "comfy_api.latest")}
    try:
        sys.modules["comfy_api"] = package
        sys.modules["comfy_api.latest"] = latest
        module = types.ModuleType("comfy_math_expression_reference")
        exec(
            compile(
                source,
                f"{COMFYUI_REFERENCE_COMMIT}:{COMFYUI_REFERENCE_PATH}",
                "exec",
            ),
            module.__dict__,
        )
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
    node = cast("Any", module.__dict__["MathExpressionNode"])
    return cast("ReferenceExecute", node.execute)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", type=Path, required=True)
    args = parser.parse_args()
    OUTPUT_PATH.write_text(
        json.dumps(
            build_vector(load_reference(args.comfyui.resolve())), indent=2, ensure_ascii=True
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
