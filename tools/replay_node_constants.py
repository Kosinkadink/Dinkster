"""Replay explicitly annotated ComfyUI node constants without changing either checkout."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ANNOTATION_PREFIX = "# comfy-replay:"


@dataclass(frozen=True)
class ReplayCase:
    line: int
    class_path: str
    method: str
    inputs: dict[str, object]
    expected: object


def parse_cases(test_path: Path) -> list[ReplayCase]:
    source = test_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(test_path))
    cases: list[ReplayCase] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert) or node.lineno < 2:
            continue
        annotation = lines[node.lineno - 2].strip()
        if not annotation.startswith(ANNOTATION_PREFIX):
            continue
        class_path, method, inputs = _parse_metadata(
            annotation.removeprefix(ANNOTATION_PREFIX).strip(), node.lineno
        )
        expected = _expected_literal(node, node.lineno)
        cases.append(
            ReplayCase(
                line=node.lineno,
                class_path=class_path,
                method=method,
                inputs=inputs,
                expected=expected,
            )
        )
    if not cases:
        raise ValueError(
            f"{test_path}: no {ANNOTATION_PREFIX} annotation immediately before an assert"
        )
    return sorted(cases, key=lambda case: case.line)


def _parse_metadata(text: str, line: int) -> tuple[str, str, dict[str, object]]:
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"line {line}: invalid replay metadata: {error.msg}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"line {line}: replay metadata must be a JSON object")
    class_path = metadata.get("class")
    method = metadata.get("method")
    inputs = metadata.get("inputs")
    if not isinstance(class_path, str) or "." not in class_path:
        raise ValueError(f"line {line}: replay metadata class must be a dotted class path")
    if not isinstance(method, str) or not method:
        raise ValueError(f"line {line}: replay metadata method must be a non-empty string")
    if not isinstance(inputs, dict) or not all(isinstance(key, str) for key in inputs):
        raise ValueError(
            f"line {line}: replay metadata inputs must be a JSON object with string keys"
        )
    return class_path, method, inputs


def _expected_literal(node: ast.Assert, line: int) -> object:
    test = node.test
    if (
        not isinstance(test, ast.Compare)
        or len(test.ops) != 1
        or not isinstance(test.ops[0], ast.Eq)
    ):
        raise ValueError(f"line {line}: replay assertion must use == with a literal expected value")
    try:
        return ast.literal_eval(test.comparators[0])
    except ValueError as error:
        raise ValueError(f"line {line}: expected value must be a Python literal") from error


def replay_case(case: ReplayCase) -> object:
    module_name, _, class_name = case.class_path.rpartition(".")
    module = importlib.import_module(module_name)
    node_class = getattr(module, class_name)
    return getattr(node_class(), case.method)(**case.inputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfyui", type=Path, help="Path to the ComfyUI checkout to execute")
    parser.add_argument("test", type=Path, help="Annotated Dinkster node test file")
    arguments = parser.parse_args()
    if not arguments.comfyui.is_dir():
        parser.error(f"ComfyUI checkout is not a directory: {arguments.comfyui}")
    if not arguments.test.is_file():
        parser.error(f"Dinkster node test is not a file: {arguments.test}")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(arguments.comfyui.resolve()))
    try:
        cases = parse_cases(arguments.test)
        mismatches = 0
        for case in cases:
            actual = replay_case(case)
            if actual == case.expected:
                print(f"MATCH line {case.line}: {case.class_path}.{case.method} -> {actual!r}")
            else:
                mismatches += 1
                print(
                    f"MISMATCH line {case.line}: {case.class_path}.{case.method}\n"
                    f"  expected: {case.expected!r}\n"
                    f"  actual:   {actual!r}"
                )
    except (AttributeError, ImportError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
