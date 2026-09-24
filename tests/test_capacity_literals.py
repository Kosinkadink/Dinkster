from __future__ import annotations

import ast
from pathlib import Path

from scripts.check_capacity_literals import (
    LIMITS_MODULES,
    capacity_literals,
    governed_sources,
    is_byte_capacity_expression,
)


def test_capacity_literal_shapes_are_detected() -> None:
    expressions = ("32 << 20", "6 << 30", "512 * 1024 * 1024", "2 * 1024**2")
    for expression in expressions:
        node = ast.parse(expression, mode="eval").body
        assert is_byte_capacity_expression(node), expression

    for expression in ("value << 12", "1024 * 1024.0", "8 * chunk_size"):
        node = ast.parse(expression, mode="eval").body
        assert not is_byte_capacity_expression(node), expression


def test_capacity_literal_scan_rejects_limits_and_chunking(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    source.write_text(
        "MAX_BLOB_BYTES = 8 * 1024 * 1024\n"
        "chunk_size = 8 * 1024 * 1024\n"
        "if payload_size > 16 * 1024 * 1024:\n"
        "    raise ValueError\n",
        encoding="utf-8",
    )

    assert capacity_literals(source) == [(1, 18), (2, 14), (3, 19)]


def test_repository_has_no_undeclared_capacity_literals() -> None:
    violations = {
        path: capacity_literals(path) for path in governed_sources() if capacity_literals(path)
    }
    assert violations == {}


def test_capacity_literal_gate_has_exactly_two_limit_homes() -> None:
    homes = {
        path.relative_to(Path(__file__).parent.parent).as_posix() for path in LIMITS_MODULES
    }
    assert homes == {
        "packages/dinkster-supervisor/src/dinkster_supervisor/limits.py",
        "packages/dinkster-values/src/dinkster_values/limits.py",
    }
