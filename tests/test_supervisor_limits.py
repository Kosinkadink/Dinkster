from __future__ import annotations

import ast
from pathlib import Path

from dinkster_supervisor.limits import (
    INGRESS_CLIENT_MAX_SIZE_BYTES,
    INGRESS_EVENT_FRAME_LIMIT_BYTES,
    INGRESS_JOB_SUBMISSION_LIMIT_BYTES,
)
from dinkster_values.limits import (
    ENGINE_EVENT_FRAME_LIMIT_BYTES,
    ENGINE_JOB_SUBMISSION_LIMIT_BYTES,
)

SUPERVISOR_SOURCE = (
    Path(__file__).parent.parent
    / "packages"
    / "dinkster-supervisor"
    / "src"
    / "dinkster_supervisor"
)


def test_supervisor_imports_no_other_dinkster_packages() -> None:
    violations: list[tuple[str, int, str]] = []
    for path in sorted(SUPERVISOR_SOURCE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: tuple[str, ...] = ()
            line = 0
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
                line = node.lineno
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = (node.module,)
                line = node.lineno
            for name in names:
                if name.startswith("dinkster_") and not name.startswith("dinkster_supervisor"):
                    violations.append((path.name, line, name))
    assert violations == []


def test_supervisor_ingress_limits_are_not_tighter_than_engine_limits() -> None:
    assert INGRESS_CLIENT_MAX_SIZE_BYTES >= INGRESS_JOB_SUBMISSION_LIMIT_BYTES
    assert INGRESS_JOB_SUBMISSION_LIMIT_BYTES >= ENGINE_JOB_SUBMISSION_LIMIT_BYTES
    assert INGRESS_EVENT_FRAME_LIMIT_BYTES >= ENGINE_EVENT_FRAME_LIMIT_BYTES
