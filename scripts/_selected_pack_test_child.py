"""Trusted engine test launcher for a disposable selected-pack environment."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def select_available_tests(tests: list[str], modules: dict[str, str]) -> None:
    """Exclude tests of unpublished symbols, not their shipped neighbors."""
    for filename in tests:
        path = Path(filename)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=filename)
        unavailable: dict[str, str] = {}
        for statement in tree.body:
            if not isinstance(statement, ast.ImportFrom) or statement.level or not statement.module:
                continue
            module = statement.module
            if not any(module == name or module.startswith(name + ".") for name in modules):
                continue
            try:
                imported = __import__(module, fromlist=[alias.name for alias in statement.names])
            except ModuleNotFoundError as error:
                # A broken dependency inside a shipped module is an admission failure.
                if error.name is None or not (
                    module == error.name or module.startswith(error.name + ".")
                ):
                    raise
                imported = None
            available = []
            for alias in statement.names:
                if alias.name == "*":
                    raise ValueError(f"selected pack tests require explicit imports: {filename}")
                if imported is None or not hasattr(imported, alias.name):
                    unavailable[alias.asname or alias.name] = f"{module}:{alias.name}"
                else:
                    available.append(alias)
            statement.names = available
        if not unavailable:
            continue
        excluded = dict(unavailable)
        removed: set[int] = set()
        # Helpers, fixtures, decorators and aliases can carry an unavailable
        # symbol into a test without that test importing it directly.
        while True:
            changed = False
            for index, statement in enumerate(tree.body):
                if index in removed or isinstance(statement, (ast.Import, ast.ImportFrom)):
                    continue
                dependencies = {
                    node.id
                    for node in ast.walk(statement)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                }
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    dependencies.update(arg.arg for arg in statement.args.args)
                    dependencies.update(arg.arg for arg in statement.args.kwonlyargs)
                missing = dependencies & excluded.keys()
                if not missing:
                    continue
                if not isinstance(
                    statement,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.Assign,
                        ast.AnnAssign,
                    ),
                ):
                    raise ValueError(
                        f"unavailable pack symbol in module setup: {filename}:{statement.lineno}"
                    )
                reason = ", ".join(sorted({excluded[name] for name in missing}))
                names = {
                    node.id
                    for node in ast.walk(statement)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                }
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names = {statement.name}
                    print(
                        f"Excluded {filename}::{statement.name}: selected pack lacks {reason}",
                        flush=True,
                    )
                excluded.update(dict.fromkeys(names, reason))
                removed.add(index)
                changed = True
            if not changed:
                break
        tree.body = [
            statement
            for index, statement in enumerate(tree.body)
            if index not in removed
            and not (isinstance(statement, ast.ImportFrom) and not statement.names)
        ]
        print(f"Unavailable selected imports in {filename}: {unavailable}", flush=True)
        path.write_text(ast.unparse(tree) + "\n", encoding="utf-8")


def main() -> int:
    config_path = Path(sys.argv[1])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for module, root in config["modules"].items():
        spec = importlib.util.find_spec(module)
        if (
            spec is None
            or spec.origin is None
            or not Path(spec.origin).resolve().is_relative_to(Path(root).resolve())
        ):
            raise RuntimeError(f"selected pack provenance refused for {module}: {spec}")
        print(f"Selected module {module}: {spec.origin}", flush=True)
    if len(sys.argv) > 2 and sys.argv[2] == "--check-only":
        return 0
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            str(config_path),
            "--check-only",
        ],
        check=True,
    )
    from dinkster_workers.doctor import prepare_catalog

    from dinkster.compose import default_pack_spec

    for name in config["packs"]:
        spec = default_pack_spec(name)
        report = prepare_catalog(spec.manifest, interpreter=sys.executable, environment=spec.env)
        if not report.ok:
            raise RuntimeError(f"selected {name} runtime catalog refused: {report.to_json()}")
    select_available_tests(config["tests"], config["modules"])
    import pytest

    class RequireExecutedTests:
        passed = 0

        def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
            if report.when == "call" and report.passed:
                self.passed += 1

        def pytest_sessionfinish(self, session: pytest.Session) -> None:
            if self.passed == 0:
                session.exitstatus = pytest.ExitCode.NO_TESTS_COLLECTED

    return int(pytest.main(["-q", *config["tests"]], plugins=[RequireExecutedTests()]))


if __name__ == "__main__":
    raise SystemExit(main())
