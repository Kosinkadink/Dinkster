"""Trusted engine test launcher for a disposable selected-pack environment."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Never

import pytest
from dinkster_registry.archive_admission import admission_timing

_running_test = False


class UnavailableExport:
    """Defer an unpublished import until pytest can skip its actual consumer."""

    def __init__(self, identity: str) -> None:
        self.identity = identity

    def _unavailable(self) -> Never:
        message = f"selected pack lacks {self.identity}"
        if not _running_test:
            raise RuntimeError(f"{message}; used outside an executing test")
        pytest.skip(message)

    def __getattr__(self, name: str) -> Never:
        if not _running_test and (name.startswith("_") or name == "pytestmark"):
            raise AttributeError(name)
        self._unavailable()

    def __call__(self, *args: object, **kwargs: object) -> Never:
        self._unavailable()

    def __bool__(self) -> Never:
        self._unavailable()

    def __eq__(self, other: object) -> Never:
        self._unavailable()


def select_available_tests(tests: list[str], modules: dict[str, str]) -> None:
    """Preserve test definitions and pytest's fixture graph, deferring missing exports only."""
    sys.modules["_selected_pack_test_child"] = sys.modules[__name__]
    for filename in tests:
        path = Path(filename)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=filename)
        body: list[ast.stmt] = []
        changed = False
        for statement in tree.body:
            if not isinstance(statement, ast.ImportFrom) or statement.level or not statement.module:
                body.append(statement)
                continue
            module = statement.module
            if not any(module == name or module.startswith(name + ".") for name in modules):
                body.append(statement)
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
            deferred = []
            for alias in statement.names:
                if alias.name == "*":
                    raise ValueError(f"selected pack tests require explicit imports: {filename}")
                if imported is None or not hasattr(imported, alias.name):
                    identity = f"{module}:{alias.name}"
                    name = alias.asname or alias.name
                    deferred.extend(
                        ast.parse(
                            "from _selected_pack_test_child import UnavailableExport "
                            "as _selected_pack_unavailable\n"
                            f"{name} = _selected_pack_unavailable({identity!r})"
                        ).body
                    )
                    print(f"Unavailable selected import in {filename}: {identity}", flush=True)
                    changed = True
                else:
                    available.append(alias)
            if available:
                statement.names = available
                body.append(statement)
            body.extend(deferred)
        if changed:
            tree.body = body
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
    nodeid = sys.argv[3] if len(sys.argv) > 2 and sys.argv[2] == "--nodeid" else None
    identity = config.get("testPack", "selected-tests")
    with admission_timing(identity, "child-provenance"):
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

    for name in () if nodeid else config["packs"]:
        with admission_timing(identity, f"catalog:{name}"):
            spec = default_pack_spec(name)
            report = prepare_catalog(
                spec.manifest, interpreter=config["runtimes"][name], environment=spec.env
            )
            if not report.ok:
                raise RuntimeError(f"selected {name} runtime catalog refused: {report.to_json()}")
    # Importlib collection isolates duplicate test names; sibling helpers still need search paths.
    for directory in dict.fromkeys(str(Path(test).resolve().parent) for test in config["tests"]):
        sys.path.append(directory)
    with admission_timing(identity, "available-imports"):
        select_available_tests(config["tests"], config["modules"])

    class RequireExecutedTests:
        passed: set[str] = set()

        def pytest_configure(self, config: pytest.Config) -> None:
            config.addinivalue_line(
                "markers", "pack_runtime(name): execute a cross-pack case in its locked runtime"
            )

        @pytest.hookimpl(tryfirst=True)
        def pytest_pyfunc_call(self, pyfuncitem: pytest.Function) -> bool | None:
            marker = pyfuncitem.get_closest_marker("pack_runtime")
            if marker is None:
                return None
            [pack] = marker.args
            python = config["runtimes"][pack]
            if Path(python).absolute() == Path(sys.executable).absolute():
                return None
            if nodeid is not None:
                raise RuntimeError(f"cross-pack test runtime differs from selected {pack}")
            result = subprocess.run(
                [
                    python,
                    "-I",
                    "-B",
                    str(Path(__file__).resolve()),
                    str(config_path),
                    "--nodeid",
                    pyfuncitem.nodeid,
                ],
                check=False,
            )
            if result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED:
                pytest.skip(f"selected cross-pack case did not execute in {pack}")
            result.check_returncode()
            return True

        @pytest.hookimpl(wrapper=True)
        def pytest_collection(self, session: pytest.Session) -> Generator[None, object, object]:
            with admission_timing(identity, "pytest-collection"):
                return (yield)

        def pytest_collection_finish(self, session: pytest.Session) -> None:
            for item in session.items:
                print(
                    json.dumps(
                        {"admissionTests": "collected", "pack": identity, "nodeid": item.nodeid}
                    ),
                    flush=True,
                )

        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_protocol(
            self, item: pytest.Item, nextitem: pytest.Item | None
        ) -> Generator[None, object, object]:
            global _running_test
            _running_test = True
            try:
                return (yield)
            finally:
                _running_test = False

        def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
            if report.when == "call" or report.failed or report.skipped:
                print(
                    json.dumps(
                        {
                            "admissionTests": "result",
                            "pack": identity,
                            "nodeid": report.nodeid,
                            "when": report.when,
                            "outcome": report.outcome,
                            "wasxfail": getattr(report, "wasxfail", None),
                            "elapsedSeconds": report.duration,
                        }
                    ),
                    flush=True,
                )
            if report.when == "call" and report.passed:
                self.passed.add(report.nodeid)
            if report.skipped:
                self.passed.discard(report.nodeid)
                print(f"Selected test skipped {report.nodeid}: {report.longrepr}", flush=True)

        def pytest_sessionfinish(self, session: pytest.Session) -> None:
            if not self.passed and (nodeid is None or session.exitstatus == pytest.ExitCode.OK):
                session.exitstatus = pytest.ExitCode.NO_TESTS_COLLECTED

    with admission_timing(identity, "pytest"):
        return int(
            pytest.main(
                # The inline plugin owns per-test unavailable-export state in this process.
                [
                    "-q",
                    "-n",
                    "0",
                    "--import-mode=importlib",
                    *([nodeid] if nodeid else config["tests"]),
                ],
                plugins=[RequireExecutedTests()],
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
