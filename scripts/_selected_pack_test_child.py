"""Trusted engine test launcher for a disposable selected-pack environment."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


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
