from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

import pytest
from dinkster_workers import interpreter
from dinkster_workers.interpreter import InterpreterPreflightError, preflight_interpreter


def test_preflight_accepts_current_supported_interpreter() -> None:
    assert preflight_interpreter(sys.executable) == sys.version_info[:2]


def test_version_script_requires_no_additional_imports() -> None:
    command = (
        "import sys\n"
        "def reject_import(event, args):\n"
        "    if event == 'import':\n"
        "        raise RuntimeError('version probe imported ' + args[0])\n"
        "sys.addaudithook(reject_import)\n" + interpreter._VERSION_SCRIPT
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", command],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    assert result.stdout.strip() == f"[{sys.version_info.major},{sys.version_info.minor}]"


@pytest.mark.parametrize("hook_name", ["startup.pth", "sitecustomize.py"])
def test_preflight_does_not_execute_site_hooks(tmp_path: Path, hook_name: str) -> None:
    root = tmp_path / "selected"
    venv.EnvBuilder(with_pip=False, symlinks=sys.platform != "win32").create(root)
    python = root / "Scripts" / "python.exe" if sys.platform == "win32" else root / "bin" / "python"
    site = subprocess.run(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    site_packages = Path(site.stdout.strip())
    (site_packages / hook_name).write_text("import os; os._exit(61)\n")
    if hook_name == "sitecustomize.py":
        # Some distributions ship a standard-library sitecustomize ahead of the venv.
        (site_packages / "startup.pth").write_text(
            f"import sys; sys.path.insert(0, {str(site_packages)!r})\n"
        )

    assert preflight_interpreter(python) == sys.version_info[:2]
    normal_start = subprocess.run(
        [str(python), "-I", "-c", "pass"], capture_output=True, timeout=30, check=False
    )
    assert normal_start.returncode == 61


@pytest.mark.parametrize("version", [(3, 12), (3, 13), (3, 14)])
def test_preflight_accepts_supported_version(
    version: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, f"[{version[0]},{version[1]}]\n", "")

    monkeypatch.setattr("dinkster_workers.interpreter.subprocess.run", run)
    assert preflight_interpreter("/selected/python") == version
    command, kwargs = seen[0]
    assert command[:4] == ["/selected/python", "-I", "-S", "-c"]
    assert "sys.version_info" in command[4]
    assert kwargs == {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 5.0,
        "check": False,
        "shell": False,
    }


@pytest.mark.parametrize(
    ("result", "match"),
    [
        (subprocess.CompletedProcess([], 0, "[3,11]\n", ""), "Python 3.11"),
        (subprocess.CompletedProcess([], 7, "", "broken"), "exited 7"),
        (subprocess.CompletedProcess([], 0, "not-json", ""), "malformed"),
        (subprocess.CompletedProcess([], 0, "\N{REPLACEMENT CHARACTER}", ""), "malformed"),
        (subprocess.CompletedProcess([], 0, "[3,12,1]", ""), "malformed"),
        (subprocess.CompletedProcess([], 0, "[true,12]", ""), "malformed"),
    ],
)
def test_preflight_fails_closed_for_bad_result(
    result: subprocess.CompletedProcess[str],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dinkster_workers.interpreter.subprocess.run", lambda *_args, **_kwargs: result
    )
    with pytest.raises(InterpreterPreflightError, match=match):
        preflight_interpreter("/selected/python")


def test_preflight_failures_are_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dinkster_workers.interpreter.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "failure"),
    )
    with pytest.raises(InterpreterPreflightError, match=r"configure an executable Python >=3\.12"):
        preflight_interpreter("/selected/python")


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        (OSError("secret path details"), "could not start"),
        (subprocess.TimeoutExpired(["python"], 5), "timed out"),
    ],
)
def test_preflight_fails_closed_when_probe_does_not_complete(
    failure: BaseException, match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("dinkster_workers.interpreter.subprocess.run", fail)
    with pytest.raises(InterpreterPreflightError, match=match) as caught:
        preflight_interpreter(Path("/selected/python"))
    assert "secret path details" not in str(caught.value)


def test_preflight_bounds_selected_interpreter_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dinkster_workers.interpreter.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "x" * 1_000),
    )
    with pytest.raises(InterpreterPreflightError) as caught:
        preflight_interpreter("/" + "selected" * 1_000)
    assert len(str(caught.value)) < 700
