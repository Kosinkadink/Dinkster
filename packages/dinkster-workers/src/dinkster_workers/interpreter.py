"""Validation for interpreters selected to run Dinkster child processes."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import cast

_MINIMUM_VERSION = (3, 12)
_PREFLIGHT_TIMEOUT = 5.0
_INTERPRETER_DISPLAY_LIMIT = 200
_VERSION_SCRIPT = "import sys;print('[%d,%d]' % sys.version_info[:2])"
_WINDOWS_MODULE_BOOTSTRAP = r"""
import os
import sys

for index, path in enumerate(sys.path):
    absolute = os.path.abspath(path)
    if not absolute.startswith("\\\\?\\"):
        if absolute.startswith("\\\\"):
            absolute = "\\\\?\\UNC\\" + absolute[2:]
        else:
            absolute = "\\\\?\\" + absolute
    sys.path[index] = absolute

import runpy

sys.argv = sys.argv[1:]
runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
"""


class InterpreterPreflightError(Exception):
    """The selected child interpreter cannot satisfy Dinkster's Python contract."""


def module_command(command: Sequence[str]) -> list[str]:
    """Extend Windows module import paths without changing interpreter identity."""
    if (
        os.name == "nt"
        and tuple(command[1:2]) == ("-m",)
        and Path(command[0]).suffix.lower() not in {".bat", ".cmd"}
    ):
        # Leave executable lookup and sys.prefix to Python and CreateProcess.
        return [command[0], "-c", _WINDOWS_MODULE_BOOTSTRAP, *command[2:]]
    return list(command)


def preflight_interpreter(interpreter: Path | str) -> tuple[int, int]:
    """Require a runnable Python >=3.12 without importing the child target."""
    selected = str(interpreter)
    selected_display = repr(selected)
    if len(selected_display) > _INTERPRETER_DISPLAY_LIMIT:
        selected_display = f"{selected_display[: _INTERPRETER_DISPLAY_LIMIT - 3]}..."
    # Version admission must not execute the environment's .pth or sitecustomize hooks.
    command = [selected, "-I", "-S", "-c", _VERSION_SCRIPT]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PREFLIGHT_TIMEOUT,
            check=False,
            shell=False,
        )
    except OSError as exc:
        raise InterpreterPreflightError(
            f"selected interpreter {selected_display} could not start; "
            "configure an executable Python >=3.12"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise InterpreterPreflightError(
            f"selected interpreter {selected_display} version check timed out after "
            f"{_PREFLIGHT_TIMEOUT:.0f}s; configure a responsive Python >=3.12"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr.strip() or completed.stdout.strip()).replace("\n", " ")
        detail = detail[:400] or "no diagnostic output"
        raise InterpreterPreflightError(
            f"selected interpreter {selected_display} version check exited "
            f"{completed.returncode}: {detail}; configure an executable Python >=3.12"
        )
    try:
        payload: object = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise InterpreterPreflightError(
            f"selected interpreter {selected_display} returned malformed version output; "
            "configure an executable Python >=3.12"
        ) from exc
    if not isinstance(payload, list):
        raise InterpreterPreflightError(
            f"selected interpreter {selected_display} returned malformed version output; "
            "configure an executable Python >=3.12"
        )
    parts = cast("list[object]", payload)
    if len(parts) != 2 or any(
        not isinstance(part, int) or isinstance(part, bool) for part in parts
    ):
        raise InterpreterPreflightError(
            f"selected interpreter {selected_display} returned malformed version output; "
            "configure an executable Python >=3.12"
        )
    major, minor = parts
    assert isinstance(major, int) and isinstance(minor, int)
    version = (major, minor)
    if version < _MINIMUM_VERSION:
        raise InterpreterPreflightError(
            f"selected interpreter {selected_display} is Python {version[0]}.{version[1]}; "
            "Dinkster requires Python >=3.12"
        )
    return version
