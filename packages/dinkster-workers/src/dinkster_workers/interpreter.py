"""Validation for interpreters selected to run Dinkster child processes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

_MINIMUM_VERSION = (3, 12)
_PREFLIGHT_TIMEOUT = 5.0
_INTERPRETER_DISPLAY_LIMIT = 200
_VERSION_SCRIPT = "import sys;print('[%d,%d]' % sys.version_info[:2])"


class InterpreterPreflightError(Exception):
    """The selected child interpreter cannot satisfy Dinkster's Python contract."""


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
