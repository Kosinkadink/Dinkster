"""Cooperative folder reading across engines on the same account and computer."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from filelock import FileLock, Timeout

from ._settings import model_preferences


def _directory() -> Path:
    home = Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster")).expanduser()
    directory = home / "executing-workflows"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextmanager
def executing_workflow() -> Generator[None]:
    directory = _directory()
    path = directory / f"{uuid.uuid4().hex}.lock"
    marker = FileLock(path, preserve_lock_file=True)
    gate = FileLock(directory / "gate.lock", preserve_lock_file=True)
    with gate:
        marker.acquire()
    try:
        yield
    finally:
        with gate:
            marker.release()
            path.unlink(missing_ok=True)


def workflow_executing() -> bool:
    directory = _directory()
    with FileLock(directory / "gate.lock", preserve_lock_file=True):
        for path in directory.glob("*.lock"):
            if path.name == "gate.lock":
                continue
            marker = FileLock(path, preserve_lock_file=True)
            try:
                marker.acquire(timeout=0)
            except Timeout:
                return True
            else:
                marker.release()
                # The gate prevents observing a new marker before it is locked.
                path.unlink(missing_ok=True)
    return False


def wait_for_workflows(report: Callable[[bool], None]) -> None:
    waited = False
    while workflow_executing():
        home = _directory().parent
        legacy = home / "model-settings.json"
        if legacy.exists():
            values = cast("dict[str, object]", json.loads(legacy.read_text("utf-8")))
        else:
            record = cast(
                "dict[str, object]", model_preferences("folders.models.get", {})["folders"]
            )
            values = cast("dict[str, object]", record["values"])
        if values.get("readFoldersWhileRunning", False) is True:
            break
        report(True)
        waited = True
        time.sleep(0.1)
    if waited:
        report(False)
