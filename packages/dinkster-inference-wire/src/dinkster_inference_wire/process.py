"""Lazy device-policy initialization for the process-owned inference runtime."""

from __future__ import annotations

import inspect
import os
import sys
import threading
from contextlib import nullcontext
from functools import wraps
from importlib import import_module
from typing import Any, cast

from dinkster_schema import Node

_initialization_lock = threading.Lock()


def inference_mode() -> Any:
    """Enter the caller's inference boundary without requiring Torch at import time."""
    try:
        torch = import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return nullcontext()
    mode = cast("dict[str, object]", vars(torch)).get("inference_mode")
    return mode() if callable(mode) else nullcontext()


def _inference_execute(method: Any) -> Any:
    @wraps(method)
    def execute(cls: type[Node], *args: Any, **inputs: Any) -> Any:
        with inference_mode():
            return method(cls, *args, **inputs)

    cast("Any", execute)._dinkster_inference_boundary = True
    return execute


def inference_nodes(nodes: tuple[type[Node], ...]) -> tuple[type[Node], ...]:
    """Scope synchronous inference bodies; async graph orchestration stays outside.

    Apply at a pack's node export. Torch mode is thread-local, so it must not
    span an await or encompass ordinary gradient-capable nodes in other packs.
    """
    for node in nodes:
        method = node.execute
        if inspect.iscoroutinefunction(method) or getattr(
            method, "_dinkster_inference_boundary", False
        ):
            continue
        cast("Any", node).execute = classmethod(_inference_execute(method.__func__))
    return nodes


def initialize_inference_process() -> None:
    """Select device policy before the runtime's first device probe.

    Existing initialized inference state belongs to its process and is never
    reconfigured. Importing this helper does not load Torch or inference.
    """
    with _initialization_lock:
        if sys.modules.get("dinkster_inference.model_management") is not None:
            return
        torch = import_module("torch")
        cli_args = import_module("dinkster_inference.cli_args")
        if os.environ.get("DINKSTER_ACCELERATOR", "auto") == "cpu" or not (
            torch.cuda.is_available()
            or torch.xpu.is_available()
            or torch.backends.mps.is_available()
        ):
            cli_args.args.cpu = True
        import_module("dinkster_inference.model_management")


__all__ = ["inference_mode", "inference_nodes", "initialize_inference_process"]
