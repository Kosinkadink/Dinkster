"""Lazy device-policy initialization for the process-owned inference runtime."""

from __future__ import annotations

import importlib
import os
import sys
import threading

_initialization_lock = threading.Lock()


def initialize_inference_process() -> None:
    """Select device policy before the runtime's first device probe.

    Existing initialized inference state belongs to its process and is never
    reconfigured. Importing this helper does not load Torch or inference.
    """
    with _initialization_lock:
        if sys.modules.get("dinkster_inference.model_management") is not None:
            return
        torch = importlib.import_module("torch")
        cli_args = importlib.import_module("dinkster_inference.cli_args")
        if os.environ.get("DINKSTER_ACCELERATOR", "auto") == "cpu" or not (
            torch.cuda.is_available()
            or torch.xpu.is_available()
            or torch.backends.mps.is_available()
        ):
            cli_args.args.cpu = True
        importlib.import_module("dinkster_inference.model_management")


__all__ = ["initialize_inference_process"]
