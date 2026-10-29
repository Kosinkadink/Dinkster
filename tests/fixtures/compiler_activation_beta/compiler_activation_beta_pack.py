"""Genuine compiler activation fixture pack beta."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from dinkster_api.v1 import (
    CompilerEmission,
    GraphCompilerDescriptor,
    InferenceContribution,
)

NODES: tuple[object, ...] = ()


def _record(view: Any) -> None:
    trace = os.environ.get("DINKSTER_COMPILER_ACTIVATION_TRACE")
    if trace is not None:
        with Path(trace).open("a", encoding="ascii") as handle:
            handle.write(f"{view.pass_index}:{view.compiler_id}\n")


def _compile_a(view: Any) -> CompilerEmission:
    _record(view)
    control = os.environ.get("DINKSTER_COMPILER_ACTIVATION_CALLBACK_CONTROL")
    mode = Path(control).read_text(encoding="ascii").strip() if control is not None else "ok"
    if mode == "failure":
        raise RuntimeError("compiler-activation-beta-callback-failure")
    if mode == "cancel":
        marker = os.environ.get("DINKSTER_COMPILER_ACTIVATION_CANCEL_MARKER")
        if marker is not None:
            Path(marker).write_text("entered", encoding="ascii")
        while True:
            try:
                view.check_cancelled()
            except BaseException:
                if marker is not None:
                    Path(marker).write_text("cancelled", encoding="ascii")
                raise
            time.sleep(0.002)
    view.check_cancelled()
    if view.pass_index != 0:
        return CompilerEmission()
    generated = view.attempt_generated_node(
        "beta-a",
        ("source",),
        "std.math.add_ints",
        {"a": 17, "b": 19},
    )
    return CompilerEmission(generated=(generated,))


def _compile_b(view: Any) -> CompilerEmission:
    _record(view)
    view.check_cancelled()
    if view.pass_index != 0:
        return CompilerEmission()
    generated = view.attempt_generated_node(
        "beta-b",
        ("source",),
        "std.math.add_ints",
        {"a": 23, "b": 29},
    )
    return CompilerEmission(generated=(generated,))


def _descriptors() -> tuple[GraphCompilerDescriptor, ...]:
    declaration_mode = os.environ.get("DINKSTER_COMPILER_ACTIVATION_DECLARATION_MODE", "ok")
    first_id = (
        "compiler_activation_alpha.first"
        if declaration_mode == "collision"
        else "compiler_activation_beta.same_a"
    )
    first = GraphCompilerDescriptor(first_id, 10, _compile_a)
    second = GraphCompilerDescriptor("compiler_activation_beta.same_b", 10, _compile_b)
    if declaration_mode == "duplicate":
        return (first, first)
    return (first, second)


def register() -> InferenceContribution:
    return InferenceContribution(graph_compilers=_descriptors())
