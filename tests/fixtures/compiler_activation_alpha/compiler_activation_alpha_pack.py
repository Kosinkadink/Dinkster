"""Genuine compiler activation fixture pack alpha."""

from __future__ import annotations

import os
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


def _compile(view: Any) -> CompilerEmission:
    _record(view)
    view.check_cancelled()
    if view.pass_index != 0:
        return CompilerEmission()
    generated = view.attempt_generated_node(
        "alpha",
        ("source",),
        "std.math.add_ints",
        {"a": 11, "b": 13},
    )
    return CompilerEmission(generated=(generated,))


COMPILER = GraphCompilerDescriptor("compiler_activation_alpha.first", 10, _compile)


def register() -> InferenceContribution:
    return InferenceContribution(graph_compilers=(COMPILER,))
