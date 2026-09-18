"""Deterministic nodes for the backend conformance recorder.

These nodes are composed only through ``DEV_NODES`` under explicit dev
composition. They expose orthogonal execution behavior without counters or
other process-global proof state; the recorder derives every claim from real
engine results and events.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import numpy as np
from dinkster_api.v1 import (
    AbsentOutput,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    report_preview,
    report_progress,
)

from .image import DEV_IMAGE

INT = TypeExpr.concrete("core.int")
BOOLEAN = TypeExpr.concrete("core.boolean")
IMAGE = TypeExpr.concrete(DEV_IMAGE)


class ConformanceSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.conformance.source",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        return cls.outputs(value=value)


class ConformanceSplit(Node):
    """One invocation produces two independently inspectable outputs."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.conformance.split",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("left", INT), OutputSpec("right", INT)),
        )

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        return cls.outputs(left=value, right=value + 1)


class ConformanceAdd(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.conformance.add",
            inputs=(InputSpec("value", INT), InputSpec("amount", INT, default=1)),
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def execute(cls, *, value: int, amount: int) -> Mapping[str, object]:
        return cls.outputs(value=value + amount)


class ConformanceEffect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.conformance.effect",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("value", INT),),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        return cls.outputs(value=value)


class ConformanceMaybe(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.conformance.maybe",
            inputs=(InputSpec("present", BOOLEAN),),
            outputs=(OutputSpec("value", INT, optional=True),),
        )

    @classmethod
    def execute(cls, *, present: bool) -> Mapping[str, object]:
        return cls.outputs(value=7 if present else AbsentOutput("proof value omitted"))


class ConformanceOmit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.conformance.omit",
            inputs=(
                InputSpec("base", INT),
                InputSpec("optional", INT, required=False),
            ),
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def execute(cls, *, base: int, optional: int | None = None) -> Mapping[str, object]:
        return cls.outputs(value=base if optional is None else base + optional)


class ConformanceFailAbsent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.conformance.fail-absent",
            inputs=(InputSpec("value", INT, on_absent="fail"),),
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def execute(cls, *, value: int) -> Mapping[str, object]:
        return cls.outputs(value=value)


class ConformancePreview(Node):
    """Ephemeral runtime frames plus distinct final preview candidates."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.conformance.preview",
            inputs=(InputSpec("seed", INT),),
            outputs=(
                OutputSpec("image", IMAGE, preview=True),
                OutputSpec("fallback", INT, preview=True),
            ),
            emits_previews=True,
        )

    @classmethod
    def execute(cls, *, seed: int) -> Mapping[str, object]:
        report_progress(1, 2, text="preview")
        report_preview(b"runtime-frame-1", mime="image/png", width=2, height=2)
        report_preview(b"runtime-frame-2", mime="image/webp", width=2, height=2)
        report_progress(2, 2, text="final")
        image = np.full((2, 2), float(seed % 256) / 255.0, dtype=np.float32)
        return cls.outputs(image=image, fallback=seed)


class ConformanceFailure(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.conformance.failure",
            outputs=(OutputSpec("value", INT),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        raise RuntimeError("stable conformance failure")


class ConformanceCancellable(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.conformance.cancellable",
            outputs=(OutputSpec("value", INT),),
            occupies=("conformance-resource",),
        )

    @classmethod
    async def execute(cls) -> Mapping[str, object]:
        report_progress(1, 2, text="waiting")
        await asyncio.Event().wait()
        return cls.outputs(value=1)


CONFORMANCE_NODES = (
    ConformanceSource,
    ConformanceSplit,
    ConformanceAdd,
    ConformanceEffect,
    ConformanceMaybe,
    ConformanceOmit,
    ConformanceFailAbsent,
    ConformancePreview,
    ConformanceFailure,
    ConformanceCancellable,
)
