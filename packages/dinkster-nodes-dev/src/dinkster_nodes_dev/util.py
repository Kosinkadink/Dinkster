"""Scheduling-observability scaffolding: nodes whose only job is to make
engine behavior visible in demos and tests."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)

FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)


class Delay(Node):
    """Passthrough with an async delay. Exists to make scheduling observable:
    demos and tests use it to prove independent nodes overlap (hazard H12).
    Also the canonical example of an async execute() - the worker awaits it,
    so a delay never blocks the engine or sibling nodes."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.util.delay",
            display_name="Delay",
            category="dev/util",
            inputs=(
                InputSpec("value", STRING),
                InputSpec("seconds", FLOAT, default=0.1),
            ),
            outputs=(OutputSpec("value", STRING),),
            # A delay waits, it doesn't compute: exempt from the default
            # compute lane, like partner/API nodes awaiting a network call.
            io_bound=True,
        )

    @classmethod
    async def execute(cls, *, value: str, seconds: float) -> Mapping[str, object]:
        await asyncio.sleep(seconds)
        return cls.outputs(value=value)
