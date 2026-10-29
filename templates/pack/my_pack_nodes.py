"""Template pack: a minimal, doctor-clean Dinkster pack.

Everything a standard pack needs, in one module: two nodes, a custom value
type with a declared codec and a rendition, and progress reporting. The
only import a pack ever needs is the frozen api door - dinkster doctor flags
anything else.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from dinkster_api.v1 import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    report_progress,
)


class Shout(Node):
    """Uppercase and repeat a string: the smallest possible node."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="my-pack.shout",
            display_name="Shout",
            category="text",
            description="Uppercase the text and repeat it.",
            inputs=(
                InputSpec("text", TypeExpr.concrete("core.string")),
                InputSpec(
                    "times",
                    TypeExpr.concrete("core.int"),
                    required=False,
                    default=1,
                ),
            ),
            outputs=(OutputSpec("shouted", TypeExpr.concrete("core.string")),),
        )

    @classmethod
    def execute(cls, *, text: str, times: int = 1) -> Mapping[str, object]:
        return cls.outputs(shouted=(text.upper() + "!") * times)


class Tally(Node):
    """Count words into a pack-owned value type (see register_types)."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="my-pack.tally",
            display_name="Word Tally",
            category="text",
            description="Count word occurrences into a my-pack.tally value.",
            inputs=(InputSpec("text", TypeExpr.concrete("core.string")),),
            outputs=(OutputSpec("tally", TypeExpr.concrete("my-pack.tally")),),
        )

    @classmethod
    def execute(cls, *, text: str) -> Mapping[str, object]:
        words = text.split()
        counts: dict[str, int] = {}
        for index, word in enumerate(words):
            counts[word] = counts.get(word, 0) + 1
            # Progress units are whatever the node counts.
            report_progress(index + 1, len(words))
        return cls.outputs(tally=counts)


def register_types(registry: TypeRegistry) -> None:
    """Register pack-owned value types.

    The declared codec is what lets values cross process/machine boundaries
    and cache to disk; without it, a pickle fallback applies and doctor
    warns. The rendition is how frontends preview the value (the first
    registered rendition is the type's default)."""
    registry.register(
        "my-pack.tally",
        encode=lambda obj: json.dumps(obj, sort_keys=True).encode(),
        decode=lambda data: json.loads(data),
    )
    registry.register_rendition(
        "my-pack.tally",
        "text",
        mime="text/plain",
        render=lambda obj: json.dumps(obj, indent=2, sort_keys=True).encode(),
    )


NODES = [Shout, Tally]
