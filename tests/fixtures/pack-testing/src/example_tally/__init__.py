"""Small installable pack whose tally codec crosses the worker boundary."""

import json
import os
from collections import Counter
from collections.abc import Mapping

from dinkster_api.v1 import Input, Node, NodeSchema, Output, TypeExpr, TypeRegistry


class Shout(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="example-tally.shout",
            inputs=(Input("text", TypeExpr.concrete("STRING")),),
            outputs=(Output("text", TypeExpr.concrete("STRING")),),
        )

    @classmethod
    def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(text=text.upper())


class Tally(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="example-tally.tally",
            inputs=(Input("text", TypeExpr.concrete("STRING")),),
            outputs=(
                Output("counts", TypeExpr.concrete("example-tally.counts")),
                Output("pid", TypeExpr.concrete("INT")),
            ),
        )

    @classmethod
    def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(counts=dict(Counter(text.split())), pid=os.getpid())


def register_types(registry: TypeRegistry) -> None:
    registry.register(
        "example-tally.counts",
        encode=lambda value: json.dumps(value, sort_keys=True).encode(),
        decode=lambda data: json.loads(data),
    )


NODES = [Shout, Tally]
