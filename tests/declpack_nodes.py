"""Test pack for worker-side declared-asset resolution. Imported by the
worker HOST process (via a manifest entry), never by the test process's
engine side (hazard H5). Goes THROUGH THE DOOR on purpose: the node uses
``dinkster_api.v1.declared_asset`` only, proving the v1 surface is sufficient
for the controlnet-aux-style "read my own [[pack.assets]] entry" story."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_INT,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    declared_asset,
)

STRING = TypeExpr.concrete(CORE_STRING)
INT = TypeExpr.concrete(CORE_INT)


class ReadDeclared(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="decl.read",
            display_name="Read Declared Asset",
            category="test",
            inputs=(InputSpec("asset_id", STRING),),
            outputs=(OutputSpec("text", STRING), OutputSpec("size", INT)),
        )

    @classmethod
    def execute(cls, *, asset_id: str) -> Mapping[str, object]:
        ref = declared_asset(asset_id)
        return cls.outputs(text=ref.read_bytes().decode("utf-8"), size=ref.size)


NODES = (ReadDeclared,)
