"""Test pack whose node takes a ``dinkster.asset`` INPUT - a job-referenced
asset, not a ``[[pack.assets]]`` declaration. Imported by the worker HOST
process (via a manifest entry), never by the test process's engine side
(hazard H5). The node reads the asset's bytes, so it fails loudly unless
the hosting process's store holds the digest - exactly the staging
contract under test."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_INT,
    CORE_STRING,
    AssetRef,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    register_asset_type,
    resolver_from_env,
)

STRING = TypeExpr.concrete(CORE_STRING)
INT = TypeExpr.concrete(CORE_INT)
ASSET = TypeExpr.concrete(ASSET_TYPE)


def register_types(registry: TypeRegistry) -> None:
    register_asset_type(registry, resolver_from_env())


class ReadAssetInput(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="apack.read",
            display_name="Read Asset Input",
            category="test",
            inputs=(InputSpec("data", ASSET),),
            outputs=(OutputSpec("text", STRING), OutputSpec("size", INT)),
        )

    @classmethod
    def execute(cls, *, data: AssetRef) -> Mapping[str, object]:
        return cls.outputs(text=data.read_bytes().decode("utf-8"), size=data.size)


NODES = (ReadAssetInput,)
