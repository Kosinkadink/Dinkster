from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    MODEL_TYPE,
    Deprecation,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    first_party_schema,
)

_MODEL = TypeExpr.concrete(MODEL_TYPE)


class wanBlockSwap(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="wanBlockSwap",
            description=(
                "Intercept wanBlockSwap custom node that causes major instability "
                "and make it no-op."
            ),
            inputs=(InputSpec("model", _MODEL),),
            outputs=(OutputSpec("MODEL", _MODEL),),
            deprecation=Deprecation(
                message="wanBlockSwap is a no-op because it conflicts with model memory management."
            ),
            search_visibility="deprecated",
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, model: Any) -> Mapping[str, object]:
        return cls.outputs(MODEL=model)
