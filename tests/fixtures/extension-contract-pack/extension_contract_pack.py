"""Ordinary third-party pack fixture for the extension contract."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    InputSpec,
    JsonField,
    JsonObjectSchema,
    Node,
    NodeSchema,
    OutputSpec,
    PackEvent,
    TypeExpr,
    report_pack_event,
)

EVENT = PackEvent(
    "fixture.extension-contract.executed",
    JsonObjectSchema(
        (
            JsonField("height", "integer"),
            JsonField("mean", "number"),
            JsonField("width", "integer"),
        )
    ),
)


def route(_request: Mapping[str, object]) -> dict[str, object]:
    return {"message": "Third-party pack route ready"}


class ExtensionContractProof(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="fixture.extension.contract",
            display_name="Extension Contract Proof",
            inputs=(
                InputSpec("width", TypeExpr.concrete("core.int"), default=13),
                InputSpec("height", TypeExpr.concrete("core.int"), default=7),
            ),
            outputs=(OutputSpec("mean", TypeExpr.concrete("core.float")),),
        )

    @classmethod
    def execute(cls, *, width: int, height: int) -> Mapping[str, object]:
        mean = 0.5
        report_pack_event(EVENT, {"height": height, "mean": mean, "width": width})
        return cls.outputs(mean=mean)


NODES = (ExtensionContractProof,)
