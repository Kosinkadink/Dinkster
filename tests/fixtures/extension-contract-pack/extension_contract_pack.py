"""Ordinary third-party pack fixture for the extension contract."""

from __future__ import annotations

import json
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
    TypeRegistry,
    report_pack_event,
)

VALUE_TYPE = "fixture.extension.value"

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


def register_types(registry: TypeRegistry) -> None:
    registry.register(
        VALUE_TYPE,
        encode=lambda value: json.dumps(value).encode(),
        decode=lambda data: json.loads(data),
    )


class ExtensionContractValue(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="fixture.extension.value",
            display_name="Extension Contract Value",
            inputs=(
                InputSpec("width", TypeExpr.concrete("core.int"), default=13),
                InputSpec("height", TypeExpr.concrete("core.int"), default=7),
            ),
            outputs=(OutputSpec("sample", TypeExpr.concrete(VALUE_TYPE)),),
        )

    @classmethod
    def execute(cls, *, width: int, height: int) -> Mapping[str, object]:
        return cls.outputs(
            sample={
                "width": width,
                "height": height,
                "mean": (width + height) / 40,
            }
        )


class ExtensionContractProof(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="fixture.extension.contract",
            display_name="Extension Contract Proof",
            inputs=(InputSpec("sample", TypeExpr.concrete(VALUE_TYPE)),),
            outputs=(OutputSpec("mean", TypeExpr.concrete("core.float")),),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, sample: Mapping[str, int | float]) -> Mapping[str, object]:
        width = int(sample["width"])
        height = int(sample["height"])
        mean = float(sample["mean"])
        report_pack_event(EVENT, {"height": height, "mean": mean, "width": width})
        return cls.outputs(mean=mean)


NODES = (ExtensionContractValue, ExtensionContractProof)
