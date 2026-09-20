"""Third-party extension contract proof carried by an ordinary fixture pack."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
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

from .image import DEV_IMAGE, GradientImage

CONTRACT_EVENT_NAME = "dev.extension-contract.executed"
CONTRACT_EVENT = PackEvent(
    CONTRACT_EVENT_NAME,
    JsonObjectSchema(
        (
            JsonField("height", "integer"),
            JsonField("mean", "number"),
            JsonField("width", "integer"),
        )
    ),
)


def extension_contract_policy(_request: Mapping[str, object]) -> dict[str, object]:
    return {"message": "Third-party pack route ready"}


class ExtensionContractProof(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dev.extension.contract",
            display_name="Extension Contract Proof",
            inputs=(InputSpec("image", TypeExpr.concrete(DEV_IMAGE)),),
            outputs=(OutputSpec("mean", TypeExpr.concrete("core.float")),),
        )

    @classmethod
    def execute(cls, *, image: np.ndarray) -> Mapping[str, object]:
        height, width = image.shape
        mean = float(image.mean())
        report_pack_event(
            CONTRACT_EVENT,
            {"height": int(height), "mean": mean, "width": int(width)},
        )
        return cls.outputs(mean=mean)


EXTENSION_CONTRACT_NODES = (GradientImage, ExtensionContractProof)
