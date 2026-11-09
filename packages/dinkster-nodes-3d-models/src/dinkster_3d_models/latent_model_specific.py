"""Empty latent layouts for image and 3D model families."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_INT,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    first_party_schema,
)


def _integer(
    name: str, default: int, maximum: int, minimum: int = 1, step: int | None = None
) -> InputSpec:
    return InputSpec(
        name,
        TypeExpr.concrete(CORE_INT),
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _schema(
    name: str,
    category: str,
    inputs: tuple[InputSpec, ...],
    display: str | None = None,
    description: str = "",
) -> NodeSchema:
    return NodeSchema(
        node_type=name,
        display_name=display or name,
        category=category,
        description=description,
        inputs=inputs,
        outputs=(OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),),
    )


class EmptyTrellis2LatentStructure(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = _schema(
            "EmptyTrellis2LatentStructure",
            "model/latent/trellis",
            (_integer("batch_size", 1, 4096),),
            "Empty Trellis2 Latent Structure",
        )
        return first_party_schema(schema)

    @classmethod
    def execute(cls, *, batch_size: int) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        samples = torch.zeros(batch_size, 32, 16, 16, 16)
        return cls.outputs(LATENT={"samples": samples, "type": "trellis2"})
