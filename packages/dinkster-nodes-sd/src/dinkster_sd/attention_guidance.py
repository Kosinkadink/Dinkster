"""Attention-guidance model patches executed by the inference sampler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_STRING,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    with_identity_replacement,
)
from dinkster_sampling.attention_guidance import SkipLayerGuidanceDiT

_FLOAT = TypeExpr.concrete(CORE_FLOAT)

_MODEL = TypeExpr.concrete(MODEL_TYPE)

_STRING = TypeExpr.concrete(CORE_STRING)


class SkipLayerGuidanceSD3(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.sd.skip_layer_guidance_sd3",
            category="advanced/guidance",
            description=(
                "Generic version of SkipLayerGuidance node that can be used on every DiT model."
            ),
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "layers",
                    _STRING,
                    default="7, 8, 9",
                    advanced=True,
                    widget=StringWidget(multiline=False),
                ),
                InputSpec(
                    "scale", _FLOAT, default=3.0, widget=NumberWidget(min=0.0, max=10.0, step=0.1)
                ),
                InputSpec(
                    "start_percent",
                    _FLOAT,
                    default=0.01,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    _FLOAT,
                    default=0.15,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )
        return with_identity_replacement(schema, "SkipLayerGuidanceSD3")

    @classmethod
    def execute(
        cls, *, model: Any, layers: str, scale: float, start_percent: float, end_percent: float
    ) -> Mapping[str, object]:
        return SkipLayerGuidanceDiT.execute(
            model=model,
            double_layers=layers,
            single_layers="",
            scale=scale,
            start_percent=start_percent,
            end_percent=end_percent,
            rescaling_scale=0.0,
        )
