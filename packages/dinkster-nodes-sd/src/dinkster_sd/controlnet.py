"""Native ControlNet conditioning and union-type selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CONTROL_NET_TYPE,
    CORE_FLOAT,
    IMAGE_TYPE,
    VAE_TYPE,
    Deprecation,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    with_identity_replacement,
)
from dinkster_conditioning.controlnet import ControlNetApplyAdvanced

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)

_CONTROL_NET = TypeExpr.concrete(CONTROL_NET_TYPE)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)

_VAE = TypeExpr.concrete(VAE_TYPE)

_CATEGORY = "model/conditioning/controlnet"


def _strength() -> InputSpec:
    return InputSpec(
        "strength",
        TypeExpr.concrete(CORE_FLOAT),
        default=1.0,
        widget=NumberWidget(min=0.0, max=10.0, step=0.01),
    )


def _range(*, advanced: bool = False) -> tuple[InputSpec, ...]:
    return tuple(
        InputSpec(
            name,
            TypeExpr.concrete(CORE_FLOAT),
            default=default,
            widget=NumberWidget(min=0.0, max=1.0, step=0.001),
            advanced=advanced,
        )
        for name, default in (("start_percent", 0.0), ("end_percent", 1.0))
    )


class ControlNetApplySD3(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.sd.control_net_apply_sd3",
            display_name="Apply Controlnet with VAE",
            category=_CATEGORY,
            deprecation=Deprecation(message="Use ControlNetApplyAdvanced with a VAE instead."),
            search_visibility="deprecated",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("control_net", _CONTROL_NET),
                InputSpec("vae", _VAE),
                InputSpec("image", _IMAGE),
                _strength(),
                *_range(),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
            ),
        )
        return with_identity_replacement(schema, "ControlNetApplySD3")

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        control_net: Any,
        image: object,
        strength: float,
        start_percent: float,
        end_percent: float,
        vae: Any = None,
    ) -> Mapping[str, object]:
        return ControlNetApplyAdvanced.execute(
            positive=positive,
            negative=negative,
            control_net=control_net,
            image=image,
            strength=strength,
            start_percent=start_percent,
            end_percent=end_percent,
            vae=vae,
        )
