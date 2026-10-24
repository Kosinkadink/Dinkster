"""Video CFG guidance patches from ComfyUI video model nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_MODEL = TypeExpr.concrete(MODEL_TYPE)


def _schema(node_type: str, display_name: str) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category="model/sampling/guiders",
        inputs=(
            InputSpec("model", _MODEL),
            InputSpec(
                "min_cfg",
                _FLOAT,
                default=1.0,
                advanced=True,
                widget=NumberWidget(min=0.0, max=100.0, step=0.5, round=0.01),
            ),
        ),
        outputs=(OutputSpec("MODEL", _MODEL),),
    )


class VideoLinearCFGGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("VideoLinearCFGGuidance", "Video Linear CFG Guidance")

    @classmethod
    def execute(cls, *, model: Any, min_cfg: float) -> Mapping[str, object]:
        def linear_cfg(args: dict[str, Any]) -> Any:
            torch = importlib.import_module("torch")
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            scale = torch.linspace(min_cfg, cond_scale, cond.shape[0], device=cond.device).reshape(
                (cond.shape[0], 1, 1, 1)
            )
            return uncond + scale * (cond - uncond)

        patched = model.clone()
        patched.set_model_sampler_cfg_function(linear_cfg)
        return cls.outputs(MODEL=patched)


class VideoTriangleCFGGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("VideoTriangleCFGGuidance", "Video Triangle CFG Guidance")

    @classmethod
    def execute(cls, *, model: Any, min_cfg: float) -> Mapping[str, object]:
        def triangle_cfg(args: dict[str, Any]) -> Any:
            torch = importlib.import_module("torch")
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            period = 1.0
            values = torch.linspace(0, 1, cond.shape[0], device=cond.device)
            values = 2 * (values / period - torch.floor(values / period + 0.5)).abs()
            scale = (values * (cond_scale - min_cfg) + min_cfg).reshape((cond.shape[0], 1, 1, 1))
            return uncond + scale * (cond - uncond)

        patched = model.clone()
        patched.set_model_sampler_cfg_function(triangle_cfg)
        return cls.outputs(MODEL=patched)


VIDEO_GUIDANCE_NODES = (VideoLinearCFGGuidance, VideoTriangleCFGGuidance)
