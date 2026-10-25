"""Hunyuan 3D conditioning from single and ordered multi-view vision features."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_clip_vision_output,
)


class Hunyuan3Dv2Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="Hunyuan3Dv2Conditioning",
            category="model/conditioning/hunyuan 3d",
            inputs=(InputSpec("clip_vision_output", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)),),
            outputs=(
                OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, *, clip_vision_output: object) -> Mapping[str, object]:
        vision = materialize_clip_vision_output(clip_vision_output)
        torch = importlib.import_module("torch")
        embeds = vision.last_hidden_state
        positive: list[list[Any]] = [[embeds, {}]]
        negative: list[list[Any]] = [[torch.zeros_like(embeds), {}]]
        return cls.outputs(positive=positive, negative=negative)


class Hunyuan3Dv2ConditioningMultiView(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="Hunyuan3Dv2ConditioningMultiView",
            category="model/conditioning/hunyuan 3d",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE), required=False)
                for name in ("front", "left", "back", "right")
            ),
            outputs=(
                OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
        )

    @classmethod
    def execute(
        cls, *, front: object = None, left: object = None, back: object = None, right: object = None
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        out: list[Any] = []
        positions = None
        for index, value in enumerate((front, left, back, right)):
            if value is not None:
                vision = materialize_clip_vision_output(value)
                if positions is None:
                    mmdit = importlib.import_module(
                        "dinkster_inference.ldm.modules.diffusionmodules.mmdit"
                    )
                    positions = mmdit.get_1d_sincos_pos_embed_from_grid_torch(
                        vision.last_hidden_state.shape[-1], torch.arange(4)
                    )
                out.append(vision.last_hidden_state + positions[index].reshape(1, 1, -1))
        embeds = torch.cat(out, dim=1)
        positive: list[list[Any]] = [[embeds, {}]]
        negative: list[list[Any]] = [[torch.zeros_like(embeds), {}]]
        return cls.outputs(positive=positive, negative=negative)


HUNYUAN3D_CONDITIONING_NODES = (Hunyuan3Dv2Conditioning, Hunyuan3Dv2ConditioningMultiView)
