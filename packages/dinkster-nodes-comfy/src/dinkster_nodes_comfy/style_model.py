"""Style-model conditioning augmentation."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    STYLE_MODEL_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_clip_vision_output,
)


class StyleModelApply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="StyleModelApply",
            display_name="Apply Style Model",
            category="model/conditioning",
            search_terms=("style transfer",),
            inputs=(
                InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("style_model", TypeExpr.concrete(STYLE_MODEL_TYPE)),
                InputSpec("clip_vision_output", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.001),
                ),
                InputSpec(
                    "strength_type",
                    TypeExpr.concrete(CORE_COMBO),
                    default="multiply",
                    widget=ComboWidget(("multiply", "attn_bias")),
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        conditioning: Sequence[Sequence[Any]],
        style_model: Any,
        clip_vision_output: object,
        strength: float,
        strength_type: str,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        vision = materialize_clip_vision_output(clip_vision_output)
        cond = style_model.get_cond(vision).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if strength_type == "multiply":
            cond *= strength

        count = cond.shape[1]
        result: list[list[Any]] = []
        for text, metadata in conditioning:
            values = metadata.copy()
            if "attention_mask" in values or (strength_type == "attn_bias" and strength != 1.0):
                attention_bias = torch.log(
                    torch.Tensor([strength if strength_type == "attn_bias" else 1.0])
                )
                reference_shape = values.get("attention_mask_img_shape", (1, 1))
                reference_height, reference_width = cast(tuple[int, int], reference_shape)
                reference_count = reference_height * reference_width
                text_count = text.shape[1]
                mask = values.get("attention_mask")
                if mask is None:
                    mask = torch.zeros(
                        (text.shape[0], text_count + reference_count, text_count + reference_count),
                        dtype=torch.float16,
                    )
                if mask.dtype == torch.bool:
                    mask = torch.log(mask.to(dtype=torch.float16))
                new_mask = torch.zeros(
                    (
                        text.shape[0],
                        text_count + count + reference_count,
                        text_count + count + reference_count,
                    ),
                    dtype=torch.float16,
                )
                new_mask[:, :text_count, :text_count] = mask[:, :text_count, :text_count]
                new_mask[:, :text_count, text_count + count :] = mask[:, :text_count, text_count:]
                new_mask[:, text_count + count :, :text_count] = mask[:, text_count:, :text_count]
                new_mask[:, text_count + count :, text_count + count :] = mask[
                    :, text_count:, text_count:
                ]
                new_mask[:, :text_count, text_count : text_count + count] = attention_bias
                new_mask[:, text_count + count :, text_count : text_count + count] = attention_bias
                values["attention_mask"] = new_mask.to(text.device)
                values["attention_mask_img_shape"] = reference_shape
            result.append([torch.cat((text, cond), dim=1), values])
        return cls.outputs(CONDITIONING=result)


STYLE_MODEL_NODES = (StyleModelApply,)
