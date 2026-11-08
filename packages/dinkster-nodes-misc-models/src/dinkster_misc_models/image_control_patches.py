"""Qwen, Z-Image, and USO image-control model patches."""

from __future__ import annotations

from dinkster_api.v1 import (
    CORE_FLOAT,
    IMAGE_TYPE,
    MASK_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    InputSpec,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    first_party_schema,
)
from dinkster_qwen.image_control_patches import QwenImageDiffsynthControlnet

_MODEL = TypeExpr.concrete(MODEL_TYPE)

_MODEL_PATCH = TypeExpr.concrete(MODEL_PATCH_TYPE)

_VAE = TypeExpr.concrete(VAE_TYPE)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)

_MASK = TypeExpr.concrete(MASK_TYPE)

_FLOAT = TypeExpr.concrete(CORE_FLOAT)


def _range() -> tuple[InputSpec, InputSpec]:
    return (
        InputSpec(
            "start_percent",
            _FLOAT,
            default=0.0,
            required=False,
            advanced=True,
            widget=NumberWidget(min=0.0, max=1.0, step=0.001),
        ),
        InputSpec(
            "end_percent",
            _FLOAT,
            default=1.0,
            required=False,
            advanced=True,
            widget=NumberWidget(min=0.0, max=1.0, step=0.001),
        ),
    )


class ZImageFunControlnet(QwenImageDiffsynthControlnet):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="ZImageFunControlnet",
            display_name="Apply Fun ControlNet",
            category="model/patch",
            search_terms=(
                "z-image controlnet",
                "qwen image 2.1 controlnet",
                "controlnet union",
                "fun controlnet",
            ),
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec("model_patch", _MODEL_PATCH),
                InputSpec("vae", _VAE),
                InputSpec(
                    "strength",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
                InputSpec("image", _IMAGE, required=False),
                InputSpec("inpaint_image", _IMAGE, required=False),
                InputSpec("mask", _MASK, required=False),
                *_range(),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )
        return first_party_schema(schema)
