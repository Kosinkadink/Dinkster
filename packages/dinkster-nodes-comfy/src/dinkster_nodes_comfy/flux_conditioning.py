"""Flux prompt streams and conditioning metadata transforms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_STRING,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

from .conditioning import conditioning_set_values


class CLIPTextEncodeFlux(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CLIPTextEncodeFlux",
            category="model/conditioning/flux",
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec(
                    "clip_l",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec(
                    "t5xxl",
                    TypeExpr.concrete(CORE_STRING),
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec(
                    "guidance",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=3.5,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )

    @classmethod
    def execute(
        cls, *, clip: Any, clip_l: str, t5xxl: str, guidance: float
    ) -> Mapping[str, object]:
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]
        return cls.outputs(
            CONDITIONING=clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance})
        )


class FluxGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="FluxGuidance",
            category="model/conditioning/flux",
            inputs=(
                InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec(
                    "guidance",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=3.5,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )

    @classmethod
    def execute(cls, *, conditioning: Any, guidance: float) -> Mapping[str, object]:
        return cls.outputs(
            CONDITIONING=conditioning_set_values(conditioning, {"guidance": guidance})
        )


class FluxDisableGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="FluxDisableGuidance",
            category="model/conditioning/flux",
            description=(
                "This node completely disables the guidance embed on Flux and Flux like models"
            ),
            inputs=(InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )

    @classmethod
    def execute(cls, *, conditioning: Any) -> Mapping[str, object]:
        return cls.outputs(CONDITIONING=conditioning_set_values(conditioning, {"guidance": None}))


class FluxKontextMultiReferenceLatentMethod(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="FluxKontextMultiReferenceLatentMethod",
            display_name="Edit Model Reference Method",
            category="model/conditioning/flux",
            inputs=(
                InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec(
                    "reference_latents_method",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=ComboWidget(
                        tuple(
                            ComboOption(name)
                            for name in (
                                "offset",
                                "index",
                                "uxo/uno",
                                "index_timestep_zero",
                                "index_timestep_zero_isolated",
                                "index_timestep_zero_text_start",
                            )
                        )
                    ),
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )

    @classmethod
    def execute(cls, *, conditioning: Any, reference_latents_method: str) -> Mapping[str, object]:
        if "uxo" in reference_latents_method or "uso" in reference_latents_method:
            reference_latents_method = "uxo"
        return cls.outputs(
            CONDITIONING=conditioning_set_values(
                conditioning, {"reference_latents_method": reference_latents_method}
            )
        )


FLUX_CONDITIONING_NODES = (
    CLIPTextEncodeFlux,
    FluxGuidance,
    FluxDisableGuidance,
    FluxKontextMultiReferenceLatentMethod,
)
