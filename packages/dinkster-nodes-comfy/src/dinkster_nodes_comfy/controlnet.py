"""Native ControlNet conditioning and union-type selection."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CONTROL_NET_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    IMAGE_TYPE,
    MASK_TYPE,
    VAE_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
)

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


class ControlNetApply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ControlNetApply",
            display_name="Apply ControlNet (DEPRECATED)",
            category=_CATEGORY,
            inputs=(
                InputSpec("conditioning", _CONDITIONING),
                InputSpec("control_net", _CONTROL_NET),
                InputSpec("image", _IMAGE),
                _strength(),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        conditioning: Sequence[Sequence[Any]],
        control_net: Any,
        image: object,
        strength: float = 1.0,
    ) -> Mapping[str, object]:
        if strength == 0:
            return cls.outputs(CONDITIONING=conditioning)
        hint: Any = materialize_image(image)
        hint = hint.movedim(-1, 1)
        result: list[list[Any]] = []
        for text, metadata in conditioning:
            control = control_net.copy().set_cond_hint(hint, strength)
            if "control" in metadata:
                control.set_previous_controlnet(metadata["control"])
            result.append([text, {**metadata, "control": control, "control_apply_to_uncond": True}])
        return cls.outputs(CONDITIONING=result)


class ControlNetApplyAdvanced(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ControlNetApplyAdvanced",
            display_name="Apply ControlNet",
            category=_CATEGORY,
            search_terms=("controlnet", "apply controlnet", "use controlnet", "control net"),
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("control_net", _CONTROL_NET),
                InputSpec("image", _IMAGE),
                _strength(),
                *_range(),
                InputSpec("vae", _VAE, required=False),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        control_net: Any,
        image: object,
        strength: float = 1.0,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        vae: Any = None,
        extra_concat: list[Any] | None = None,
    ) -> Mapping[str, object]:
        if strength == 0:
            return cls.outputs(positive=positive, negative=negative)
        hint: Any = materialize_image(image)
        hint = hint.movedim(-1, 1)
        controls: dict[Any, Any] = {}
        outputs: list[list[list[Any]]] = []
        for conditioning in (positive, negative):
            result: list[list[Any]] = []
            for text, metadata in conditioning:
                previous = metadata.get("control")
                if previous in controls:
                    control = controls[previous]
                else:
                    control = control_net.copy().set_cond_hint(
                        hint,
                        strength,
                        (start_percent, end_percent),
                        vae=vae,
                        extra_concat=[] if extra_concat is None else extra_concat,
                    )
                    control.set_previous_controlnet(previous)
                    controls[previous] = control
                result.append(
                    [text, {**metadata, "control": control, "control_apply_to_uncond": False}]
                )
            outputs.append(result)
        return cls.outputs(positive=outputs[0], negative=outputs[1])


class SetUnionControlNetType(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        types = importlib.import_module("dinkster_inference.cldm.control_types")
        return NodeSchema(
            node_type="SetUnionControlNetType",
            display_name="Set Union ControlNet Type",
            category=_CATEGORY,
            search_terms=("set controlnet type", "union controlnet type"),
            inputs=(
                InputSpec("control_net", _CONTROL_NET),
                InputSpec(
                    "type",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=ComboWidget(
                        tuple(ComboOption(name) for name in ("auto", *types.UNION_CONTROLNET_TYPES))
                    ),
                ),
            ),
            outputs=(OutputSpec("CONTROL_NET", _CONTROL_NET),),
        )

    @classmethod
    def execute(cls, *, control_net: Any, type: str) -> Mapping[str, object]:
        types = importlib.import_module("dinkster_inference.cldm.control_types")
        result = control_net.copy()
        number = types.UNION_CONTROLNET_TYPES.get(type, -1)
        result.set_extra_arg("control_type", [number] if number >= 0 else [])
        return cls.outputs(CONTROL_NET=result)


class ControlNetInpaintingAliMamaApply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ControlNetInpaintingAliMamaApply",
            display_name="Apply ControlNet Inpainting (AliMama)",
            category=_CATEGORY,
            search_terms=("masked controlnet",),
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("control_net", _CONTROL_NET),
                InputSpec("vae", _VAE),
                InputSpec("image", _IMAGE),
                InputSpec("mask", TypeExpr.concrete(MASK_TYPE)),
                _strength(),
                *_range(advanced=True),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        control_net: Any,
        vae: Any,
        image: object,
        mask: object,
        strength: float = 1.0,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
    ) -> Mapping[str, object]:
        extra_concat: list[Any] = []
        image_value: Any = materialize_image(image)
        if control_net.concat_mask:
            utils = importlib.import_module("dinkster_inference.utils")
            mask_value: Any = materialize_image(mask)
            inverse_mask = 1.0 - mask_value.reshape(
                (-1, 1, mask_value.shape[-2], mask_value.shape[-1])
            )
            mask_apply = utils.common_upscale(
                inverse_mask, image_value.shape[2], image_value.shape[1], "bilinear", "center"
            ).round()
            image_value = image_value * mask_apply.movedim(1, -1).repeat(
                1, 1, 1, image_value.shape[3]
            )
            extra_concat = [inverse_mask]
        return ControlNetApplyAdvanced.execute(
            positive=positive,
            negative=negative,
            control_net=control_net,
            image=image_value,
            strength=strength,
            start_percent=start_percent,
            end_percent=end_percent,
            vae=vae,
            extra_concat=extra_concat,
        )


CONTROLNET_NODES = (
    ControlNetApply,
    ControlNetApplyAdvanced,
    SetUnionControlNetType,
    ControlNetInpaintingAliMamaApply,
)
