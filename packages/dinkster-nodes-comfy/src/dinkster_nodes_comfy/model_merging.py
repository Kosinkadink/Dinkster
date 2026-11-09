"""Model and text-encoder merging through the inference patcher contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)


def _coefficient(name: str, *, subtract: bool = False) -> InputSpec:
    return InputSpec(
        name,
        TypeExpr.concrete(CORE_FLOAT),
        default=1.0,
        widget=NumberWidget(
            min=-10.0 if subtract else 0.0, max=10.0 if subtract else 1.0, step=0.01
        ),
    )


def _merge_schema(node_type: str, *, clip: bool = False, coefficient: str = "") -> NodeSchema:
    type_id = CLIP_TYPE if clip else MODEL_TYPE
    prefix = "clip" if clip else "model"
    inputs = (
        InputSpec(prefix + "1", TypeExpr.concrete(type_id)),
        InputSpec(prefix + "2", TypeExpr.concrete(type_id)),
    )
    if coefficient:
        inputs += (_coefficient(coefficient, subtract=coefficient == "multiplier"),)
    return NodeSchema(
        node_type=node_type,
        category="model/merging",
        inputs=inputs,
        outputs=(OutputSpec("CLIP" if clip else "MODEL", TypeExpr.concrete(type_id)),),
    )


def _merge(
    first: Any, second: Any, patch_strength: float, model_strength: float, *, clip: bool = False
) -> Any:
    merged = first.clone()
    patches = second.get_key_patches() if clip else second.get_key_patches("diffusion_model.")
    for key in patches:
        if clip and (key.endswith(".position_ids") or key.endswith(".logit_scale")):
            continue
        merged.add_patches({key: patches[key]}, patch_strength, model_strength)
    return merged


class ModelMergeSimple(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _merge_schema("ModelMergeSimple", coefficient="ratio")

    @classmethod
    def execute(cls, *, model1: Any, model2: Any, ratio: float) -> Mapping[str, object]:
        return cls.outputs(MODEL=_merge(model1, model2, 1.0 - ratio, ratio))


class ModelMergeSubtract(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _merge_schema("ModelMergeSubtract", coefficient="multiplier")

    @classmethod
    def execute(cls, *, model1: Any, model2: Any, multiplier: float) -> Mapping[str, object]:
        return cls.outputs(MODEL=_merge(model1, model2, -multiplier, multiplier))


class ModelMergeAdd(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _merge_schema("ModelMergeAdd")

    @classmethod
    def execute(cls, *, model1: Any, model2: Any) -> Mapping[str, object]:
        return cls.outputs(MODEL=_merge(model1, model2, 1.0, 1.0))


class CLIPMergeSimple(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _merge_schema("CLIPMergeSimple", clip=True, coefficient="ratio")

    @classmethod
    def execute(cls, *, clip1: Any, clip2: Any, ratio: float) -> Mapping[str, object]:
        return cls.outputs(CLIP=_merge(clip1, clip2, 1.0 - ratio, ratio, clip=True))


class CLIPMergeSubtract(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _merge_schema("CLIPMergeSubtract", clip=True, coefficient="multiplier")

    @classmethod
    def execute(cls, *, clip1: Any, clip2: Any, multiplier: float) -> Mapping[str, object]:
        return cls.outputs(CLIP=_merge(clip1, clip2, -multiplier, multiplier, clip=True))


class CLIPMergeAdd(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _merge_schema("CLIPMergeAdd", clip=True)

    @classmethod
    def execute(cls, *, clip1: Any, clip2: Any) -> Mapping[str, object]:
        return cls.outputs(CLIP=_merge(clip1, clip2, 1.0, 1.0, clip=True))


def block_merge_schema(
    node_type: str,
    block_names: tuple[str, ...],
    *,
    category: str = "model/merging",
    description: str = "",
) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        category=category,
        description=description,
        inputs=(
            InputSpec("model1", TypeExpr.concrete(MODEL_TYPE)),
            InputSpec("model2", TypeExpr.concrete(MODEL_TYPE)),
            *(_coefficient(name) for name in block_names),
        ),
        outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
    )


class ModelMergeBlocks(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return block_merge_schema("ModelMergeBlocks", ("input", "middle", "out"))

    @classmethod
    def execute(cls, *, model1: Any, model2: Any, **ratios: float) -> Mapping[str, object]:
        merged = model1.clone()
        patches = model2.get_key_patches("diffusion_model.")
        default_ratio = next(iter(ratios.values()))
        for key in patches:
            ratio = default_ratio
            unet_key = key[len("diffusion_model.") :]
            longest = 0
            for prefix, candidate in ratios.items():
                if unet_key.startswith(prefix) and longest < len(prefix):
                    ratio = candidate
                    longest = len(prefix)
            merged.add_patches({key: patches[key]}, 1.0 - ratio, ratio)
        return cls.outputs(MODEL=merged)


MODEL_MERGE_NODES = (
    ModelMergeSimple,
    ModelMergeSubtract,
    ModelMergeAdd,
    CLIPMergeSimple,
    CLIPMergeSubtract,
    CLIPMergeAdd,
    ModelMergeBlocks,
)
