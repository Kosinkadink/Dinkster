"""Native attention projection multiplier nodes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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


def _strength(name: str) -> InputSpec:
    return InputSpec(
        name,
        TypeExpr.concrete(CORE_FLOAT),
        default=1.0,
        advanced=True,
        widget=NumberWidget(min=0.0, max=10.0, step=0.01),
    )


def _schema(node_type: str, *, clip: bool = False) -> NodeSchema:
    value_type = CLIP_TYPE if clip else MODEL_TYPE
    name = "clip" if clip else "model"
    return NodeSchema(
        node_type=node_type,
        category="experimental/attention_experiments",
        inputs=(
            InputSpec(name, TypeExpr.concrete(value_type)),
            _strength("q"),
            _strength("k"),
            _strength("v"),
            _strength("out"),
        ),
        outputs=(OutputSpec(value_type, TypeExpr.concrete(value_type)),),
        search_terms=("clip attention scale", "text encoder attention") if clip else (),
    )


def _multiply(model: Any, keys: Iterable[str], strengths: Mapping[str, float]) -> Any:
    multiplied = model.clone()
    for key in keys:
        for projection, strength in strengths.items():
            if key.endswith(projection):
                multiplied.add_patches({key: (None,)}, 0.0, strength)
                break
    return multiplied


def _unet_attention(model: Any, attention: str, q: float, k: float, v: float, out: float) -> Any:
    strengths = {
        f"{attention}.to_q.bias": q,
        f"{attention}.to_q.weight": q,
        f"{attention}.to_k.bias": k,
        f"{attention}.to_k.weight": k,
        f"{attention}.to_v.bias": v,
        f"{attention}.to_v.weight": v,
        f"{attention}.to_out.0.bias": out,
        f"{attention}.to_out.0.weight": out,
    }
    return _multiply(model, model.model_state_dict(), strengths)


class UNetSelfAttentionMultiply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(cls.__name__)

    @classmethod
    def execute(
        cls, *, model: Any, q: float, k: float, v: float, out: float
    ) -> Mapping[str, object]:
        return cls.outputs(MODEL=_unet_attention(model, "attn1", q, k, v, out))


class UNetCrossAttentionMultiply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(cls.__name__)

    @classmethod
    def execute(
        cls, *, model: Any, q: float, k: float, v: float, out: float
    ) -> Mapping[str, object]:
        return cls.outputs(MODEL=_unet_attention(model, "attn2", q, k, v, out))


class CLIPAttentionMultiply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(cls.__name__, clip=True)

    @classmethod
    def execute(
        cls, *, clip: Any, q: float, k: float, v: float, out: float
    ) -> Mapping[str, object]:
        strengths = {
            "self_attn.q_proj.weight": q,
            "self_attn.q_proj.bias": q,
            "self_attn.k_proj.weight": k,
            "self_attn.k_proj.bias": k,
            "self_attn.v_proj.weight": v,
            "self_attn.v_proj.bias": v,
            "self_attn.out_proj.weight": out,
            "self_attn.out_proj.bias": out,
        }
        return cls.outputs(CLIP=_multiply(clip, clip.patcher.model_state_dict(), strengths))


class UNetTemporalAttentionMultiply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=cls.__name__,
            category="experimental/attention_experiments",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                _strength("self_structural"),
                _strength("self_temporal"),
                _strength("cross_structural"),
                _strength("cross_temporal"),
            ),
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        self_structural: float,
        self_temporal: float,
        cross_structural: float,
        cross_temporal: float,
    ) -> Mapping[str, object]:
        multiplied = model.clone()
        for key in model.model_state_dict():
            if key.endswith(("attn1.to_out.0.bias", "attn1.to_out.0.weight")):
                strength = self_temporal if ".time_stack." in key else self_structural
            elif key.endswith(("attn2.to_out.0.bias", "attn2.to_out.0.weight")):
                strength = cross_temporal if ".time_stack." in key else cross_structural
            else:
                continue
            multiplied.add_patches({key: (None,)}, 0.0, strength)
        return cls.outputs(MODEL=multiplied)


ATTENTION_MULTIPLY_NODES = (
    UNetSelfAttentionMultiply,
    UNetCrossAttentionMultiply,
    CLIPAttentionMultiply,
    UNetTemporalAttentionMultiply,
)
