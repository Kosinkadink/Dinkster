"""Qwen, Z-Image, and USO image-control model patches."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_clip_vision_output,
    with_identity_replacement,
)

_MODEL = TypeExpr.concrete(MODEL_TYPE)

_MODEL_PATCH = TypeExpr.concrete(MODEL_PATCH_TYPE)


def _modules() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    initialize_inference_process()
    return (
        importlib.import_module("torch"),
        importlib.import_module("dinkster_inference.utils"),
        importlib.import_module("dinkster_inference.latent_formats"),
        importlib.import_module("dinkster_inference.model_management"),
        importlib.import_module("dinkster_inference.model_prefetch"),
        importlib.import_module("dinkster_inference.patcher_extension"),
        importlib.import_module("dinkster_inference.ldm.qwen_image21.model"),
    )


class UsoStyleProjectorPatch:
    def __init__(self, model_patch: Any, encoded_image: Any) -> None:
        self._device_type = _modules()[0].device
        self.model_patch, self.encoded_image = model_patch, encoded_image

    def __call__(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        torch, *_ = _modules()
        text, ids = kwargs["txt"], kwargs["txt_ids"]
        embedding = self.model_patch.model(self.encoded_image.to(text.dtype)).to(text.dtype)
        kwargs["txt"] = torch.cat((embedding, text), dim=1)
        kwargs["txt_ids"] = torch.cat(
            (
                torch.zeros(
                    embedding.shape[0], embedding.shape[1], 3, dtype=ids.dtype, device=ids.device
                ),
                ids,
            ),
            dim=1,
        )
        return kwargs

    def to(self, device_or_dtype: Any) -> UsoStyleProjectorPatch:
        if isinstance(device_or_dtype, self._device_type):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
        return self

    def models(self) -> list[Any]:
        return [self.model_patch]


class USOStyleReference(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.flux.uso_style_reference",
            display_name="Apply USO Style Reference",
            category="model/patch/flux",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec("model_patch", _MODEL_PATCH),
                InputSpec("clip_vision_output", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )
        return with_identity_replacement(schema, "USOStyleReference")

    @classmethod
    def execute(
        cls, *, model: Any, model_patch: Any, clip_vision_output: object
    ) -> Mapping[str, object]:
        torch, *_ = _modules()
        vision = materialize_clip_vision_output(clip_vision_output)
        encoded = torch.stack(
            (
                vision.all_hidden_states[:, -20],
                vision.all_hidden_states[:, -11],
                vision.penultimate_hidden_states,
            )
        )
        patched = model.clone()
        patched.set_model_post_input_patch(UsoStyleProjectorPatch(model_patch, encoded))
        return cls.outputs(MODEL=patched)
