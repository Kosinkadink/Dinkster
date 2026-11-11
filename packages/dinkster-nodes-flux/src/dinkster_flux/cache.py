"""Model sampling and reference-attention cache patches."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    with_identity_replacement,
)

_MODEL = TypeExpr.concrete(MODEL_TYPE)


class _ReferenceAttentionCache:
    def __init__(self) -> None:
        self.cache: dict[str, tuple[Any, Any]] = {}

    def __call__(
        self, q: Any, k: Any, v: Any, extra_options: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        counts = extra_options.get("reference_image_num_tokens", [])
        if not counts:
            return {}
        torch = importlib.import_module("torch")
        key = f"{extra_options['block_type']}_{extra_options['block_index']}"
        if key in self.cache:
            initialize_inference_process()
            repeat = importlib.import_module("dinkster_inference.utils").repeat_to_batch_size
            cached_k, cached_v = self.cache[key]
            self.set_cache = False
            return {
                "q": q,
                "k": torch.cat((k, repeat(cached_k, k.shape[0])), dim=2),
                "v": torch.cat((v, repeat(cached_v, v.shape[0])), dim=2),
            }
        count = sum(counts)
        self.cache[key] = (k[:, :, -count:].clone(), v[:, :, -count:].clone())
        self.set_cache = True
        return {"q": q, "k": k, "v": v}

    def cleanup(self) -> None:
        self.cache = {}


class FluxKVCache(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.flux.flux_kv_cache",
            display_name="Flux KV Cache",
            description="Enables KV Cache optimization for reference images on Flux family models.",
            category="experimental",
            inputs=(InputSpec("model", _MODEL),),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )
        return with_identity_replacement(schema, "FluxKVCache")

    @classmethod
    def execute(cls, *, model: Any) -> Mapping[str, object]:
        patched = model.clone()
        attention = _ReferenceAttentionCache()

        def input_patch(inputs: dict[str, Any]) -> dict[str, Any]:
            if attention.cache:
                count = sum(inputs["transformer_options"].get("reference_image_num_tokens", []))
                if count > 0:
                    inputs["img"] = inputs["img"][:, :-count]
            return inputs

        patched.set_model_attn1_patch(attention)
        patched.set_model_post_input_patch(input_patch)
        target = (
            "diffusion_model.params.default_ref_method"
            if hasattr(model.model.diffusion_model, "params")
            else "diffusion_model.default_ref_method"
        )
        patched.add_object_patch(target, "index_timestep_zero")
        return cls.outputs(MODEL=patched)
