"""Ideogram 4 native family boundaries."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any

from ..family_registry import encode_component_text
from ..native_arm_core import (
    _NATIVE_PREPARED_CONDITIONING_KEY,
    NativeRuntimeHandle,
    _component_bound_carrier,
    _torch,
    importlib,
)
from ..native_arm_runtime import _torch_dtype


def encode_text(value: object, text: str, options: Any) -> object:
    return encode_component_text(
        value,
        text,
        options,
        runtime="dinkster_inference_torch:Ideogram4TextRuntime",
        carrier="dinkster_inference_torch:ideogram4_conditioning_to_carrier",
        role="qwen3vl_8b",
    )


def resolve_ideogram4_component_execution(
    handle: NativeRuntimeHandle,
    positive: object,
    negative: object,
    inference: Any,
    *,
    negative_handle: NativeRuntimeHandle | None = None,
    image_only_negative: bool = False,
) -> tuple[Any, object, object] | None:
    recipe = handle.recipe
    family_id = inference.IDEOGRAM4_CONFIG.family_id
    if recipe.family_id != family_id:
        return None
    if tuple(source.role for source in recipe.sources) != ("diffusion",):
        return None
    inference_torch = importlib.import_module("dinkster_inference_torch")
    base_runtime = handle.runtime
    if (
        not isinstance(base_runtime, inference_torch.Ideogram4DiffusionRuntime)
        or recipe.runtime_identity != base_runtime.runtime_identity
    ):
        raise TypeError("model must be a native Ideogram 4 diffusion component")
    negative_runtime = None
    if negative_handle is not None:
        negative_recipe = negative_handle.recipe
        negative_base = negative_handle.runtime
        if (
            negative_recipe.family_id != family_id
            or tuple(source.role for source in negative_recipe.sources) != ("diffusion",)
            or not isinstance(negative_base, inference_torch.Ideogram4DiffusionRuntime)
            or negative_recipe.runtime_identity != negative_base.runtime_identity
        ):
            raise TypeError("model_negative must be a native Ideogram 4 diffusion component")
    positive_carrier, positive_binding = _component_bound_carrier(positive, inference)
    if positive_binding is None:
        raise TypeError("positive must be Ideogram 4 component-bound conditioning")
    if positive_binding.family_id != family_id or positive_binding.role != "qwen3vl_8b":
        raise ValueError("positive Ideogram 4 conditioning has the wrong component binding")
    negative_carrier = None
    if negative not in ([], None):
        negative_carrier, negative_binding = _component_bound_carrier(negative, inference)
        if negative_binding is None:
            raise TypeError("negative must be Ideogram 4 component-bound conditioning or empty")
        if negative_binding != positive_binding:
            raise ValueError("Ideogram 4 conditioning lanes must share one component binding")
    components = {
        "diffusion": recipe.runtime_identity,
        "qwen3vl_8b": positive_binding.identity,
    }
    if negative_handle is not None:
        components["negative-diffusion"] = negative_handle.recipe.runtime_identity
    composition = inference.compose_execution(family_id, components)
    torch = _torch()
    runtime = inference_torch.Ideogram4DiffusionRuntime(
        base_runtime.assembled.diffusion,
        runtime_identity=composition.execution_identity,
        compute_dtype=_torch_dtype(torch, recipe.knobs.diffusion_dtype),
    )
    if negative_handle is not None:
        negative_runtime = inference_torch.Ideogram4DiffusionRuntime(
            negative_handle.runtime.assembled.diffusion,
            runtime_identity=composition.execution_identity,
            compute_dtype=_torch_dtype(torch, negative_handle.recipe.knobs.diffusion_dtype),
        )
    conditioning = runtime.prepare_single_stream_conditioning(positive_carrier)
    rows = [[conditioning.embeddings, {_NATIVE_PREPARED_CONDITIONING_KEY: conditioning}]]
    negative_rows: object = []
    target_runtime = runtime if negative_runtime is None else negative_runtime
    uncond = (
        target_runtime.image_only_conditioning()
        if image_only_negative
        else (
            None
            if negative_carrier is None
            else target_runtime.prepare_single_stream_conditioning(negative_carrier)
        )
    )
    if uncond is not None:
        if negative_runtime is not None:
            uncond = inference_torch.RoutedConditioning(
                embeddings=uncond.embeddings,
                pooled=uncond.pooled,
                evaluation=negative_runtime.conditioning_evaluation(),
                source=uncond,
            )
        negative_rows = [[uncond.embeddings, {_NATIVE_PREPARED_CONDITIONING_KEY: uncond}]]
    return runtime, rows, negative_rows
