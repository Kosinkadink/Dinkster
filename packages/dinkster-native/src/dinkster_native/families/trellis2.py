"""TRELLIS.2 native family execution boundaries."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any

from ..native_arm_core import NativeRuntimeHandle, importlib
from .conditioning import _resident_payload


def resolve_trellis2_component_execution(
    handle: NativeRuntimeHandle, positive: object, negative: object, inference: Any
) -> tuple[Any, object, object] | None:
    recipe = handle.recipe
    if recipe.family_id != inference.TRELLIS2.id:
        return None
    source_roles = tuple(source.role for source in recipe.sources)
    if source_roles not in (
        ("diffusion",),
        ("shape", "shape-512", "structure", "texture", "texture-512"),
    ):
        return None
    inference_torch = importlib.import_module("dinkster_inference_torch")
    runtime = handle.runtime
    if (
        not isinstance(runtime, inference_torch.Trellis2DiffusionRuntime)
        or recipe.runtime_identity != runtime.runtime_identity
    ):
        raise TypeError("model must be a native TRELLIS.2 diffusion component")
    resource_type = inference_torch.Trellis2ConditioningResource
    try:
        positive_resource = _resident_payload(positive, inference, "positive")
    except TypeError as error:
        raise TypeError("positive must be resident TRELLIS.2 conditioning") from error
    if not isinstance(positive_resource, resource_type):
        raise TypeError("positive must be resident TRELLIS.2 conditioning")
    if positive_resource.guidance_role is not inference.GuidanceRole.CONDITIONAL:
        raise ValueError("positive TRELLIS.2 conditioning has the wrong guidance lane")
    negative_resource = None
    if negative not in ([], None):
        try:
            negative_resource = _resident_payload(negative, inference, "negative")
        except TypeError as error:
            raise TypeError("negative must be resident TRELLIS.2 conditioning or empty") from error
        if not isinstance(negative_resource, resource_type):
            raise TypeError("negative must be resident TRELLIS.2 conditioning or empty")
        if negative_resource.guidance_role is not inference.GuidanceRole.UNCONDITIONAL:
            raise ValueError("negative TRELLIS.2 conditioning has the wrong guidance lane")
        if not positive_resource.shares_backing(negative_resource):
            raise ValueError("TRELLIS.2 conditioning lanes must share one backing resource")
        if negative_resource.stage != positive_resource.stage:
            raise ValueError("TRELLIS.2 conditioning lanes must use the same stage")

    def rows(resource: object) -> list[list[object]]:
        return [
            [
                inference.PreparedMultiStreamConditioning(
                    runtime.conditioning_identity,
                    resource,
                ),
                dict[str, object](),
            ]
        ]

    return (
        runtime,
        rows(positive_resource),
        ([] if negative_resource is None else rows(negative_resource)),
    )
