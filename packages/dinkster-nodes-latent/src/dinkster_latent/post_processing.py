"""Native ComfyUI resize, batching, and color-transfer nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    LATENT_TYPE,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
    with_identity_replacement,
)

_LATENT = TypeExpr.concrete(LATENT_TYPE)


def _common_upscale(value: Any, width: int, height: int, method: str, crop: str) -> Any:
    return importlib.import_module("dinkster_inference.utils").common_upscale(
        value, width, height, method, crop
    )


def _batch_latents(latents: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not latents:
        return None
    torch = importlib.import_module("torch")
    latents = [materialize_latent(latent) for latent in latents]
    result = latents[0].copy()
    first = latents[0]["samples"]
    tensors: list[Any] = []
    result_batch_indices: list[int] = []
    result["batch_index"] = result_batch_indices
    for latent in latents:
        samples = latent["samples"]
        if not hasattr(samples, "detach"):
            samples = torch.as_tensor(samples)
        if samples.shape[1:] != first.shape[1:]:
            samples = _common_upscale(
                samples, first.shape[-1], first.shape[-2], "bilinear", "center"
            )
        tensors.append(samples)
        batch_indices = cast("list[int]", latent.get("batch_index", list(range(samples.shape[0]))))
        result_batch_indices.extend(batch_indices)
    result["samples"] = torch.cat(tensors, dim=0)
    return result


class BatchLatentsNode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.latent.batch_latents_node",
                display_name="Batch Latents",
                category="model/latent/batch",
                input_families=(
                    InputFamilySpec(
                        "latents",
                        (InputSpec("latent", _LATENT),),
                        1,
                        50,
                        member_prefix="latent",
                        optional_after_minimum=True,
                    ),
                ),
                outputs=(OutputSpec("LATENT", _LATENT),),
            ),
            "BatchLatentsNode",
        )

    @classmethod
    def execute(cls, *, latents: Mapping[str, dict[str, Any]]) -> Mapping[str, object]:
        return cls.outputs(LATENT=_batch_latents(list(latents.values())))
