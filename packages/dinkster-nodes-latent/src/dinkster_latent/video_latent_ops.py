"""Video latent transform nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_latent,
    with_identity_replacement,
)

_LATENT = TypeExpr.concrete(LATENT_TYPE)


class TrimVideoLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.latent.trim_video_latent",
                display_name="Trim Video Latent",
                category="model/latent",
                inputs=(
                    InputSpec("samples", _LATENT),
                    InputSpec(
                        "trim_amount",
                        TypeExpr.concrete(CORE_INT),
                        default=0,
                        widget=NumberWidget(min=0, max=99999),
                    ),
                ),
                outputs=(OutputSpec("LATENT", _LATENT),),
            ),
            "TrimVideoLatent",
        )

    @classmethod
    def execute(cls, *, samples: dict[str, Any], trim_amount: int) -> Mapping[str, object]:
        samples = materialize_latent(samples)
        result = samples.copy()
        result["samples"] = samples["samples"][:, :, trim_amount:]
        return cls.outputs(LATENT=result)


class NormalizeVideoLatentStart(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.latent.normalize_video_latent_start",
                display_name="NormalizeVideoLatentStart",
                category="model/conditioning",
                inputs=(
                    InputSpec("latent", _LATENT),
                    InputSpec(
                        "start_frame_count",
                        TypeExpr.concrete(CORE_INT),
                        default=4,
                        widget=NumberWidget(min=1, max=16384, step=1),
                    ),
                    InputSpec(
                        "reference_frame_count",
                        TypeExpr.concrete(CORE_INT),
                        default=5,
                        widget=NumberWidget(min=1, max=16384, step=1),
                    ),
                ),
                outputs=(OutputSpec("latent", _LATENT),),
            ),
            "NormalizeVideoLatentStart",
        )

    @classmethod
    def execute(
        cls, *, latent: dict[str, Any], start_frame_count: int, reference_frame_count: int
    ) -> Mapping[str, object]:
        latent = materialize_latent(latent)
        samples = latent["samples"]
        if samples.shape[2] <= 1:
            return cls.outputs(latent=latent)

        torch = importlib.import_module("torch")
        result = latent.copy()
        normalized_samples = samples.clone()
        source = normalized_samples[:, :, :start_frame_count]
        reference = normalized_samples[
            :,
            :,
            start_frame_count : start_frame_count
            + min(reference_frame_count, samples.shape[2] - 1),
        ]
        source_mean = source.mean(dim=(1, 3, 4), keepdim=True)
        source_std = source.std(dim=(1, 3, 4), keepdim=True)
        reference_mean = torch.clamp(reference.mean(), source_mean - 0.3, source_mean + 0.35)
        reference_std = torch.clamp(reference.std(), source_std - 0.35, source_std + 0.5)
        normalized_samples[:, :, :start_frame_count] = (source - source_mean) / (
            source_std + 1e-8
        ) * reference_std + reference_mean
        result["samples"] = normalized_samples
        return cls.outputs(latent=result)
