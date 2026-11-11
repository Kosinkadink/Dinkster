"""ComfyUI image and latent dimension inspection nodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
    report_progress,
    with_identity_replacement,
)

_LATENT = TypeExpr.concrete(LATENT_TYPE)


_INT = TypeExpr.concrete(CORE_INT)


class GetLatentSize(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return with_identity_replacement(
            NodeSchema(
                node_type="dinkster.latent.get_latent_size",
                display_name="Get Image/Video Latent Size",
                description=(
                    "Returns the width, height, frame count, and batch size of an image or "
                    "video latent."
                ),
                category="latent",
                inputs=(InputSpec("latent", _LATENT, doc="The image or video latent to inspect."),),
                outputs=(
                    OutputSpec("width", _INT, doc="The latent width."),
                    OutputSpec("height", _INT, doc="The latent height."),
                    OutputSpec(
                        "frames", _INT, doc="The latent frame count. Image latents have one frame."
                    ),
                    OutputSpec("batch_size", _INT, doc="The number of latents in the batch."),
                ),
                search_terms=("dimensions", "resolution", "latent info", "video latent size"),
            ),
            "GetLatentSize",
        )

    @classmethod
    def execute(cls, *, latent: Any) -> Mapping[str, object]:
        samples = materialize_latent(latent)["samples"]
        if samples.is_nested:
            samples = samples.unbind()[0]
        if samples.ndim == 5:
            batch_size, _, frames, height, width = samples.shape
        else:
            batch_size, _, height, width = samples.shape
            frames = 1
        report_progress(
            1,
            1,
            text=f"width: {width}, height: {height}\n frames: {frames}, batch size: {batch_size}",
        )
        return cls.outputs(width=width, height=height, frames=frames, batch_size=batch_size)
