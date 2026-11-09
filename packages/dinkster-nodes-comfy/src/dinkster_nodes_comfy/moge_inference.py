"""Chunked monocular geometry prediction from image batches."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    MOGE_GEOMETRY_TYPE,
    MOGE_MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
)


class MoGeInference(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MoGeInference",
            display_name="Run MoGe Inference",
            description="Run MoGe on a single image to estimate depth and geometry.",
            category="image/geometry estimation",
            search_terms=("moge", "depth", "geometry", "depth estimation", "geometry estimation"),
            inputs=(
                InputSpec("moge_model", TypeExpr.concrete(MOGE_MODEL_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "resolution_level",
                    TypeExpr.concrete(CORE_INT),
                    default=9,
                    widget=NumberWidget(min=0, max=9),
                    doc="0 = fastest, 9 = most detail.",
                ),
                InputSpec(
                    "fov_x_degrees",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=170.0, step=0.1),
                    doc=(
                        "Horizontal field of view of the source camera. Sets the focal length "
                        "used to unproject the depth map into 3D. 0 = auto-recover from the "
                        "predicted points."
                    ),
                ),
                InputSpec(
                    "batch_size",
                    TypeExpr.concrete(CORE_INT),
                    default=4,
                    widget=NumberWidget(min=1, max=64),
                    doc="Images per inference call. Lower if you OOM on a long video / image set.",
                ),
                InputSpec(
                    "force_projection",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=True,
                    advanced=True,
                ),
                InputSpec(
                    "apply_mask",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=True,
                    advanced=True,
                    doc=(
                        "Set masked-out (sky / invalid) pixels to inf in points and depth so "
                        "meshing culls them. Disable to keep the raw predicted geometry "
                        "everywhere; the mask is still returned separately."
                    ),
                ),
                InputSpec(
                    "refine_steps",
                    TypeExpr.concrete(CORE_INT),
                    default=3,
                    advanced=True,
                    widget=NumberWidget(min=0, max=8),
                    doc=(
                        "MoGe-3 only: sparse volumetric refinement passes over the predicted "
                        "depth. "
                        "More passes sharpen fine detail and edges at a roughly linear cost. "
                        "0 disables refinement. Ignored by MoGe-1 / MoGe-2."
                    ),
                ),
            ),
            outputs=(OutputSpec("moge_geometry", TypeExpr.concrete(MOGE_GEOMETRY_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        moge_model: Any,
        image: Any,
        resolution_level: int = 9,
        fov_x_degrees: float = 0.0,
        batch_size: int = 4,
        force_projection: bool = True,
        apply_mask: bool = True,
        refine_steps: int = 3,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        pixels = materialize_image(image)[..., :3]
        bchw = pixels.movedim(-1, -3).contiguous()
        chunks: list[dict[str, Any]] = []
        for i in range(0, bchw.shape[0], batch_size):
            chunks.append(
                moge_model.infer(
                    bchw[i : i + batch_size],
                    resolution_level=resolution_level,
                    fov_x=None if fov_x_degrees <= 0 else float(fov_x_degrees),
                    force_projection=force_projection,
                    apply_mask=apply_mask,
                    refine_steps=refine_steps,
                )
            )
        geometry: dict[str, Any] = {"image": pixels.cpu()}
        for field in ("points", "depth", "intrinsics", "mask", "normal"):
            values = [chunk[field] for chunk in chunks if field in chunk]
            if values:
                geometry[field] = torch.cat(values, dim=0)
        return {"moge_geometry": geometry}
