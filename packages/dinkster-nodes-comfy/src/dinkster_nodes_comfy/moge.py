"""Camera measurements from MoGe geometry."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    MOGE_GEOMETRY_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)


class MoGeGeometryToFOV(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MoGeGeometryToFOV",
            display_name="Get FoV from MoGe Geometry",
            description="Derive the field of view and focal length from MoGe intrinsics.",
            category="image/geometry estimation",
            inputs=(
                InputSpec("moge_geometry", TypeExpr.concrete(MOGE_GEOMETRY_TYPE)),
                InputSpec(
                    "axis",
                    TypeExpr.concrete(CORE_COMBO),
                    default="vertical",
                    widget=ComboWidget(("vertical", "horizontal", "diagonal")),
                    doc="'vertical' (fov_y), 'horizontal' (fov_x), or 'diagonal'.",
                ),
                InputSpec(
                    "unit",
                    TypeExpr.concrete(CORE_COMBO),
                    default="degrees",
                    widget=ComboWidget(("degrees", "radians")),
                    doc="Output unit for the FOV.",
                ),
            ),
            outputs=(
                OutputSpec("fov", TypeExpr.concrete(CORE_FLOAT)),
                OutputSpec("focal_pixels", TypeExpr.concrete(CORE_FLOAT)),
            ),
        )

    @classmethod
    def execute(cls, *, moge_geometry: object, axis: str, unit: str) -> Mapping[str, object]:
        value = materialize_latent(moge_geometry)
        geometry = cast("dict[str, Any]", value) if isinstance(value, dict) else {}
        intrinsics = geometry.get("intrinsics")
        if intrinsics is None:
            raise ValueError("moge_geometry has no intrinsics (panorama geometry has none).")
        if intrinsics.ndim == 3:
            intrinsics = intrinsics[0]
        hx = 0.5 / float(intrinsics[0, 0].item())
        hy = 0.5 / float(intrinsics[1, 1].item())
        half_tan = {"horizontal": hx, "vertical": hy, "diagonal": math.hypot(hx, hy)}[axis]
        radians = 2.0 * math.atan(half_tan)
        fov = radians if unit == "radians" else math.degrees(radians)
        source = next(
            (geometry[key] for key in ("image", "points", "depth") if key in geometry), None
        )
        if source is None:
            raise ValueError(
                "moge_geometry has no image/points/depth to read the pixel height from."
            )
        focal_pixels = float(intrinsics[1, 1].item()) * int(source.shape[1])
        return cls.outputs(fov=fov, focal_pixels=focal_pixels)
