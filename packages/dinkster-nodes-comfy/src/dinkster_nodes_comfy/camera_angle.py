"""Native camera orbit coordinates and shot descriptions."""

from __future__ import annotations

import math
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    LOAD3D_CAMERA_TYPE,
    CustomWidgetDescriptor,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

_HORIZONTAL_TERMS = (
    "front view",
    "front-right quarter view",
    "right side view",
    "back-right quarter view",
    "back view",
    "back-left quarter view",
    "left side view",
    "front-left quarter view",
)


class CameraAngle(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CameraAngle",
            display_name="Compose Camera Angle Prompt",
            category="3d",
            description=(
                "Pick a camera angle around a subject with a 3D preview. Outputs a camera_info "
                "for 3D nodes and a plain-English shot description for prompts."
            ),
            search_terms=(
                "multi angle",
                "camera view",
                "orbit camera",
                "shot angle",
                "camera prompt",
            ),
            inputs=(
                InputSpec(
                    "horizontal_angle",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    widget=NumberWidget(min=0, max=360, step=1),
                    doc=(
                        "Azimuth around the subject in degrees. "
                        "0 is the front, 90 the right side, 180 the back."
                    ),
                ),
                InputSpec(
                    "vertical_angle",
                    TypeExpr.concrete(CORE_INT),
                    default=0,
                    widget=NumberWidget(min=-30, max=60, step=1),
                    doc=(
                        "Elevation in degrees. "
                        "Negative looks up from below, positive looks down from above."
                    ),
                ),
                InputSpec(
                    "zoom",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=5.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.1),
                    doc=(
                        "Lens zoom on the subject: 0 is a wide shot, 10 a close-up. "
                        "Carried into camera_info.zoom."
                    ),
                ),
                InputSpec(
                    "image",
                    TypeExpr.concrete(IMAGE_TYPE),
                    required=False,
                    doc=(
                        "Optional reference image shown on the front of the subject cube "
                        "in the 3D preview."
                    ),
                ),
                InputSpec(
                    "view",
                    TypeExpr.concrete(CORE_STRING),
                    default="",
                    required=False,
                    socketless=True,
                    widget=CustomWidgetDescriptor("CAMERA_ANGLE_VIEW", {"multiline": False}),
                ),
            ),
            outputs=(
                OutputSpec("camera_info", TypeExpr.concrete(LOAD3D_CAMERA_TYPE)),
                OutputSpec("prompt", TypeExpr.concrete(CORE_STRING)),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        horizontal_angle: int,
        vertical_angle: int,
        zoom: float,
        image: object = None,
        view: object = None,
    ) -> Mapping[str, object]:
        horizontal_angle = max(0, min(360, int(horizontal_angle)))
        vertical_angle = max(-30, min(60, int(vertical_angle)))
        zoom = max(0.0, min(10.0, float(zoom)))
        yaw, pitch = math.radians(horizontal_angle), math.radians(vertical_angle)
        camera = {
            "position": {
                "x": 0.0 + 6.0 * math.cos(pitch) * math.sin(yaw),
                "y": 0.0 + 6.0 * math.sin(pitch),
                "z": 0.0 + 6.0 * math.cos(pitch) * math.cos(yaw),
            },
            "target": {"x": 0.0, "y": 0.0, "z": 0.0},
            "zoom": 1.0 + (1.875 - 1.0) * (zoom / 10.0),
            "cameraType": "perspective",
            "fov": 35.0,
        }
        horizontal = _HORIZONTAL_TERMS[int(((horizontal_angle % 360) + 22.5) // 45) % 8]
        vertical = (
            "low-angle shot"
            if vertical_angle < -15
            else "eye-level shot"
            if vertical_angle < 15
            else "elevated shot"
            if vertical_angle < 45
            else "high-angle shot"
        )
        distance = "wide shot" if zoom < 2 else "medium shot" if zoom < 6 else "close-up"
        return cls.outputs(camera_info=camera, prompt=f"{horizontal} {vertical} {distance}")
