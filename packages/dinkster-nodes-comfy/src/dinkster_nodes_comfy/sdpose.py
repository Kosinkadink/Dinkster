"""Face bounding boxes from canonical SDPose keypoint frames."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CORE_BOOLEAN,
    CORE_FLOAT,
    POSE_KEYPOINT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)


def _face_bbox(kp2ds: Any, scale: float, image_shape: tuple[int, int]) -> list[int]:
    np = importlib.import_module("numpy")
    h, w = image_shape
    face = kp2ds.copy()[1:] * (w, h)
    min_x, min_y = np.min(face, axis=0)
    max_x, max_y = np.max(face, axis=0)
    width, height = max_x - min_x, max_y - min_y
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    area = width * height * scale
    new_width = np.sqrt(area * (width / height))
    new_height = np.sqrt(area * (height / width))
    dx, dy = (new_width - width) / 2, (new_height - height) / 4
    return [
        int(max(min_x - dx, 0)),
        int(min(max_x + dx, w)),
        int(max(min_y - 3 * dy, 0)),
        int(min(max_y + dy, h)),
    ]


class SDPoseFaceBBoxes(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SDPoseFaceBBoxes",
            display_name="SDPose Face Bounding Boxes",
            category="image/detection",
            search_terms=("face bbox", "face bounding box", "pose", "keypoints"),
            inputs=(
                InputSpec("keypoints", TypeExpr.concrete(POSE_KEYPOINT_TYPE)),
                InputSpec(
                    "scale",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.5,
                    widget=NumberWidget(min=1.0, max=10.0, step=0.1),
                    doc="Multiplier for the bounding box area around each detected face.",
                ),
                InputSpec(
                    "force_square",
                    TypeExpr.concrete(CORE_BOOLEAN),
                    default=True,
                    doc="Expand the shorter bbox axis so the crop region is always square.",
                ),
            ),
            outputs=(
                OutputSpec(
                    "bboxes",
                    TypeExpr.concrete(BOUNDING_BOX_TYPE),
                    doc="Face bounding boxes per frame, compatible with SDPoseKeypointExtractor "
                    "bboxes input.",
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, keypoints: list[dict[str, Any]], scale: float, force_square: bool
    ) -> Mapping[str, object]:
        np = importlib.import_module("numpy")
        all_bboxes: list[list[dict[str, int]]] = []
        for frame in keypoints:
            h, w = frame["canvas_height"], frame["canvas_width"]
            frame_bboxes: list[dict[str, int]] = []
            for person in frame["people"]:
                face_flat = person.get("face_keypoints_2d", [])
                if not face_flat:
                    continue
                face_xy = np.array(face_flat, dtype=np.float32).reshape(-1, 3)[:, :2]
                normalized = face_xy / np.array([w, h], dtype=np.float32)
                padded = np.vstack([np.zeros((1, 2), dtype=np.float32), normalized])
                x1, x2, y1, y2 = _face_bbox(padded, scale, (h, w))
                if x2 > x1 and y2 > y1:
                    if force_square:
                        bw, bh = x2 - x1, y2 - y1
                        if bw != bh:
                            side = max(bw, bh)
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            half = side // 2
                            x1, y1 = max(0, cx - half), max(0, cy - half)
                            x2, y2 = min(w, x1 + side), min(h, y1 + side)
                            x1, y1 = max(0, x2 - side), max(0, y2 - side)
                    frame_bboxes.append({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1})
            all_bboxes.append(frame_bboxes)
        return cls.outputs(bboxes=all_bboxes)
