"""Face boxes and whole-body rendering from canonical SDPose keypoint frames."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    BOUNDING_BOX_TYPE,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    POSE_KEYPOINT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
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


class SDPoseDrawKeypoints(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        boolean = TypeExpr.concrete(CORE_BOOLEAN)
        integer = TypeExpr.concrete(CORE_INT)
        return NodeSchema(
            node_type="SDPoseDrawKeypoints",
            display_name="SDPose Draw Keypoints",
            category="image/detection",
            search_terms=("openpose", "pose detection", "preprocessor", "keypoints", "pose"),
            inputs=(
                InputSpec("keypoints", TypeExpr.concrete(POSE_KEYPOINT_TYPE)),
                InputSpec("draw_body", boolean, default=True),
                InputSpec("draw_hands", boolean, default=True),
                InputSpec("draw_face", boolean, default=True),
                InputSpec("draw_feet", boolean, default=False),
                InputSpec(
                    "stick_width", integer, default=4, widget=NumberWidget(min=1, max=10, step=1)
                ),
                InputSpec(
                    "face_point_size",
                    integer,
                    default=3,
                    widget=NumberWidget(min=1, max=10, step=1),
                ),
                InputSpec(
                    "score_threshold",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.3,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec("draw_head", boolean, default=True),
            ),
            outputs=(OutputSpec("IMAGE", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        keypoints: list[dict[str, Any]],
        draw_body: bool,
        draw_hands: bool,
        draw_face: bool,
        draw_feet: bool,
        stick_width: int,
        face_point_size: int,
        score_threshold: float,
        draw_head: bool,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        if not keypoints:
            return cls.outputs(IMAGE=torch.zeros((1, 64, 64, 3), dtype=torch.float32))
        initialize_inference_process()
        management = importlib.import_module("dinkster_inference.model_management")
        np = importlib.import_module("numpy")
        from .pose_draw import draw_pose

        height, width = keypoints[0]["canvas_height"], keypoints[0]["canvas_width"]

        def parse(flat: Any, count: int) -> tuple[Any, Any]:
            arr = np.array(flat, dtype=np.float32).reshape(count, 3)
            return arr[:, :2], arr[:, 2]

        outputs: list[Any] = []
        for frame in keypoints:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for person in frame["people"]:
                body, body_scores = parse(person["pose_keypoints_2d"], 18)
                foot_raw = person.get("foot_keypoints_2d")
                feet, feet_scores = (
                    parse(foot_raw, 6)
                    if foot_raw
                    else (np.zeros((6, 2), dtype=np.float32), np.zeros(6, dtype=np.float32))
                )
                face, face_scores = parse(person["face_keypoints_2d"], 70)
                right, right_scores = parse(person["hand_right_keypoints_2d"], 21)
                left, left_scores = parse(person["hand_left_keypoints_2d"], 21)
                points = np.concatenate([body, feet, face[:68], right, left], axis=0)
                scores = np.concatenate(
                    [body_scores, feet_scores, face_scores[:68], right_scores, left_scores], axis=0
                )
                canvas = draw_pose(
                    canvas,
                    points,
                    scores,
                    threshold=score_threshold,
                    draw_body=draw_body,
                    draw_head=draw_head,
                    draw_feet=draw_feet,
                    draw_face=draw_face,
                    draw_hands=draw_hands,
                    stick_width=stick_width,
                    face_point_size=face_point_size,
                )
            outputs.append(canvas)
        images = np.stack(outputs) if len(outputs) > 1 else np.expand_dims(outputs[0], 0)
        result = (
            torch.from_numpy(images).to(
                device=management.intermediate_device(), dtype=management.intermediate_dtype()
            )
            / 255.0
        )
        return cls.outputs(IMAGE=result)
