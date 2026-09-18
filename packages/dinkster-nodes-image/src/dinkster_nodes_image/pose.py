"""Typed pose JSON interchange and deterministic control-image rendering."""

from __future__ import annotations

import colorsys
import json
import math
from collections.abc import Mapping, Sequence
from typing import cast

import cv2
import numpy as np
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

from .support import check_output_size as _check_output_size
from .types import (
    MAX_POSE_SKELETONS,
    MAX_SKELETON_KEYPOINTS,
    POSE_TYPE,
    Pose,
    PoseKeypoint,
    PoseSkeleton,
)

IMAGE = TypeExpr.concrete("dinkster.image")
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)
POSE = TypeExpr.concrete(POSE_TYPE)
POSE_LIST = TypeExpr.list_of(POSE)
MAX_POSE_FRAMES = 1_024
MAX_POSE_JSON_CHARACTERS = 16 * 1024 * 1024

BODY_EDGES = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)
FOOT_EDGES = ((13, 18), (18, 19), (18, 20), (10, 21), (21, 22), (21, 23))
HAND_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
ANIMAL_EDGES = (
    (0, 1),
    (1, 2),
    (0, 2),
    (2, 3),
    (3, 8),
    (8, 9),
    (9, 10),
    (3, 5),
    (5, 6),
    (6, 7),
    (3, 4),
    (4, 14),
    (14, 15),
    (15, 16),
    (4, 11),
    (11, 12),
    (12, 13),
)
BODY_COLORS = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
)
ANIMAL_COLORS = (
    (255, 255, 255),
    (100, 255, 100),
    (150, 255, 255),
    (100, 50, 255),
    (50, 150, 200),
    (0, 255, 255),
    (0, 150, 0),
    (0, 0, 255),
    (0, 0, 150),
    (255, 50, 255),
    (255, 0, 255),
    (255, 0, 0),
    (150, 0, 0),
    (255, 255, 100),
    (0, 150, 0),
    (255, 255, 0),
    (150, 150, 150),
)


def _number(value: object, subject: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{subject} must be a number")
    result = float(cast("int | float", value))
    if not math.isfinite(result):
        raise ValueError(f"{subject} must be finite")
    return result


def _keypoints(value: object, subject: str) -> tuple[tuple[PoseKeypoint | None, ...], bool]:
    if value is None:
        return (), True
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{subject} must be an array")
    raw = cast("Sequence[object]", value)
    if not raw:
        return (), True
    rows: list[Sequence[object] | None]
    if all(
        item is None or isinstance(item, Sequence) and not isinstance(item, (str, bytes))
        for item in raw
    ):
        rows = [cast("Sequence[object] | None", item) for item in raw]
    else:
        if len(raw) % 3:
            raise ValueError(f"{subject} flat array length must be divisible by 3")
        rows = [raw[index : index + 3] for index in range(0, len(raw), 3)]
    if len(rows) > MAX_SKELETON_KEYPOINTS:
        raise ValueError(f"{subject} cannot exceed {MAX_SKELETON_KEYPOINTS} keypoints")

    points: list[PoseKeypoint | None] = []
    for index, row in enumerate(rows):
        if row is None:
            points.append(None)
            continue
        if len(row) not in (2, 3):
            raise ValueError(
                f"{subject} keypoint {index} must contain x, y, and optional confidence"
            )
        x = _number(row[0], f"{subject} keypoint {index} x")
        y = _number(row[1], f"{subject} keypoint {index} y")
        confidence = (
            _number(row[2], f"{subject} keypoint {index} confidence") if len(row) == 3 else 1.0
        )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{subject} keypoint {index} confidence must be between 0 and 1")
        points.append(None if confidence == 0.0 else PoseKeypoint(x, y, confidence))
    present = [point for point in points if point is not None]
    normalized = not present or all(
        abs(point.x) <= 1.0 and abs(point.y) <= 1.0 for point in present
    )
    return tuple(points), normalized


def _mapping(value: object, subject: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{subject} must be an object")
    return cast("Mapping[str, object]", value)


def _sequence(value: object, subject: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{subject} must be an array")
    return cast("Sequence[object]", value)


def _skeleton(
    kind: str,
    layout: str,
    subject: int,
    raw: object,
    label: str,
    edges: tuple[tuple[int, int], ...],
) -> PoseSkeleton | None:
    points, normalized = _keypoints(raw, label)
    if not points:
        return None
    return PoseSkeleton(
        kind=kind,
        layout=layout,
        subject=subject,
        normalized=normalized,
        keypoints=points,
        edges=tuple(edge for edge in edges if max(edge) < len(points)),
    )


def _pose_from_record(value: object, frame_index: int) -> Pose:
    record = _mapping(value, f"pose frame {frame_index}")
    width = record.get("canvas_width")
    height = record.get("canvas_height")
    if type(width) is not int or type(height) is not int:
        raise TypeError(f"pose frame {frame_index} requires integer canvas_width and canvas_height")
    skeletons: list[PoseSkeleton] = []
    people = _sequence(record.get("people", ()), f"pose frame {frame_index} people")
    if len(people) > MAX_POSE_SKELETONS:
        raise ValueError(f"pose frame {frame_index} cannot exceed {MAX_POSE_SKELETONS} people")

    def append(skeleton: PoseSkeleton | None) -> None:
        if skeleton is None:
            return
        if len(skeletons) == MAX_POSE_SKELETONS:
            raise ValueError(
                f"pose frame {frame_index} cannot exceed {MAX_POSE_SKELETONS} skeletons"
            )
        skeletons.append(skeleton)

    for subject, raw_person in enumerate(people):
        person = _mapping(raw_person, f"pose frame {frame_index} person {subject}")
        body, body_normalized = _keypoints(
            person.get("pose_keypoints_2d"), f"pose frame {frame_index} person {subject} body"
        )
        feet, feet_normalized = _keypoints(
            person.get("foot_keypoints_2d"), f"pose frame {frame_index} person {subject} feet"
        )
        if feet:
            body_present = any(point is not None for point in body)
            feet_present = any(point is not None for point in feet)
            if body_present and feet_present and body_normalized != feet_normalized:
                raise ValueError(
                    f"pose frame {frame_index} person {subject} body and feet use "
                    "different coordinate spaces"
                )
            if body and len(body) != 18:
                raise ValueError(
                    f"pose frame {frame_index} person {subject} feet require 18 body keypoints"
                )
            body = (body or (None,) * 18) + feet
            body_normalized = (
                feet_normalized if feet_present and not body_present else body_normalized
            )
        if body:
            body_edges = BODY_EDGES + (FOOT_EDGES if len(body) == 24 else ())
            append(
                PoseSkeleton(
                    kind="person",
                    layout="openpose_body_feet" if len(body) == 24 else "openpose_body",
                    subject=subject,
                    normalized=body_normalized,
                    keypoints=body,
                    edges=tuple(edge for edge in body_edges if max(edge) < len(body)),
                )
            )
        for kind, field, layout, edges in (
            ("left_hand", "hand_left_keypoints_2d", "openpose_hand", HAND_EDGES),
            ("right_hand", "hand_right_keypoints_2d", "openpose_hand", HAND_EDGES),
            ("face", "face_keypoints_2d", "openpose_face", ()),
        ):
            part = _skeleton(
                kind,
                layout,
                subject,
                person.get(field),
                f"pose frame {frame_index} person {subject} {field}",
                edges,
            )
            append(part)

    animals = _sequence(record.get("animals", ()), f"pose frame {frame_index} animals")
    if len(animals) > MAX_POSE_SKELETONS:
        raise ValueError(f"pose frame {frame_index} cannot exceed {MAX_POSE_SKELETONS} animals")
    if animals and record.get("version") != "ap10k":
        raise ValueError(f"pose frame {frame_index} animal version must be ap10k")
    for subject, raw_animal in enumerate(animals):
        animal = _skeleton(
            "animal",
            "ap10k",
            subject,
            raw_animal,
            f"pose frame {frame_index} animal {subject}",
            ANIMAL_EDGES,
        )
        append(animal)
    return Pose(canvas_width=width, canvas_height=height, skeletons=tuple(skeletons))


def poses_from_json(text: str) -> list[Pose]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"pose JSON contains non-finite number {value}")

    if len(text) > MAX_POSE_JSON_CHARACTERS:
        raise ValueError(f"pose JSON cannot exceed {MAX_POSE_JSON_CHARACTERS} characters")
    try:
        raw = cast("object", json.loads(text, parse_constant=reject_constant))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid pose JSON") from exc
    frames: list[object] = list(cast("list[object]", raw)) if isinstance(raw, list) else [raw]
    if not frames:
        raise ValueError("pose JSON must contain at least one frame")
    if len(frames) > MAX_POSE_FRAMES:
        raise ValueError(f"pose JSON cannot exceed {MAX_POSE_FRAMES} frames")
    return [_pose_from_record(frame, index) for index, frame in enumerate(frames)]


def _flatten_keypoints(points: Sequence[PoseKeypoint | None]) -> list[float]:
    values: list[float] = []
    for point in points:
        if point is None:
            values.extend((0.0, 0.0, 0.0))
        else:
            values.extend((point.x, point.y, point.confidence))
    return values


def _pose_record(pose: Pose) -> dict[str, object]:
    people: dict[int, dict[str, object]] = {}
    animals: list[list[float]] = []
    for skeleton in pose.skeletons:
        if skeleton.kind == "animal":
            animals.append(_flatten_keypoints(skeleton.keypoints))
            continue
        person = people.setdefault(
            skeleton.subject,
            {
                "pose_keypoints_2d": None,
                "face_keypoints_2d": None,
                "hand_left_keypoints_2d": None,
                "hand_right_keypoints_2d": None,
            },
        )
        field = {
            "person": "pose_keypoints_2d",
            "face": "face_keypoints_2d",
            "left_hand": "hand_left_keypoints_2d",
            "right_hand": "hand_right_keypoints_2d",
        }[skeleton.kind]
        if person[field] is not None:
            raise ValueError(f"pose subject {skeleton.subject} repeats {skeleton.kind} skeleton")
        if skeleton.layout == "openpose_body_feet":
            person[field] = _flatten_keypoints(skeleton.keypoints[:18])
            person["foot_keypoints_2d"] = _flatten_keypoints(skeleton.keypoints[18:])
        else:
            person[field] = _flatten_keypoints(skeleton.keypoints)
    record: dict[str, object] = {
        "people": [people[index] for index in sorted(people)],
        "animals": animals,
        "canvas_height": pose.canvas_height,
        "canvas_width": pose.canvas_width,
    }
    if animals:
        record["version"] = "ap10k"
    if _pose_from_record(record, 0) != pose:
        raise ValueError("pose cannot be represented losslessly as OpenPose/controlnet_aux JSON")
    return record


def poses_to_json(poses: Sequence[Pose]) -> str:
    if not poses:
        raise ValueError("poses must contain at least one frame")
    if len(poses) > MAX_POSE_FRAMES:
        raise ValueError(f"poses cannot exceed {MAX_POSE_FRAMES} frames")
    records = [_pose_record(pose) for pose in poses]
    payload: object = records[0] if len(records) == 1 else records
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    if len(text) > MAX_POSE_JSON_CHARACTERS:
        raise ValueError(f"pose JSON cannot exceed {MAX_POSE_JSON_CHARACTERS} characters")
    return text


def _poses(value: object) -> list[Pose]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("poses must be a list of Pose values")
    poses = list(cast("Sequence[object]", value))
    if not poses:
        raise ValueError("poses must contain at least one frame")
    if any(not isinstance(pose, Pose) for pose in poses):
        raise TypeError("poses must contain only Pose values")
    return cast("list[Pose]", poses)


def _pixel(point: PoseKeypoint, skeleton: PoseSkeleton, pose: Pose) -> tuple[float, float]:
    if skeleton.normalized:
        return point.x * pose.canvas_width, point.y * pose.canvas_height
    return point.x, point.y


def _visible(point: PoseKeypoint | None, threshold: float) -> bool:
    return point is not None and point.confidence >= threshold


def _draw_person(
    canvas: np.ndarray,
    pose: Pose,
    skeleton: PoseSkeleton,
    threshold: float,
    xinsr_stick_scaling: bool,
) -> None:
    stick_scale = 1
    if xinsr_stick_scaling:
        longest = max(pose.canvas_width, pose.canvas_height)
        stick_scale = 1 if longest < 500 else min(2 + longest // 1000, 7)
    for index, (start, end) in enumerate(skeleton.edges):
        first, second = skeleton.keypoints[start], skeleton.keypoints[end]
        if not _visible(first, threshold) or not _visible(second, threshold):
            continue
        assert first is not None and second is not None
        x1, y1 = _pixel(first, skeleton, pose)
        x2, y2 = _pixel(second, skeleton, pose)
        middle_x, middle_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        length = math.hypot(y1 - y2, x1 - x2)
        angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
        polygon = cv2.ellipse2Poly(
            (int(middle_x), int(middle_y)),
            (int(length / 2.0), 4 * stick_scale),
            int(angle),
            0,
            360,
            1,
        )
        color = BODY_COLORS[index % len(BODY_COLORS)]
        cv2.fillConvexPoly(
            canvas,
            np.asarray(polygon, dtype=np.int32),
            tuple(int(channel * 0.6) for channel in color),
        )
    for index, point in enumerate(skeleton.keypoints):
        if not _visible(point, threshold):
            continue
        assert point is not None
        x, y = _pixel(point, skeleton, pose)
        cv2.circle(canvas, (int(x), int(y)), 4, BODY_COLORS[index % len(BODY_COLORS)], -1)


def _draw_hand(canvas: np.ndarray, pose: Pose, skeleton: PoseSkeleton, threshold: float) -> None:
    for index, (start, end) in enumerate(skeleton.edges):
        first, second = skeleton.keypoints[start], skeleton.keypoints[end]
        if not _visible(first, threshold) or not _visible(second, threshold):
            continue
        assert first is not None and second is not None
        x1, y1 = _pixel(first, skeleton, pose)
        x2, y2 = _pixel(second, skeleton, pose)
        if min(int(x1), int(y1), int(x2), int(y2)) <= 0:
            continue
        rgb = colorsys.hsv_to_rgb(index / float(len(skeleton.edges)), 1.0, 1.0)
        cv2.line(
            canvas,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            tuple(channel * 255.0 for channel in rgb),
            2,
        )
    for point in skeleton.keypoints:
        if not _visible(point, threshold):
            continue
        assert point is not None
        x, y = _pixel(point, skeleton, pose)
        if int(x) > 0 and int(y) > 0:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), -1)


def _draw_face(canvas: np.ndarray, pose: Pose, skeleton: PoseSkeleton, threshold: float) -> None:
    for point in skeleton.keypoints:
        if not _visible(point, threshold):
            continue
        assert point is not None
        x, y = _pixel(point, skeleton, pose)
        if int(x) > 0 and int(y) > 0:
            cv2.circle(canvas, (int(x), int(y)), 3, (255, 255, 255), -1)


def _draw_animal(canvas: np.ndarray, pose: Pose, skeleton: PoseSkeleton, threshold: float) -> None:
    for index, (start, end) in enumerate(skeleton.edges):
        first, second = skeleton.keypoints[start], skeleton.keypoints[end]
        if not _visible(first, threshold) or not _visible(second, threshold):
            continue
        assert first is not None and second is not None
        x1, y1 = _pixel(first, skeleton, pose)
        x2, y2 = _pixel(second, skeleton, pose)
        cv2.line(
            canvas,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            ANIMAL_COLORS[index % len(ANIMAL_COLORS)],
            5,
        )


def render_poses(
    poses: Sequence[Pose],
    *,
    render_body: bool,
    render_hand: bool,
    render_face: bool,
    render_animal: bool,
    confidence_threshold: float,
    xinsr_stick_scaling: bool,
) -> np.ndarray:
    if not math.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not poses:
        raise ValueError("poses must contain at least one frame")
    if len(poses) > MAX_POSE_FRAMES:
        raise ValueError(f"poses cannot exceed {MAX_POSE_FRAMES} frames")
    first = poses[0]
    if any(
        pose.canvas_width != first.canvas_width or pose.canvas_height != first.canvas_height
        for pose in poses
    ):
        raise ValueError("all pose frames must use the same canvas dimensions")
    _check_output_size((len(poses), first.canvas_height, first.canvas_width, 3))
    frames: list[np.ndarray] = []
    for pose in poses:
        canvas = np.zeros((pose.canvas_height, pose.canvas_width, 3), dtype=np.uint8)
        for skeleton in pose.skeletons:
            if skeleton.kind == "person" and render_body:
                _draw_person(canvas, pose, skeleton, confidence_threshold, xinsr_stick_scaling)
            elif skeleton.kind in ("left_hand", "right_hand") and render_hand:
                _draw_hand(canvas, pose, skeleton, confidence_threshold)
            elif skeleton.kind == "face" and render_face:
                _draw_face(canvas, pose, skeleton, confidence_threshold)
            elif skeleton.kind == "animal" and render_animal:
                _draw_animal(canvas, pose, skeleton, confidence_threshold)
        frames.append(canvas)
    return np.ascontiguousarray(np.stack(frames).astype(np.float32) / 255.0)


class ImportPoseJson(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.pose.import_json",
            display_name="Import Pose JSON",
            category="image/pose",
            inputs=(InputSpec("json", STRING, widget=StringWidget(multiline=True)),),
            outputs=(OutputSpec("poses", POSE_LIST),),
            search_terms=("OpenPose JSON", "DWPose keypoints", "AP10K pose"),
        )

    @classmethod
    def execute(cls, *, json: str) -> Mapping[str, object]:
        return cls.outputs(poses=poses_from_json(json))


class ExportPoseJson(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.pose.export_json",
            display_name="Export Pose JSON",
            category="image/pose",
            inputs=(InputSpec("poses", POSE_LIST),),
            outputs=(OutputSpec("json", STRING),),
            search_terms=("OpenPose JSON", "pose keypoints", "AP10K pose"),
        )

    @classmethod
    def execute(cls, *, poses: object) -> Mapping[str, object]:
        return cls.outputs(json=poses_to_json(_poses(poses)))


class RenderPose(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.pose.render",
            display_name="Render Pose",
            category="image/pose",
            inputs=(
                InputSpec("poses", POSE_LIST),
                InputSpec("render_body", BOOLEAN, required=False, default=True),
                InputSpec("render_hand", BOOLEAN, required=False, default=True),
                InputSpec("render_face", BOOLEAN, required=False, default=True),
                InputSpec("render_animal", BOOLEAN, required=False, default=True),
                InputSpec(
                    "confidence_threshold",
                    FLOAT,
                    required=False,
                    default=0.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "xinsr_stick_scaling",
                    BOOLEAN,
                    required=False,
                    default=False,
                    hidden=True,
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
            search_terms=("OpenPose", "DWPose", "animal pose", "controlnet keypoints"),
        )

    @classmethod
    def execute(
        cls,
        *,
        poses: object,
        render_body: bool = True,
        render_hand: bool = True,
        render_face: bool = True,
        render_animal: bool = True,
        confidence_threshold: float = 0.0,
        xinsr_stick_scaling: bool = False,
    ) -> Mapping[str, object]:
        return cls.outputs(
            image=render_poses(
                _poses(poses),
                render_body=render_body,
                render_hand=render_hand,
                render_face=render_face,
                render_animal=render_animal,
                confidence_threshold=confidence_threshold,
                xinsr_stick_scaling=xinsr_stick_scaling,
            )
        )


POSE_NODES: tuple[type[Node], ...] = (ImportPoseJson, ExportPoseJson, RenderPose)


__all__ = [
    "POSE_NODES",
    "ExportPoseJson",
    "ImportPoseJson",
    "RenderPose",
    "poses_from_json",
    "poses_to_json",
    "render_poses",
]
