"""Drive SAM3D Body expressions from its resident face landmarker."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    MHR_POSE_DATA_TYPE,
    SAM3D_BODY_MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_vision_packet,
    report_progress,
)

# Multiple routes take their maximum, not their sum, to avoid doubling a driver.
_AXIS_TO_ARKIT = {
    0: (("browDownLeft", 1.0),),
    1: (("browDownRight", 1.0),),
    2: (("cheekPuff", 1.0),),
    3: (("cheekPuff", 1.0),),
    4: (("cheekSquintLeft", 1.0),),
    5: (("cheekSquintRight", 1.0),),
    6: (("mouthStretchLeft", 1.0),),
    7: (("mouthStretchRight", 1.0),),
    8: (("mouthShrugLower", 1.0),),
    9: (("mouthShrugUpper", 1.0),),
    10: (("mouthDimpleLeft", 1.0),),
    11: (("mouthDimpleRight", 1.0),),
    12: (("eyeLookDownLeft", 0.3),),
    13: (("eyeLookDownRight", 0.3),),
    14: (("eyeBlinkLeft", 1.0),),
    15: (("eyeBlinkRight", 1.0),),
    16: (("eyeLookOutLeft", 1.0),),
    17: (("eyeLookInRight", 1.0),),
    18: (("eyeLookInLeft", 1.0),),
    19: (("eyeLookOutRight", 1.0),),
    22: (("eyeLookUpLeft", 1.0), ("browInnerUp", 0.5)),
    23: (("eyeLookUpRight", 1.0), ("browInnerUp", 0.5)),
    24: (("jawOpen", 1.0), ("mouthLowerDownLeft", 0.3), ("mouthLowerDownRight", 0.3)),
    25: (("jawLeft", 1.0),),
    26: (("jawRight", 1.0),),
    27: (("jawForward", 1.0),),
    28: (("eyeSquintLeft", 1.0),),
    29: (("eyeSquintRight", 1.0),),
    32: (("mouthSmileLeft", 1.0),),
    33: (("mouthSmileRight", 1.0),),
    40: (("mouthLeft", 1.0),),
    41: (("mouthRight", 1.0),),
    42: (("mouthFrownLeft", 1.0),),
    43: (("mouthFrownRight", 1.0),),
    54: (("mouthLowerDownLeft", 1.0),),
    55: (("mouthLowerDownRight", 1.0),),
    60: (("noseSneerLeft", 1.0),),
    61: (("noseSneerRight", 1.0),),
    66: (("browOuterUpLeft", 1.0),),
    67: (("browOuterUpRight", 1.0),),
    68: (("eyeWideLeft", 1.0),),
    69: (("eyeWideRight", 1.0),),
    70: (("mouthUpperUpLeft", 1.0),),
    71: (("mouthUpperUpRight", 1.0),),
}


class SAM3DBody_FaceExpression(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SAM3DBody_FaceExpression",
            display_name="Face Expression to SAM3D Body",
            description="Drive MHR face blendshapes from the core MediaPipe Face Landmarker.",
            category="image/detection",
            inputs=(
                InputSpec("sam3d_body_model", TypeExpr.concrete(SAM3D_BODY_MODEL_TYPE)),
                InputSpec("mhr_pose_data", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=4.0, step=0.05),
                    doc="Global multiplier on all blendshapes. >1 exaggerates.",
                ),
                InputSpec(
                    "mouth_strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=4.0, step=0.05),
                    advanced=True,
                    doc="Multiplier on mouth/jaw shapes. MediaPipe's jawOpen saturates near 1.0.",
                ),
                InputSpec(
                    "eye_strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=2.0,
                    widget=NumberWidget(min=0.0, max=4.0, step=0.05),
                    advanced=True,
                    doc="Multiplier on eye shapes. MediaPipe rarely exceeds 0.5; "
                    "2-3x often needed.",
                ),
                InputSpec(
                    "brow_strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=2.0,
                    widget=NumberWidget(min=0.0, max=4.0, step=0.05),
                    advanced=True,
                    doc="Multiplier on brow/cheek/sneer shapes. MediaPipe outputs ~0.1-0.3; 2-3x.",
                ),
                InputSpec(
                    "input_threshold",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.02,
                    widget=NumberWidget(min=0.0, max=0.5, step=0.01),
                    advanced=True,
                    doc="Deadzone on MediaPipe's raw output (below = zero, above = linear remap). ",
                ),
                InputSpec(
                    "blendshape_smooth_window",
                    TypeExpr.concrete(CORE_INT),
                    default=7,
                    widget=NumberWidget(min=1, max=31, step=2),
                    advanced=True,
                    doc="Gaussian window on MediaPipe's per-frame signal before MHR mapping. "
                    "MediaPipe's raw output swings 30-70% frame-to-frame on static faces. "
                    "1 = disabled. Use odd values.",
                ),
            ),
            outputs=(OutputSpec("mhr_pose_data", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        sam3d_body_model: Any,
        mhr_pose_data: Any,
        image: Any,
        strength: float = 1.0,
        mouth_strength: float = 1.0,
        eye_strength: float = 2.0,
        brow_strength: float = 2.0,
        input_threshold: float = 0.02,
        blendshape_smooth_window: int = 7,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        management = importlib.import_module("dinkster_inference.model_management")
        management.load_model_gpu(sam3d_body_model)
        inner = sam3d_body_model.model
        pose = materialize_vision_packet(mhr_pose_data)
        frames = pose["frames"]
        count = len(frames)
        if not count:
            return cls.outputs(mhr_pose_data=mhr_pose_data)
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        pixels = (materialize_image(image) * 255.0).clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
        copied = [[dict(person) for person in frame] for frame in frames]
        maximum = max((len(frame) for frame in copied), default=0)
        coefs: list[list[Any]] = [[None] * count for _ in range(maximum)]
        height, width = pixels.shape[1:3]
        minimum = 32.0 * max(height, width) / (192.0 * 0.20)
        crops = any(
            person["bbox"][3] - person["bbox"][1] < minimum for frame in copied for person in frame
        )
        populated = 0
        for index, persons in enumerate(copied):
            frame_image = pixels[min(index, len(pixels) - 1)]
            if persons:
                populated += 1
                boxes = [np.asarray(person["bbox"], dtype=np.float32) for person in persons]
                h, w = frame_image.shape[:2]
                if crops:
                    for person, box in enumerate(boxes):
                        crop = _head_crop(persons[person].get("pred_keypoints_2d"), box, w, h)
                        faces = _detect_crop(inner, frame_image, crop)
                        if faces:
                            cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
                            best = min(
                                faces,
                                key=lambda face: (
                                    (0.5 * (face["bbox_xyxy"][0] + face["bbox_xyxy"][2]) - cx) ** 2
                                    + (0.5 * (face["bbox_xyxy"][1] + face["bbox_xyxy"][3]) - cy)
                                    ** 2
                                ),
                            )
                            coefs[person][index] = best["blendshapes"]
                else:
                    faces = inner.face_landmarker.detect_batch(
                        [frame_image], num_faces=max(1, len(persons))
                    )[0]
                    for person, face in enumerate(_assign_faces(faces, boxes)):
                        if face is not None:
                            coefs[person][index] = faces[face]["blendshapes"]
            report_progress(index + 1, count)
        for person in range(maximum):
            coefs[person] = _temporal_coefficients(
                coefs[person], populated >= 30, int(blendshape_smooth_window)
            )
        written = False
        for index, persons in enumerate(copied):
            for person, value in enumerate(persons):
                coefficients = coefs[person][index]
                if coefficients is not None:
                    value["expr_params"] = _expression(
                        coefficients,
                        strength,
                        mouth_strength,
                        eye_strength,
                        brow_strength,
                        input_threshold,
                    )
                    written = True
        if written:
            _regenerate(inner, copied)
        return cls.outputs(mhr_pose_data={**pose, "frames": copied})


def _expression(
    coefs: Any, strength: float, mouth: float, eye: float, brow: float, threshold: float
) -> Any:
    np = importlib.import_module("numpy")
    result = np.zeros(72, dtype=np.float32)
    for axis, routes in _AXIS_TO_ARKIT.items():
        best = 0.0
        for name, weight in routes:
            raw = float(coefs.get(name, 0.0))
            deadzone = (
                0.0 if name in {"jawOpen", "jawForward", "jawLeft", "jawRight"} else threshold
            )
            if deadzone > 0:
                raw = 0.0 if raw <= deadzone else (raw - deadzone) / (1.0 - deadzone)
            gain = (
                mouth
                if name.startswith(("jaw", "mouth"))
                else eye
                if name.startswith("eye")
                else brow
                if name.startswith(("brow", "cheek", "nose"))
                else 1.0
            )
            best = max(best, raw * gain * weight)
        result[axis] = best * strength
    return result


def _temporal_coefficients(series: list[Any], baseline: bool, window: int) -> list[Any]:
    np = importlib.import_module("numpy")
    names: set[str] = set()
    for value in series:
        if value is not None:
            names.update(value.keys())
    if baseline:
        offsets = {
            name: float(
                np.percentile(
                    [value[name] for value in series if value is not None and name in value], 5.0
                )
            )
            for name in names
        }
        series = [
            None
            if value is None
            else {name: max(0.0, float(raw) - offsets[name]) for name, raw in value.items()}
            for value in series
        ]
    if window > 1 and names:
        window += int(window % 2 == 0)
        sigma = max(1.0, window / 5.0)
        x = np.arange(window) - (window - 1) / 2.0
        kernel = np.exp(-(x**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        smoothed: list[Any] = [None] * len(series)
        mask = np.array([value is not None for value in series], dtype=bool)
        for name in names:
            values = np.array(
                [0.0 if value is None else float(value.get(name, 0.0)) for value in series],
                dtype=np.float32,
            )
            if not mask.all():
                indices = np.arange(len(series))
                values = np.interp(indices, indices[mask], values[mask])
            pad = window // 2
            padded = np.concatenate(
                [np.repeat(values[:1], pad), values, np.repeat(values[-1:], pad)]
            )
            filtered = np.zeros_like(values)
            for index, weight in enumerate(kernel):
                filtered += weight * padded[index : index + len(series)]
            for index, value in enumerate(series):
                if value is not None:
                    if smoothed[index] is None:
                        smoothed[index] = {}
                    smoothed[index][name] = float(filtered[index])
        series = smoothed
    # Only interior gaps with both endpoints within twelve frames are filled.
    result = list(series)
    for index, value in enumerate(series):
        if value is not None:
            continue
        previous = next((i for i in range(index - 1, -1, -1) if series[i] is not None), None)
        following = next((i for i in range(index + 1, len(series)) if series[i] is not None), None)
        if previous is None or following is None or max(index - previous, following - index) > 12:
            continue
        weight = (index - previous) / (following - previous)
        a, b = series[previous], series[following]
        result[index] = {
            name: (1.0 - weight) * a.get(name, 0.0) + weight * b.get(name, 0.0)
            for name in set(a) | set(b)
        }
    return result


def _head_crop(keypoints: Any, box: Any, width: int, height: int) -> Any:
    np = importlib.import_module("numpy")
    if keypoints is not None:
        kp = np.asarray(keypoints, dtype=np.float32)
        if kp.ndim == 2 and kp.shape[0] > 4:
            face = kp[:5, :2]
            valid = face[
                (face[:, 0] > 0) & (face[:, 1] > 0) & (face[:, 0] < width) & (face[:, 1] < height)
            ]
            if len(valid) >= 2:
                low, high = valid.min(axis=0), valid.max(axis=0)
                cx, cy = (
                    0.5 * (float(low[0]) + float(high[0])),
                    0.5 * (float(low[1]) + float(high[1])),
                )
                half = (
                    0.5
                    * max(float(high[0]) - float(low[0]), float(high[1]) - float(low[1]), 1.0)
                    * 1.8
                    * 1.2
                )
                return np.array(
                    [
                        max(0.0, cx - half),
                        max(0.0, cy - half),
                        min(float(width), cx + half),
                        min(float(height), cy + half),
                    ],
                    dtype=np.float32,
                )
    x1, y1, x2, y2 = (float(v) for v in box)
    if y2 <= y1 or x2 <= x1:
        return np.zeros(4, dtype=np.float32)
    y2 = y1 + (y2 - y1) * 0.4
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    hw, hh = 0.5 * (x2 - x1) * 1.2, 0.5 * (y2 - y1) * 1.2
    return np.array(
        [
            max(0.0, cx - hw),
            max(0.0, cy - hh),
            min(float(width), cx + hw),
            min(float(height), cy + hh),
        ],
        dtype=np.float32,
    )


def _detect_crop(inner: Any, image: Any, crop: Any) -> list[Any]:
    np = importlib.import_module("numpy")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in crop)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return []
    faces = inner.face_landmarker.detect_batch(
        [np.ascontiguousarray(image[y1:y2, x1:x2])], num_faces=1
    )[0]
    for face in faces:
        face["bbox_xyxy"] = face["bbox_xyxy"] + np.array([x1, y1, x1, y1], dtype=np.float32)
        face["landmarks_xy"] = face["landmarks_xy"] + np.array([x1, y1], dtype=np.float32)
    return faces


def _assign_faces(faces: Any, boxes: list[Any]) -> list[int | None]:
    result: list[int | None] = [None] * len(boxes)
    used: set[int] = set()
    order = sorted(
        range(len(boxes)),
        key=lambda p: -((boxes[p][2] - boxes[p][0]) * (boxes[p][3] - boxes[p][1])),
    )
    for person in order:
        pb = boxes[person]
        best_score = 0.01
        for index, face in enumerate(faces):
            if index in used:
                continue
            fb = face["bbox_xyxy"]
            cx, cy = 0.5 * (fb[0] + fb[2]), 0.5 * (fb[1] + fb[3])
            inside = pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]
            intersection = max(0.0, min(fb[2], pb[2]) - max(fb[0], pb[0])) * max(
                0.0, min(fb[3], pb[3]) - max(fb[1], pb[1])
            )
            union = (
                max(0.0, fb[2] - fb[0]) * max(0.0, fb[3] - fb[1])
                + max(0.0, pb[2] - pb[0]) * max(0.0, pb[3] - pb[1])
                - intersection
            )
            iou = float(intersection / union) if union > 0.0 and intersection > 0.0 else 0.0
            score = max(iou, 0.5 if inside else 0.0)
            if score > best_score:
                result[person], best_score = index, score
        selected = result[person]
        if selected is not None:
            used.add(selected)
    return result


def _regenerate(inner: Any, frames: list[list[dict[str, Any]]]) -> None:
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    device = importlib.import_module("dinkster_inference.model_management").get_torch_device()
    needed = (
        "global_rot",
        "body_pose_params",
        "hand_pose_params",
        "shape_params",
        "scale_params",
        "expr_params",
        "pred_cam_t",
        "focal_length",
    )
    for person in range(max((len(frame) for frame in frames), default=0)):
        active = [
            index
            for index, frame in enumerate(frames)
            if person < len(frame) and all(frame[person].get(key) is not None for key in needed)
        ]
        if not active:
            continue
        params = {
            key: torch.from_numpy(
                np.stack(
                    [np.asarray(frames[index][person][key], dtype=np.float32) for index in active]
                )
            ).to(device)
            for key in needed[:-1]
        }
        focal = torch.tensor(
            [
                float(np.asarray(frames[index][person]["focal_length"]).reshape(-1)[0])
                for index in active
            ],
            device=device,
            dtype=torch.float32,
        )
        vertices, keypoints, joints, _, rotations = inner.head_pose.mhr_forward(
            global_trans=torch.zeros_like(params["global_rot"]),
            global_rot=params["global_rot"],
            body_pose_params=params["body_pose_params"],
            hand_pose_params=params["hand_pose_params"],
            shape_params=params["shape_params"],
            scale_params=params["scale_params"],
            expr_params=params["expr_params"],
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
        )
        vertices, joints = vertices.clone(), joints.clone()
        body_kp, face_kp = keypoints[:, :70].clone(), keypoints[:, 70:].clone()
        for values in (vertices, joints, body_kp, face_kp):
            values[..., [1, 2]] *= -1
        raw = frames[active[0]][person]
        old_point = np.asarray(raw["pred_keypoints_3d"], dtype=np.float32)[0] + np.asarray(
            raw["pred_cam_t"], dtype=np.float32
        )
        old_uv = np.asarray(raw["pred_keypoints_2d"], dtype=np.float32)[0]
        old_focal = float(np.asarray(raw["focal_length"]).reshape(-1)[0])
        cx = float(old_uv[0] - old_focal * old_point[0] / max(old_point[2], 1e-6))
        cy = float(old_uv[1] - old_focal * old_point[1] / max(old_point[2], 1e-6))

        projected: list[Any] = []
        for points in (body_kp, face_kp):
            camera = points + params["pred_cam_t"].unsqueeze(1)
            u = focal[:, None] * camera[..., 0] / camera[..., 2].clamp(min=1e-6) + cx
            v = focal[:, None] * camera[..., 1] / camera[..., 2].clamp(min=1e-6) + cy
            projected.append(torch.stack([u, v], dim=-1))
        updates = {
            "pred_vertices": vertices,
            "pred_keypoints_3d": body_kp,
            "pred_keypoints_2d": projected[0],
            "pred_face_keypoints_3d": face_kp,
            "pred_face_keypoints_2d": projected[1],
            "pred_joint_coords": joints,
            "pred_global_rots": rotations,
        }
        for key, value in updates.items():
            array = value.float().cpu().numpy()
            for row, index in enumerate(active):
                frames[index][person][key] = array[row]
