"""OpenPose body, hand and projected face overlays for SAM3D pose packets."""

from __future__ import annotations

import importlib
from typing import Any

from .pose_draw import draw_disks, draw_pose
from .sam3d_pose import OPENPOSE_HAND_COLORS_21, openpose_render_keypoints

np = importlib.import_module("numpy")
Image = importlib.import_module("PIL.Image")

_FACE_TARGETS = (
    (-0.058, 1.690, 0.090),
    (-0.040, 1.695, 0.105),
    (-0.020, 1.692, 0.115),
    (+0.020, 1.692, 0.115),
    (+0.040, 1.695, 0.105),
    (+0.058, 1.690, 0.090),
    (-0.058, 1.660, 0.085),
    (-0.040, 1.673, 0.090),
    (-0.022, 1.665, 0.092),
    (-0.040, 1.652, 0.090),
    (+0.058, 1.660, 0.085),
    (+0.040, 1.673, 0.090),
    (+0.022, 1.665, 0.092),
    (+0.040, 1.652, 0.090),
    (0.000, 1.660, 0.110),
    (0.000, 1.620, 0.125),
    (0.000, 1.585, 0.135),
    (-0.014, 1.580, 0.115),
    (+0.014, 1.580, 0.115),
    (-0.030, 1.540, 0.105),
    (+0.000, 1.555, 0.115),
    (+0.030, 1.540, 0.105),
    (+0.000, 1.530, 0.110),
    (0.000, 1.498, 0.108),
    (-0.038, 1.512, 0.100),
    (-0.062, 1.535, 0.080),
    (-0.078, 1.562, 0.060),
    (+0.038, 1.512, 0.100),
    (+0.062, 1.535, 0.080),
    (+0.078, 1.562, 0.060),
)


def _face_vertex_ids(positions: Any, face_mask: Any) -> Any:
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    if face_mask is not None and np.asarray(face_mask).any():
        valid = np.where(np.asarray(face_mask).reshape(-1))[0]
    else:
        valid = np.where(
            (positions[:, 1] > 1.47) & (np.abs(positions[:, 0]) < 0.11) & (positions[:, 2] > 0.04)
        )[0]
    if valid.size == 0:
        raise ValueError("No head vertices matched the canonical positions/face_mask filter")
    candidates = positions[valid]
    return np.asarray(
        [
            valid[
                int(np.sum((candidates - np.asarray(xyz, dtype=np.float32)) ** 2, axis=1).argmin())
            ]
            for xyz in _FACE_TARGETS
        ],
        dtype=np.int64,
    )


def _project_face(person: Any, indices: Any, H: int, W: int) -> Any:
    vertices, translation, focal = (
        person.get("pred_vertices"),
        person.get("pred_cam_t"),
        person.get("focal_length"),
    )
    if vertices is None or translation is None or focal is None:
        return None
    points = np.asarray(vertices, dtype=np.float32)[indices]
    points = points + np.asarray(translation, dtype=np.float32).reshape(1, 3)
    z = np.maximum(points[:, 2:3], 1e-6)
    f = float(np.asarray(focal, dtype=np.float32).reshape(-1)[0])
    xy = points[:, :2] * f
    xy = xy + np.array([W * 0.5, H * 0.5], dtype=np.float32)[None, :] * z
    return (xy / z).astype(np.float32)


def render_pose_data_openpose(
    pose_data: Any,
    *,
    frame_idx: int,
    W: int,
    H: int,
    background: Any = None,
    composite: str = "over",
    marker_radius_px: int = 4,
    stick_width_px: int = 4,
    limb_alpha: float = 0.6,
    include_body: bool = True,
    include_hands: bool = False,
    face_style: str = "disabled",
    hand_color_style: str = "dwpose",
    face_marker_radius_px: int = 3,
    person_brightness_falloff: float = 0.0,
) -> Any:
    torch = importlib.import_module("torch")
    persons = pose_data["frames"][frame_idx]
    if composite == "over" and background is not None:
        canvas = (np.clip(background.cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)
        if canvas.shape[:2] != (H, W):
            canvas = np.array(Image.fromarray(canvas).resize((W, H), Image.Resampling.LANCZOS))
    else:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas = np.ascontiguousarray(canvas)
    face_indices = None
    if face_style != "disabled":
        any_real = face_style != "eyes_mouth" and any(
            person.get("pred_face_keypoints_2d") is not None for person in persons
        )
        if not any_real:
            canonical: Any = pose_data.get("canonical_colors") or {}
            if canonical.get("positions") is not None:
                face_indices = _face_vertex_ids(canonical["positions"], canonical.get("face_mask"))
                if face_style == "eyes_mouth":
                    face_indices = face_indices[[6, 7, 8, 9, 10, 11, 12, 13, 19, 20, 21, 22]]
    hand_colors = (
        (OPENPOSE_HAND_COLORS_21 * 255.0).astype(int)
        if hand_color_style == "openpose"
        else (0, 0, 255)
    )
    falloff = max(0.0, min(1.0, float(person_brightness_falloff)))
    for index, person in enumerate(persons):
        pastel = 0.0 if index == 0 else 1.0 - falloff**index
        before = canvas.copy() if pastel > 0 else None
        keypoints = np.zeros((134, 2), dtype=np.float32)
        scores = np.zeros(134, dtype=np.float32)
        parts = ([(0, "body")] if include_body else []) + (
            [(92, "hand_r"), (113, "hand_l")] if include_hands else []
        )
        for start, part in parts:
            points = openpose_render_keypoints(person, pose_data, part, dim=2, H=H, W=W)
            if points is not None:
                finite = np.isfinite(points).all(axis=1)
                keypoints[start : start + len(points)][finite] = points[finite]
                scores[start : start + len(points)][finite] = 1.0
        draw_pose(
            canvas,
            keypoints,
            scores,
            threshold=0.5,
            draw_body=include_body,
            draw_head=True,
            draw_feet=False,
            draw_face=False,
            draw_hands=include_hands,
            stick_width=stick_width_px,
            face_point_size=3,
            marker_radius=marker_radius_px,
            hand_stick_width=max(1, int(round(stick_width_px * 0.5))),
            hand_marker_radius=max(1, int(round(marker_radius_px * 0.7))),
            limb_alpha=limb_alpha,
            hand_dot_color=hand_colors,
        )
        if face_style != "disabled":
            face = None
            real_face = person.get("pred_face_keypoints_2d")
            if real_face is not None:
                points = np.asarray(real_face, dtype=np.float32)
                if points.ndim == 2 and points.shape[1] == 2:
                    face = points
            elif face_indices is not None:
                face = _project_face(person, face_indices, H, W)
            if face is not None:
                centers: list[tuple[int, int]] = []
                radius = int(face_marker_radius_px)
                for x_float, y_float in face:
                    if not (np.isfinite(x_float) and np.isfinite(y_float)):
                        continue
                    x, y = int(round(float(x_float))), int(round(float(y_float)))
                    if x + radius < 0 or x - radius >= W or y + radius < 0 or y - radius >= H:
                        continue
                    centers.append((x, y))
                draw_disks(canvas, centers, radius, (255, 255, 255))
        if before is not None:
            changed = (canvas != before).any(axis=-1)
            touched = canvas[changed].astype(np.float32)
            canvas[changed] = np.clip(touched * (1.0 - pastel) + 255.0 * pastel, 0.0, 255.0).astype(
                np.uint8
            )
    return torch.from_numpy(canvas.astype(np.float32) / 255.0)
