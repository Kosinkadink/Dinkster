"""Resolve MHR and external rig keypoints into OpenPose camera coordinates."""

from __future__ import annotations

import importlib
from typing import Any

from .pose_draw import BODY_EDGES, COLORS, HAND_EDGES, HEAD_EDGES

np = importlib.import_module("numpy")

OPENPOSE_18_PAIRS = tuple((a - 1, b - 1) for a, b in (*BODY_EDGES, *HEAD_EDGES))
OPENPOSE_RAINBOW_18 = np.array(COLORS, dtype=np.float32) / 255.0
OPENPOSE_HAND_PAIRS = tuple(tuple(edge) for edge in HAND_EDGES)
SCAIL_LIMB_COLORS_17 = (
    np.array(
        [
            [255, 0, 0],
            [0, 255, 255],
            [255, 85, 0],
            [255, 170, 0],
            [0, 170, 255],
            [0, 85, 255],
            [180, 255, 0],
            [0, 255, 0],
            [0, 255, 85],
            [0, 0, 255],
            [85, 0, 255],
            [170, 0, 255],
            [150, 150, 150],
            [255, 0, 170],
            [50, 0, 255],
            [255, 0, 170],
            [50, 0, 255],
        ],
        dtype=np.float32,
    )
    / 255.0
)
OPENPOSE_HAND_COLORS_21 = (
    np.array(
        [
            [100, 100, 100],
            [100, 0, 0],
            [150, 0, 0],
            [200, 0, 0],
            [255, 0, 0],
            [100, 100, 0],
            [150, 150, 0],
            [200, 200, 0],
            [255, 255, 0],
            [0, 100, 50],
            [0, 150, 75],
            [0, 200, 100],
            [0, 255, 125],
            [0, 100, 100],
            [0, 150, 150],
            [0, 200, 200],
            [0, 255, 255],
            [100, 0, 100],
            [150, 0, 150],
            [200, 0, 200],
            [255, 0, 255],
        ],
        dtype=np.float32,
    )
    / 255.0
)

_MAPS = {
    "body": (
        "openpose18_joint_indices",
        "openpose18_joint_weights",
        [0, 69, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 2, 1, 4, 3],
    ),
    "hand_r": (
        "openpose_hand21_r_joint_indices",
        "openpose_hand21_r_joint_weights",
        [41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37],
    ),
    "hand_l": (
        "openpose_hand21_l_joint_indices",
        "openpose_hand21_l_joint_weights",
        [62, 45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50, 57, 56, 55, 54, 61, 60, 59, 58],
    ),
}


def openpose_render_keypoints(
    person: Any,
    pose_data: Any,
    part: str,
    *,
    dim: int,
    H: int = 0,
    W: int = 0,
) -> Any:
    map_key, weight_key, mhr_map = _MAPS[part]
    override: Any = pose_data.get("_skeleton_override") if pose_data is not None else None
    external_map = override.get(map_key) if override is not None else None
    if external_map is not None:
        joints = person.get("pred_joint_coords")
        if joints is None:
            return None
        joints = np.asarray(joints, dtype=np.float32)
        mapping = np.asarray(external_map, dtype=np.int64)
        a, b = mapping[:, 0].astype(np.int64), mapping[:, 1].astype(np.int64)
        has_b = b >= 0
        if not has_b.any():
            points = joints[a].astype(np.float32, copy=False).copy()
        else:
            weights = override.get(weight_key)
            weights = (
                np.where(has_b, 0.5, 1.0).astype(np.float32)
                if weights is None
                else np.where(has_b, np.asarray(weights, dtype=np.float32), 1.0)
            )
            other_weights = (1.0 - weights).astype(np.float32)
            points = (
                (
                    joints[a] * weights[:, None]
                    + joints[np.where(has_b, b, a)] * other_weights[:, None]
                )
                .astype(np.float32, copy=False)
                .copy()
            )
        points[:, 1:3] *= -1.0
        if dim == 3:
            return points
        translation, focal = person.get("pred_cam_t"), person.get("focal_length")
        if translation is None or focal is None:
            return None
        points = points + np.asarray(translation, dtype=np.float32).reshape(1, 3)
        z = np.maximum(points[:, 2:3], 1e-6)
        f = float(np.asarray(focal, dtype=np.float32).reshape(-1)[0])
        xy = points[:, :2] * f + np.array([W * 0.5, H * 0.5], dtype=np.float32)[None, :] * z
        return (xy / z).astype(np.float32)
    keypoints = person.get("pred_keypoints_3d" if dim == 3 else "pred_keypoints_2d")
    if keypoints is None:
        return None
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[0] < 70:
        return None
    return keypoints[mhr_map]
