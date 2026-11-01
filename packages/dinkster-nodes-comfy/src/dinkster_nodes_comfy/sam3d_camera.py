"""Letterboxing and camera reprojection for SAM3D and external rig poses."""

from __future__ import annotations

import importlib
import math
from typing import Any

np = importlib.import_module("numpy")


def scale_pose_data(pose: Any, height: int, width: int) -> Any:
    old_h, old_w = pose["image_size"]
    if (height, width) == (old_h, old_w):
        return pose
    scale = min(width / old_w, height / old_h)
    offset_x, offset_y = (width - scale * old_w) * 0.5, (height - scale * old_h) * 0.5
    frames: list[list[dict[str, Any]]] = []
    for frame in pose["frames"]:
        scaled: list[dict[str, Any]] = []
        for person in frame:
            person = dict(person)
            focal = person.get("focal_length")
            if focal is not None:
                person["focal_length"] = np.asarray(focal, dtype=np.float32) * scale
            for key in ("pred_keypoints_2d", "pred_face_keypoints_2d"):
                value = person.get(key)
                if value is not None:
                    points = np.asarray(value, dtype=np.float32).copy()
                    points[..., 0] = points[..., 0] * scale + offset_x
                    points[..., 1] = points[..., 1] * scale + offset_y
                    person[key] = points
            bbox = person.get("bbox")
            if bbox is not None:
                bbox = np.asarray(bbox, dtype=np.float32).copy()
                bbox[..., [0, 2]] = bbox[..., [0, 2]] * scale + offset_x
                bbox[..., [1, 3]] = bbox[..., [1, 3]] * scale + offset_y
                person["bbox"] = bbox
            scaled.append(person)
        frames.append(scaled)
    return dict(pose, image_size=(height, width), frames=frames)


def _quat_to_mat(w: float, x: float, y: float, z: float) -> Any:
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def apply_camera_override(pose: Any, camera: Any, height: int, width: int) -> Any:
    first: list[dict[str, Any]] = pose["frames"][0] if pose["frames"] else []
    if not first:
        return pose
    roots: list[Any] = []
    centroids: list[Any] = []
    for person in first:
        translation = person.get("pred_cam_t")
        if translation is None:
            continue
        translation = np.asarray(translation, dtype=np.float32).reshape(3)
        roots.append(translation)
        vertices = person.get("pred_vertices")
        centroids.append(
            np.asarray(vertices, dtype=np.float32).reshape(-1, 3).mean(axis=0) + translation
            if vertices is not None
            else translation
        )
    if not roots:
        return pose
    root, centroid = np.mean(np.stack(roots), axis=0), np.mean(np.stack(centroids), axis=0)
    position: Any = camera.get("position") or {}
    target_info: Any = camera.get("target") or {}
    position = np.array(
        [float(position.get("x", 0)), -float(position.get("y", 5)), -float(position.get("z", 0))],
        dtype=np.float32,
    )
    target_vector = np.array(
        [
            float(target_info.get("x", 0)),
            -float(target_info.get("y", 0)),
            -float(target_info.get("z", 0)),
        ],
        dtype=np.float32,
    )
    offset = position - target_vector
    has_offset = float(np.linalg.norm(offset)) >= 1e-6
    quaternion = camera.get("quaternion")
    if not has_offset and not quaternion:
        return pose
    zoom = float(camera.get("zoom", 1.0)) or 1.0
    target = centroid if float(np.linalg.norm(target_vector)) < 1e-6 else root + target_vector
    if quaternion:
        rotation = _quat_to_mat(
            float(quaternion.get("w", 1)),
            float(quaternion.get("x", 0)),
            float(quaternion.get("y", 0)),
            float(quaternion.get("z", 0)),
        )

        def image_axis(axis: Any) -> Any:
            axis = np.array([axis[0], -axis[1], -axis[2]], dtype=np.float32)
            return axis / max(1e-6, float(np.linalg.norm(axis)))

        x_axis, y_axis, z_axis = (
            image_axis(rotation[:, 0]),
            image_axis(-rotation[:, 1]),
            image_axis(-rotation[:, 2]),
        )
    else:
        z_axis = -offset / float(np.linalg.norm(offset))
        x_axis = np.cross(z_axis, np.array([0, -1, 0], dtype=np.float32))
        norm = float(np.linalg.norm(x_axis))
        x_axis = x_axis / norm if norm > 1e-6 else np.array([1, 0, 0], dtype=np.float32)
        y_axis = np.cross(z_axis, x_axis)
    rotation = np.stack([x_axis, y_axis, z_axis]).astype(np.float32)
    if has_offset:
        eye = target + offset / max(0.01, zoom)
    else:
        eye = target - z_axis * (max(0.1, float(target[2])) / max(0.01, zoom))
    fov = float(camera.get("fov", 0) or 0)
    if fov > 0:
        focal = float(height) / (2.0 * float(np.tan(np.deg2rad(fov) / 2.0)))
    else:
        original_focal = first[0].get("focal_length")
        focal = (
            float(np.asarray(original_focal, dtype=np.float32).reshape(-1)[0])
            if original_focal is not None
            else float(height) / (2.0 * float(np.tan(np.deg2rad(50.0) / 2.0)))
        )
    center = np.array([width * 0.5, height * 0.5], dtype=np.float32)
    projected = {
        "pred_keypoints_3d": "pred_keypoints_2d",
        "pred_face_keypoints_3d": "pred_face_keypoints_2d",
    }
    override = pose.get("_skeleton_override")
    joints_y_up = override is not None and not bool(override.get("per_frame_y_down", False))
    frames: list[list[dict[str, Any]]] = []
    for frame in pose["frames"]:
        transformed: list[dict[str, Any]] = []
        for person in frame:
            person = dict(person)
            translation = person.get("pred_cam_t")
            if translation is None:
                transformed.append(person)
                continue
            translation = np.asarray(translation, dtype=np.float32).reshape(3)
            for key in ("pred_keypoints_3d", "pred_vertices", "pred_face_keypoints_3d"):
                value = person.get(key)
                if value is None:
                    continue
                points = (np.asarray(value, dtype=np.float32) + translation - eye) @ rotation.T
                person[key] = points.astype(np.float32)
                if key in projected:
                    z = np.maximum(points[..., 2:3], 1e-6)
                    person[projected[key]] = (points[..., :2] * focal / z + center).astype(
                        np.float32
                    )
            joints = person.get("pred_joint_coords")
            if joints is not None:
                joints = np.asarray(joints, dtype=np.float32).copy()
                if joints_y_up:
                    joints[..., 1:3] *= -1.0
                joints = (joints + translation - eye) @ rotation.T
                if joints_y_up:
                    joints[..., 1:3] *= -1.0
                person["pred_joint_coords"] = joints.astype(np.float32)
            person["pred_cam_t"] = np.zeros(3, dtype=np.float32)
            person["focal_length"] = np.array(focal, dtype=np.float32)
            transformed.append(person)
        frames.append(transformed)
    return dict(pose, frames=frames)
