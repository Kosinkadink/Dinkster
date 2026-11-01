"""Temporal smoothing of canonical SAM3D Body pose packets."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    MHR_POSE_DATA_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_vision_packet,
)


class SAM3DBody_Smooth(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SAM3DBody_Smooth",
            display_name="Smooth SAM3D Body Pose Data",
            description="Reduce frame-to-frame jitter via temporal smoothing.",
            category="image/detection",
            inputs=(
                InputSpec("mhr_pose_data", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.05),
                    doc="Smoothing strength. 0 = raw, 1 = smoothed.",
                ),
                InputSpec(
                    "method",
                    TypeExpr.concrete(CORE_COMBO),
                    default="savgol",
                    widget=ComboWidget(("gaussian", "savgol")),
                    doc="gaussian: symmetric weighted average, best general-purpose smoother.\n"
                    "savgol: sliding polynomial fit, preserves sharp peaks.",
                ),
                InputSpec(
                    "window",
                    TypeExpr.concrete(CORE_INT),
                    default=7,
                    widget=NumberWidget(min=1, max=51, step=2),
                    advanced=True,
                    doc="Temporal window in frames (odd values).",
                ),
                InputSpec(
                    "rotation_threshold_degrees",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=30.0,
                    widget=NumberWidget(min=0.0, max=90.0, step=1.0),
                    advanced=True,
                    doc="Disables smoothing for this root rotation rate (degree/frame) to preserve "
                    "fast spins. 30 degrees suits most content, low values might disable smoothing "
                    "on ordinary jitter and silently impacts quality. 0 = disable.",
                ),
            ),
            outputs=(OutputSpec("mhr_pose_data", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        mhr_pose_data: Any,
        strength: float,
        method: str,
        window: int,
        rotation_threshold_degrees: float,
    ) -> Mapping[str, object]:
        if strength <= 0.0 or window <= 1:
            return cls.outputs(mhr_pose_data=mhr_pose_data)
        pose = materialize_vision_packet(mhr_pose_data)
        frames = pose["frames"]
        count = len(frames)
        if count < 2:
            return cls.outputs(mhr_pose_data=mhr_pose_data)
        max_people = max((len(frame) for frame in frames), default=0)
        if max_people == 0:
            return cls.outputs(mhr_pose_data=mhr_pose_data)
        np = importlib.import_module("numpy")
        geometry_keys = {
            "pred_vertices",
            "pred_keypoints_3d",
            "pred_joint_coords",
            "pred_global_rots",
            "mhr_model_params",
            "body_pose_params",
            "global_rot",
            "pred_pose_raw",
        }
        camera_keys = {
            "pred_cam_t",
            "pred_keypoints_2d",
            "focal_length",
            "shape_params",
            "scale_params",
            "hand_pose_params",
            "expr_params",
        }
        kernel = _smoothing_kernel(method, window)
        smoothed = [list(frame) for frame in frames]
        base_blend = float(strength)
        threshold = float(np.deg2rad(max(0.0, rotation_threshold_degrees)))
        for person in range(max_people):
            valid = np.array([person < len(frame) for frame in frames], dtype=bool)
            if valid.sum() < 2:
                continue
            geometry_blend = np.full(count, base_blend, dtype=np.float32)
            if threshold > 0.0:
                rotations: list[Any] = []
                valid_root: list[bool] = []
                for index in range(count):
                    root = frames[index][person].get("global_rot") if valid[index] else None
                    if root is None:
                        rotations.append(np.eye(3, dtype=np.float32))
                        valid_root.append(False)
                        continue
                    euler = np.asarray(root, dtype=np.float32).reshape(3)
                    cz, sz = np.cos(euler[0]), np.sin(euler[0])
                    cy, sy = np.cos(euler[1]), np.sin(euler[1])
                    cx, sx = np.cos(euler[2]), np.sin(euler[2])
                    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
                    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
                    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
                    rotations.append(rz @ ry @ rx)
                    valid_root.append(True)
                angles = np.zeros(count, dtype=np.float32)
                for index in range(1, count):
                    if valid_root[index] and valid_root[index - 1]:
                        delta = rotations[index] @ rotations[index - 1].T
                        cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
                        angles[index] = abs(np.arccos(cosine))
                # Neighbors must also back off, or averaging still flattens a spinning mesh.
                half = max(1, window // 2)
                peaks = np.zeros_like(angles)
                for index in range(count):
                    lo, hi = max(0, index - half), min(count, index + half + 1)
                    peaks[index] = angles[lo:hi].max() if hi > lo else 0.0
                geometry_blend = base_blend * (1.0 - np.clip(peaks / threshold, 0.0, 1.0))
            for key in sorted(geometry_keys | camera_keys):
                reference = None
                for index in range(count):
                    if valid[index] and frames[index][person].get(key) is not None:
                        reference = np.asarray(frames[index][person][key])
                        break
                if reference is None or reference.dtype.kind not in "fiu":
                    continue
                stacked = np.zeros((count,) + reference.shape, dtype=np.float32)
                for index in range(count):
                    if valid[index] and frames[index][person].get(key) is not None:
                        stacked[index] = np.asarray(frames[index][person][key], dtype=np.float32)
                    else:
                        stacked[index] = stacked[index - 1] if index > 0 else 0.0
                filter_input = _unwrap_pose_angles(key, stacked)
                filtered = _apply_temporal_filter(filter_input, kernel)
                blend_input = (
                    filter_input
                    if key in {"global_rot", "body_pose_params", "mhr_model_params"}
                    else stacked
                )
                if key in geometry_keys:
                    blend = geometry_blend
                    while blend.ndim < stacked.ndim:
                        blend = blend[..., None]
                    output = (1.0 - blend) * blend_input + blend * filtered
                else:
                    output = (1.0 - base_blend) * blend_input + base_blend * filtered
                if key == "pred_global_rots":
                    output = _project_rotation_matrices(output)
                for index in range(count):
                    if valid[index]:
                        smoothed[index][person] = dict(smoothed[index][person])
                        smoothed[index][person][key] = output[index].astype(reference.dtype)
        result = dict(pose)
        result["frames"] = smoothed
        return cls.outputs(mhr_pose_data=result)


def _smoothing_kernel(method: str, window: int) -> Any:
    np = importlib.import_module("numpy")
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if method == "savgol":
        order = 3 if window >= 5 else min(window - 1, 1)
        return (
            importlib.import_module("scipy.signal").savgol_coeffs(window, order).astype(np.float32)
        )
    sigma = max(1.0, window / 5.0)
    x = np.arange(window) - (window - 1) / 2.0
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return (kernel / kernel.sum()).astype(np.float32)


def _apply_temporal_filter(stacked: Any, kernel: Any) -> Any:
    np = importlib.import_module("numpy")
    count = stacked.shape[0]
    pad = len(kernel) // 2
    flat = stacked.reshape(count, -1)
    padded = np.concatenate(
        [np.repeat(flat[:1], pad, axis=0), flat, np.repeat(flat[-1:], pad, axis=0)],
        axis=0,
    )
    output = np.zeros_like(flat)
    for index, weight in enumerate(kernel):
        output += weight * padded[index : index + count]
    return output.reshape(stacked.shape)


def _unwrap_pose_angles(key: str, values: Any) -> Any:
    np = importlib.import_module("numpy")
    output = values.copy()
    if key == "global_rot":
        output = np.unwrap(output, axis=0)
    elif key == "body_pose_params":
        output[..., :124] = np.unwrap(output[..., :124], axis=0)
    elif key == "mhr_model_params":
        output[..., 3:130] = np.unwrap(output[..., 3:130], axis=0)
    return output


def _project_rotation_matrices(values: Any) -> Any:
    np = importlib.import_module("numpy")
    matrices = values.reshape(-1, 3, 3)
    u, _, vh = np.linalg.svd(matrices)
    rotations = u @ vh
    u[np.linalg.det(rotations) < 0, :, -1] *= -1
    return (u @ vh).reshape(values.shape).astype(values.dtype, copy=False)
