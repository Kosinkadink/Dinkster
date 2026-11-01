"""Serialize SAM3D and external skeletal motion as BVH Z-X-Y channels."""

from __future__ import annotations

import importlib
from io import StringIO
from typing import Any

from .sam3d_animation import (
    bind_skeleton,
    bone_locals_from_globals,
    collect_tracks,
    global_skel_state_from_pose_data,
    quat_sign_fix_per_joint,
    unflip,
)

np = importlib.import_module("numpy")


def _quat_to_zxy_euler_deg(quaternion: Any) -> Any:
    q = np.asarray(quaternion, dtype=np.float64)
    x, y, z, w = (q[..., index] for index in range(4))
    angle_x = np.arcsin(np.clip(2.0 * (y * z + x * w), -1.0, 1.0))
    angle_z = np.arctan2(-2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z))
    angle_y = np.arctan2(-2.0 * (x * z - y * w), 1.0 - 2.0 * (x * x + y * y))
    return np.stack(
        [np.rad2deg(angle_z), np.rad2deg(angle_x), np.rad2deg(angle_y)], axis=-1
    ).astype(np.float32)


def build_bvh(
    pose_data: Any,
    model: Any = None,
    *,
    fps: float = 24.0,
    camera_translation: str = "off",
    track_index: int = -1,
    units: str = "cm",
) -> bytes:
    if units not in ("cm", "m"):
        raise ValueError(f"build_bvh: units must be 'cm' or 'm', got {units!r}")
    unit_scale = 100.0 if units == "cm" else 1.0
    parents, bind, offsets, y_down = bind_skeleton(pose_data, model)
    count = len(parents)
    tracks = collect_tracks(pose_data, track_index)
    if not tracks:
        raise ValueError("build_bvh: no valid tracks in pose_data")
    person, frame_indices = tracks[0]
    children: list[list[int]] = [[] for _ in range(count)]
    roots: list[int] = []
    for joint, parent in enumerate(parents):
        if 0 <= int(parent) < count and parent != joint:
            children[int(parent)].append(joint)
        else:
            roots.append(joint)
    root = roots[0] if roots else 0
    if roots and children[root]:
        external = "_skeleton_override" in pose_data
        if not (external and len(children[root]) > 1):
            root = children[root][0]
    order: list[int] = []

    def visit(joint: int) -> None:
        order.append(joint)
        for child in children[joint]:
            visit(child)

    visit(root)
    globals_ = global_skel_state_from_pose_data(
        pose_data, frame_indices, person, count, y_down=y_down
    )
    globals_[..., 3:7] = quat_sign_fix_per_joint(globals_[..., 3:7])
    bvh_parents = parents.copy()
    bvh_parents[root] = -1
    locals_ = bone_locals_from_globals(globals_, bvh_parents)
    locals_[..., 3:7] = quat_sign_fix_per_joint(locals_[..., 3:7])
    angles = _quat_to_zxy_euler_deg(locals_[..., 3:7])
    bind_positions = bind[:, :3].astype(np.float64) * 0.01
    offsets_m = offsets.astype(np.float64) * 0.01
    if camera_translation in ("absolute", "centered"):
        camera = np.stack(
            [
                unflip(np.asarray(pose_data["frames"][t][person]["pred_cam_t"], dtype=np.float32))
                for t in frame_indices
            ],
            axis=0,
        ).astype(np.float64)
        if camera_translation == "centered":
            camera -= camera[0:1]
        root_positions = bind_positions[root][None, :] + camera
    else:
        root_positions = np.tile(bind_positions[root], (len(frame_indices), 1))
    lines = ["HIERARCHY"]

    def emit(joint: int, depth: int, *, is_root: bool) -> None:
        indent = "  " * depth
        keyword, name = ("ROOT", "Hips") if is_root else ("JOINT", f"joint_{joint:03d}")
        lines.extend([f"{indent}{keyword} {name}", indent + "{"])
        offset = (bind_positions[joint] if is_root else offsets_m[joint]) * unit_scale
        lines.append(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
        channels = "6 Xposition Yposition Zposition " if is_root else "3 "
        lines.append(f"{indent}  CHANNELS {channels}Zrotation Xrotation Yrotation")
        if children[joint]:
            for child in children[joint]:
                emit(child, depth + 1, is_root=False)
        else:
            tip = offsets_m[joint] * unit_scale * 0.3
            if float(np.linalg.norm(tip)) < 0.5 * unit_scale * 0.01:
                tip = np.array([0.0, 0.05 * unit_scale, 0.0], dtype=np.float64)
            lines.extend(
                [
                    indent + "  End Site",
                    indent + "  {",
                    f"{indent}    OFFSET {tip[0]:.6f} {tip[1]:.6f} {tip[2]:.6f}",
                    indent + "  }",
                ]
            )
        lines.append(indent + "}")

    emit(root, 0, is_root=True)
    lines.extend(["MOTION", f"Frames: {len(frame_indices)}", f"Frame Time: {1.0 / fps:.6f}"])
    descendants = np.asarray(order[1:], dtype=np.int64)
    motion = np.concatenate(
        [
            root_positions * unit_scale,
            angles[:, root].astype(np.float64),
            angles[:, descendants, :].reshape(len(frame_indices), -1),
        ],
        axis=1,
    )
    buffer = StringIO()
    np.savetxt(buffer, motion, fmt="%.6f")
    lines.append(buffer.getvalue().rstrip("\n"))
    return ("\n".join(lines) + "\n").encode("utf-8")
