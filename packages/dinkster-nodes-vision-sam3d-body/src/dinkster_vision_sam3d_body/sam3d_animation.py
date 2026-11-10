"""Track selection and rig transforms for SAM3D animation files."""

from __future__ import annotations

import importlib
from typing import Any

np = importlib.import_module("numpy")


def collect_tracks(pose_data: Any, track_index: int) -> list[tuple[int, list[int]]]:
    frames = pose_data["frames"]
    count = max((len(frame) for frame in frames), default=0)
    wanted = range(count) if track_index < 0 else [track_index]
    return [
        (person, indices)
        for person in wanted
        if (indices := [t for t, frame in enumerate(frames) if person < len(frame)])
    ]


def unflip(value: Any) -> Any:
    result = np.array(value, copy=True)
    result[..., 1:3] *= -1.0
    return result


def _quat_multiply(a: Any, b: Any) -> Any:
    x1, y1, z1, w1 = (a[..., i] for i in range(4))
    x2, y2, z2, w2 = (b[..., i] for i in range(4))
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def _quat_rotate(q: Any, value: Any) -> Any:
    cross = np.cross(q[..., :3], value, axis=-1)
    return value + 2.0 * (cross * q[..., 3:4] + np.cross(q[..., :3], cross, axis=-1))


def bone_locals_from_globals(globals_: Any, parents: Any) -> Any:
    result = np.zeros_like(globals_)
    for joint, parent in enumerate(parents):
        if not 0 <= int(parent) < len(parents) or parent == joint:
            result[:, joint] = globals_[:, joint]
            continue
        parent_state, child = globals_[:, parent], globals_[:, joint]
        q = parent_state[..., 3:7].copy()
        q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
        q[..., :3] *= -1.0
        scale = parent_state[..., 7:8]
        inverse_scale = 1.0 / np.where(np.abs(scale) > 1e-12, scale, 1.0)
        inverse_position = -inverse_scale * _quat_rotate(q, parent_state[..., :3])
        child_q = child[..., 3:7]
        normalized_q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
        child_q = child_q / np.maximum(np.linalg.norm(child_q, axis=-1, keepdims=True), 1e-12)
        result[:, joint] = np.concatenate(
            [
                inverse_position + inverse_scale * _quat_rotate(normalized_q, child[..., :3]),
                _quat_multiply(normalized_q, child_q),
                inverse_scale * child[..., 7:8],
            ],
            axis=-1,
        )
    return result


def quat_sign_fix_per_joint(sequence: Any) -> Any:
    result = np.array(sequence, dtype=np.float64, copy=True)
    result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)
    for t in range(1, result.shape[0]):
        dots = (result[t - 1] * result[t]).sum(axis=-1)
        result[t] *= np.where(dots < 0.0, -1.0, 1.0)[:, None]
    return result.astype(np.float32)


def rotmat_to_quat(matrix: Any) -> Any:
    shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3).astype(np.float64)
    result = np.zeros((len(m), 4), dtype=np.float64)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    positive = trace > 0
    if positive.any():
        values = m[positive]
        scale = np.sqrt(trace[positive] + 1.0) * 2.0
        result[positive, 3] = 0.25 * scale
        result[positive, 0] = (values[:, 2, 1] - values[:, 1, 2]) / scale
        result[positive, 1] = (values[:, 0, 2] - values[:, 2, 0]) / scale
        result[positive, 2] = (values[:, 1, 0] - values[:, 0, 1]) / scale
    largest_x = ~positive & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    largest_y = ~positive & ~largest_x & (m[:, 1, 1] > m[:, 2, 2])
    largest_z = ~positive & ~largest_x & ~largest_y
    for selected, axis in ((largest_x, 0), (largest_y, 1), (largest_z, 2)):
        if not selected.any():
            continue
        values = m[selected]
        a, b = (axis + 1) % 3, (axis + 2) % 3
        scale = np.sqrt(1.0 + values[:, axis, axis] - values[:, a, a] - values[:, b, b]) * 2.0
        result[selected, axis] = 0.25 * scale
        result[selected, 3] = (values[:, b, a] - values[:, a, b]) / scale
        result[selected, a] = (values[:, axis, a] + values[:, a, axis]) / scale
        result[selected, b] = (values[:, axis, b] + values[:, b, axis]) / scale
    return result.reshape(shape + (4,)).astype(np.float32)


def global_skel_state_from_pose_data(
    pose_data: Any, frame_indices: list[int], person: int, num_joints: int, *, y_down: bool
) -> Any:
    rotations = np.zeros((len(frame_indices), num_joints, 3, 3), dtype=np.float32)
    positions = np.zeros((len(frame_indices), num_joints, 3), dtype=np.float32)
    for index, t in enumerate(frame_indices):
        frame = pose_data["frames"][t][person]
        rotations[index] = np.asarray(frame["pred_global_rots"], dtype=np.float32)[:num_joints]
        positions[index] = np.asarray(frame["pred_joint_coords"], dtype=np.float32)[:num_joints]
    if y_down:
        positions[..., 1:3] *= -1.0
    state = np.zeros((len(frame_indices), num_joints, 8), dtype=np.float32)
    state[..., :3] = positions
    state[..., 3:7] = rotmat_to_quat(rotations)
    state[..., 7] = 1.0
    return state


def bind_skeleton(pose_data: Any, model: Any) -> tuple[Any, Any, Any, bool]:
    """Return parents, bind globals and local offsets in cm, and coordinate convention."""
    override = pose_data.get("_skeleton_override")
    if override is not None:
        parents = np.asarray(override["parents"], dtype=np.int32)
        bind = np.asarray(override["bind_global_m"], dtype=np.float32).copy()
        local = bone_locals_from_globals(bind[None], parents)[0]
        offsets = (local[:, :3] * 100.0).astype(np.float32)
        bind[:, :3] *= 100.0
        return parents, bind, offsets, bool(override.get("per_frame_y_down", False))
    torch = importlib.import_module("torch")
    inner = model.model if hasattr(model, "model") else model
    rig = inner.head_pose.mhr
    parameter = next(rig.parameters())
    # FK must use the model's own parameter transform and compute dtype.
    params = torch.from_numpy(np.zeros((1, 204), dtype=np.float32)).to(
        device=parameter.device, dtype=parameter.dtype
    )
    shape = torch.zeros(1, rig.NUM_IDENTITY, device=parameter.device, dtype=parameter.dtype)
    expression = torch.zeros(1, rig.NUM_EXPR, device=parameter.device, dtype=parameter.dtype)
    _, state = rig(shape, params, expression, apply_correctives=False)
    return (
        rig.skel_joint_parents.cpu().numpy().astype(np.int32),
        state.cpu().float().numpy()[0],
        rig.skel_joint_translation_offsets.cpu().numpy().astype(np.float32),
        True,
    )
