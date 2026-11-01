"""Rig-independent OpenPose and SCAIL skinned GLB animation."""

from __future__ import annotations

import importlib
from typing import Any

from .sam3d_animation import collect_tracks, quat_sign_fix_per_joint, rotmat_to_quat, unflip
from .sam3d_glb import GLBWriter, make_lit_material, rotation_align, smooth_time
from .sam3d_markers import spheres, sticks
from .sam3d_pose import (
    OPENPOSE_18_PAIRS,
    OPENPOSE_HAND_COLORS_21,
    OPENPOSE_HAND_PAIRS,
    OPENPOSE_RAINBOW_18,
    SCAIL_LIMB_COLORS_17,
    openpose_render_keypoints,
)

np = importlib.import_module("numpy")

_BODY_NAMES = (
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
)
_HAND_NAMES = (
    "wrist",
    *(
        f"{finger}{joint}"
        for finger in ("thumb", "index", "middle", "ring", "pinky")
        for joint in range(1, 5)
    ),
)
_FACE_TARGETS = (
    ("r_brow_outer", (-0.058, 1.690, 0.090)),
    ("r_brow_mid", (-0.040, 1.695, 0.105)),
    ("r_brow_inner", (-0.020, 1.692, 0.115)),
    ("l_brow_inner", (0.020, 1.692, 0.115)),
    ("l_brow_mid", (0.040, 1.695, 0.105)),
    ("l_brow_outer", (0.058, 1.690, 0.090)),
    ("r_eye_outer", (-0.058, 1.660, 0.085)),
    ("r_eye_top", (-0.040, 1.673, 0.090)),
    ("r_eye_inner", (-0.022, 1.665, 0.092)),
    ("r_eye_bot", (-0.040, 1.652, 0.090)),
    ("l_eye_outer", (0.058, 1.660, 0.085)),
    ("l_eye_top", (0.040, 1.673, 0.090)),
    ("l_eye_inner", (0.022, 1.665, 0.092)),
    ("l_eye_bot", (0.040, 1.652, 0.090)),
    ("nose_bridge", (0, 1.660, 0.110)),
    ("nose_mid", (0, 1.620, 0.125)),
    ("nose_tip", (0, 1.585, 0.135)),
    ("nostril_r", (-0.014, 1.580, 0.115)),
    ("nostril_l", (0.014, 1.580, 0.115)),
    ("mouth_r_corner", (-0.030, 1.540, 0.105)),
    ("upper_lip_mid", (0, 1.555, 0.115)),
    ("mouth_l_corner", (0.030, 1.540, 0.105)),
    ("lower_lip_mid", (0, 1.530, 0.110)),
    ("chin", (0, 1.498, 0.108)),
    ("r_jaw_low", (-0.038, 1.512, 0.100)),
    ("r_jaw_mid", (-0.062, 1.535, 0.080)),
    ("r_jaw_high", (-0.078, 1.562, 0.060)),
    ("l_jaw_low", (0.038, 1.512, 0.100)),
    ("l_jaw_mid", (0.062, 1.535, 0.080)),
    ("l_jaw_high", (0.078, 1.562, 0.060)),
)


def _keypoints(pose: Any, indices: list[int], person: int, part: str) -> Any:
    size = 18 if part == "body" else 21
    override: Any = pose.get("_skeleton_override") or {}
    key = {"body": "openpose18", "hand_r": "openpose_hand21_r", "hand_l": "openpose_hand21_l"}[part]
    mapping = override.get(key + "_joint_indices")
    if mapping is not None and np.asarray(mapping).shape != (size, 2):
        raise ValueError(f"build_glb_openpose: {key}_joint_indices must be shape ({size}, 2)")
    weights = override.get(key + "_joint_weights")
    if weights is not None and np.asarray(weights).shape != (size,):
        raise ValueError(f"build_glb_openpose: {key}_joint_weights must be shape ({size},)")
    if part != "body" and any(
        override.get(f"openpose_hand21_{side}_joint_indices") is not None for side in ("r", "l")
    ):
        if any(
            override.get(f"openpose_hand21_{side}_joint_indices") is None for side in ("r", "l")
        ):
            raise ValueError("build_glb_openpose: external skeleton must supply BOTH hand mappings")
    output = np.zeros((len(indices), size, 3), dtype=np.float32)
    for index, frame in enumerate(indices):
        points = openpose_render_keypoints(pose["frames"][frame][person], pose, part, dim=3)
        if points is None:
            raise ValueError(
                f"build_glb_openpose: missing {part} keypoints at frame={frame}, track={person}"
            )
        output[index] = unflip(points)
    return output


def _rest_points(pose: Any, parts: list[str], face_ids: Any) -> Any:
    override = pose.get("_skeleton_override")
    if override is None or "bind_global_m" not in override:
        return None
    keys = {"body": "openpose18", "hand_r": "openpose_hand21_r", "hand_l": "openpose_hand21_l"}
    if any(override.get(keys[part] + "_joint_indices") is None for part in parts):
        return None
    person = {"pred_joint_coords": np.asarray(override["bind_global_m"], dtype=np.float32)[:, :3]}
    chunks = [unflip(openpose_render_keypoints(person, pose, part, dim=3)) for part in parts]
    if face_ids is not None:
        rest = override.get("rest_verts_m")
        if rest is None:
            return None
        chunks.append(np.asarray(rest, dtype=np.float32)[face_ids])
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _neck_stub(points: Any) -> Any:
    result = points.copy()
    neck, nose = points[..., 1, :], points[..., 0, :]
    midhip = 0.5 * (points[..., 8, :] + points[..., 11, :])

    def unit(value: Any) -> Any:
        return value / np.linalg.norm(value, axis=-1, keepdims=True).clip(min=1e-6)

    direction = nose - neck
    mixed = unit(0.6 * unit(neck - midhip) + 0.4 * unit(direction))
    result[..., 0, :] = neck + mixed * (np.linalg.norm(direction, axis=-1, keepdims=True) * 0.5)
    return result


def _limb_motion(bind: Any, sequence: Any, pairs: Any) -> tuple[Any, Any, Any]:
    rest = np.zeros((len(pairs), 3), dtype=np.float32)
    centers = np.zeros((len(sequence), len(pairs), 3), dtype=np.float32)
    matrices = np.tile(np.eye(3, dtype=np.float32), (len(sequence), len(pairs), 1, 1))
    for index, (a, b) in enumerate(pairs):
        rest[index] = 0.5 * (bind[a] + bind[b])
        direction = bind[b] - bind[a]
        length = float(np.linalg.norm(direction))
        axis = direction / length if length > 1e-9 else np.array([0, 1, 0], dtype=np.float32)
        for t in range(len(sequence)):
            centers[t, index] = 0.5 * (sequence[t, a] + sequence[t, b])
            direction = sequence[t, b] - sequence[t, a]
            length = float(np.linalg.norm(direction))
            if length > 1e-9:
                matrices[t, index] = rotation_align(axis, direction / length)
    return rest, centers, rotmat_to_quat(matrices).astype(np.float32)


def build_glb_openpose(
    pose_data: Any,
    *,
    fps: float = 24.0,
    camera_translation: str = "off",
    track_index: int = -1,
    marker_radius_m: float = 0.025,
    stick_radius_m: float = 0.008,
    include_hands: bool = False,
    hand_marker_radius_m: float = 0.0,
    hand_stick_radius_m: float = 0.0,
    hand_color_style: str = "dwpose",
    face_style: str = "disabled",
    face_marker_radius_m: float = 0.0,
    palette: str = "openpose",
    shape: str = "ellipsoid",
    smooth_shade: bool = False,
    material_roughness: float = 0.85,
    material_double_sided: bool = False,
    stick_end_width_frac: float = 0.6,
    bone_smooth_window: int = 0,
) -> bytes:
    scail = palette == "scail"
    if palette not in ("openpose", "scail"):
        raise ValueError(f"build_glb_openpose: unknown palette={palette!r}")
    body_pairs = OPENPOSE_18_PAIRS[:13] if scail else OPENPOSE_18_PAIRS
    sphere_indices = np.arange(14 if scail else 18, dtype=np.int64)
    sphere_colors = OPENPOSE_RAINBOW_18
    stick_colors = OPENPOSE_RAINBOW_18
    if scail:
        sphere_colors = np.tile(np.array([0.6, 0.6, 0.6], dtype=np.float32), (18, 1))
        for index, (_, b) in enumerate(OPENPOSE_18_PAIRS):
            if (sphere_colors[b] == 0.6).all():
                sphere_colors[b] = SCAIL_LIMB_COLORS_17[index]
        stick_colors = SCAIL_LIMB_COLORS_17
    hand_marker_radius_m = (
        hand_marker_radius_m if hand_marker_radius_m > 0 else marker_radius_m * 0.4
    )
    hand_stick_radius_m = hand_stick_radius_m if hand_stick_radius_m > 0 else stick_radius_m * 0.5
    face_marker_radius_m = (
        face_marker_radius_m if face_marker_radius_m > 0 else marker_radius_m * 0.3
    )
    if hand_color_style == "dwpose":
        hand_colors = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (21, 1))
    elif hand_color_style == "openpose":
        hand_colors = OPENPOSE_HAND_COLORS_21
    else:
        raise ValueError(f"build_glb_openpose: unknown hand_color_style={hand_color_style!r}")
    tracks = collect_tracks(pose_data, track_index)
    if not tracks:
        raise ValueError("build_glb_openpose: no valid tracks in pose_data")
    face_ids = None
    targets = list(range(len(_FACE_TARGETS)))
    if face_style in ("full", "eyes_mouth"):
        canonical: Any = pose_data.get("canonical_colors") or {}
        if canonical.get("positions") is None:
            raise ValueError("build_glb_openpose: face_style needs canonical_colors.positions")
        positions = np.asarray(canonical["positions"], dtype=np.float32).reshape(-1, 3)
        mask = canonical.get("face_mask")
        valid = (
            np.where(np.asarray(mask).reshape(-1))[0]
            if mask is not None and np.asarray(mask).any()
            else np.where(
                (positions[:, 1] > 1.47)
                & (np.abs(positions[:, 0]) < 0.11)
                & (positions[:, 2] > 0.04)
            )[0]
        )
        if valid.size == 0:
            raise ValueError(
                "select_face_landmark_vert_ids: no head verts matched the canonical filter"
            )
        if face_style == "eyes_mouth":
            targets = [6, 7, 8, 9, 10, 11, 12, 13, 19, 20, 21, 22]
        face_ids = np.array(
            [
                valid[
                    int(
                        np.sum(
                            (positions[valid] - np.asarray(_FACE_TARGETS[i][1], dtype=np.float32))
                            ** 2,
                            axis=1,
                        ).argmin()
                    )
                ]
                for i in targets
            ],
            dtype=np.int64,
        )
    elif face_style != "disabled":
        raise ValueError(f"build_glb_openpose: unknown face_style={face_style!r}")
    parts = ["body", "hand_r", "hand_l"] if include_hands else ["body"]
    count = 18 + (42 if include_hands else 0) + (len(face_ids) if face_ids is not None else 0)
    names = [f"openpose_{name}" for name in _BODY_NAMES]
    limb_names = [f"openpose_limb_{_BODY_NAMES[a]}_{_BODY_NAMES[b]}" for a, b in body_pairs]
    if include_hands:
        for side in ("R", "L"):
            names.extend(f"openpose_{side}_{name}" for name in _HAND_NAMES)
            limb_names.extend(
                f"openpose_{side}hand_limb_{_HAND_NAMES[a]}_{_HAND_NAMES[b]}"
                for a, b in OPENPOSE_HAND_PAIRS
            )
    if face_ids is not None:
        names.extend(f"openpose_face_{_FACE_TARGETS[i][0]}" for i in targets)
    writer = GLBWriter()
    nodes: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    skins: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    roots: list[int] = []
    samplers: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []

    def animate(node: int, path: str, values: Any, kind: str, time: int) -> None:
        values = values.astype(np.float32)
        if (np.ptp(values, axis=0) < 1e-6).all():
            nodes[node][path] = values[0].tolist()
        else:
            output = writer.add(values, kind)
            samplers.append({"input": time, "output": output, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})

    for track, (person, frames) in enumerate(tracks):
        chunks = [_keypoints(pose_data, frames, person, part) for part in parts]
        if face_ids is not None:
            chunks.append(
                unflip(
                    np.stack(
                        [
                            np.asarray(
                                pose_data["frames"][t][person]["pred_vertices"], dtype=np.float32
                            ).reshape(-1, 3)[face_ids]
                            for t in frames
                        ]
                    )
                )
            )
        sequence = np.concatenate(chunks, axis=1)
        if bone_smooth_window > 1 and len(sequence) >= 2:
            sequence = smooth_time(sequence, bone_smooth_window).astype(np.float32)
        rest_points = _rest_points(pose_data, parts, face_ids)
        bind = rest_points if rest_points is not None else sequence[0].astype(np.float32)
        if scail:
            sequence[:, :18] = _neck_stub(sequence[:, :18])
            bind[:18] = _neck_stub(bind[:18])
        root_index = len(nodes)
        root: dict[str, Any] = {"name": f"track{track:02d}", "children": []}
        nodes.append(root)
        roots.append(root_index)
        joints: list[int] = []
        for index, name in enumerate(names):
            joints.append(len(nodes))
            nodes.append(
                {
                    "name": name,
                    "translation": bind[index].tolist(),
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                }
            )
        root["children"].extend(joints)
        motions = [_limb_motion(bind[:18], sequence[:, :18], body_pairs)]
        if include_hands:
            motions.extend(
                _limb_motion(
                    bind[offset : offset + 21],
                    sequence[:, offset : offset + 21],
                    OPENPOSE_HAND_PAIRS,
                )
                for offset in (18, 39)
            )
        rest = np.concatenate([motion[0] for motion in motions], axis=0)
        centers = np.concatenate([motion[1] for motion in motions], axis=1)
        rotations = quat_sign_fix_per_joint(
            np.concatenate([motion[2] for motion in motions], axis=1)
        ).astype(np.float32)
        limb_joints: list[int] = []
        for index, name in enumerate(limb_names):
            limb_joints.append(len(nodes))
            nodes.append(
                {
                    "name": name,
                    "translation": rest[index].tolist(),
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                }
            )
        root["children"].extend(limb_joints)
        inverse = np.tile(np.eye(4, dtype=np.float32), (count + len(rest), 1, 1))
        inverse[:count, :3, 3], inverse[count:, :3, 3] = -bind, -rest
        skins.append(
            {
                "joints": joints + limb_joints,
                "inverseBindMatrices": writer.add(
                    inverse.transpose(0, 2, 1).astype(np.float32), "MAT4"
                ),
                "skeleton": root_index,
            }
        )
        groups = [
            spheres(
                bind[sphere_indices],
                marker_radius_m,
                sphere_colors[sphere_indices],
                sphere_indices,
                smooth_shade,
            ),
            sticks(
                bind[:18],
                body_pairs,
                stick_radius_m,
                stick_colors,
                count,
                shape,
                smooth_shade,
                stick_end_width_frac,
            ),
        ]
        if include_hands:
            pair_colors = OPENPOSE_HAND_COLORS_21[[b for _, b in OPENPOSE_HAND_PAIRS]]
            for hand, offset in enumerate((18, 39)):
                groups.append(
                    spheres(
                        bind[offset : offset + 21],
                        hand_marker_radius_m,
                        hand_colors,
                        np.arange(offset, offset + 21),
                        smooth_shade,
                    )
                )
                groups.append(
                    sticks(
                        bind[offset : offset + 21],
                        OPENPOSE_HAND_PAIRS,
                        hand_stick_radius_m,
                        pair_colors,
                        count + len(body_pairs) + hand * len(OPENPOSE_HAND_PAIRS),
                        shape,
                        smooth_shade,
                        stick_end_width_frac,
                    )
                )
        if face_ids is not None:
            offset = 60 if include_hands else 18
            groups.append(
                spheres(
                    bind[offset:],
                    face_marker_radius_m,
                    np.ones((len(face_ids), 3), dtype=np.float32),
                    np.arange(offset, count),
                    smooth_shade,
                )
            )
        primitives: list[dict[str, Any]] = []
        for vertices, normals, faces, skin_joints, weights, colors in groups:
            if not len(vertices):
                continue
            attributes = {
                "POSITION": writer.add(vertices, "VEC3", target=34962, bounds=True),
                "NORMAL": writer.add(normals, "VEC3", target=34962, bounds=True),
                "JOINTS_0": writer.add(skin_joints, "VEC4", dtype="uint16", target=34962),
                "WEIGHTS_0": writer.add(weights, "VEC4", target=34962),
                "COLOR_0": writer.add(colors, "VEC3", target=34962, bounds=True),
            }
            materials.append(make_lit_material(material_roughness, material_double_sided))
            primitives.append(
                {
                    "attributes": attributes,
                    "indices": writer.add(faces, "SCALAR", dtype="uint32", target=34963),
                    "mode": 4,
                    "material": len(materials) - 1,
                }
            )
        if not primitives:
            continue
        meshes.append({"primitives": primitives})
        nodes.append(
            {"name": f"track{track:02d}_openpose", "mesh": len(meshes) - 1, "skin": len(skins) - 1}
        )
        root["children"].append(len(nodes) - 1)
        time_accessor = writer.add(
            np.asarray(frames, dtype=np.float32) / float(fps), "SCALAR", bounds=True
        )

        for index, node in enumerate(joints):
            animate(node, "translation", sequence[:, index], "VEC3", time_accessor)
        for index, node in enumerate(limb_joints):
            animate(node, "translation", centers[:, index], "VEC3", time_accessor)
            animate(node, "rotation", rotations[:, index], "VEC4", time_accessor)
        if camera_translation != "off":
            camera = np.stack(
                [
                    unflip(
                        np.asarray(pose_data["frames"][t][person]["pred_cam_t"], dtype=np.float32)
                    )
                    for t in frames
                ]
            )
            if camera_translation == "centered":
                camera = camera - camera[0:1]
            animate(root_index, "translation", camera, "VEC3", time_accessor)
    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "ComfyUI-SAM3DBody"},
        "scene": 0,
        "scenes": [{"nodes": roots}],
        "nodes": nodes,
        "meshes": meshes,
        "skins": skins,
    }
    if materials:
        gltf["materials"] = materials
    if samplers:
        gltf["animations"] = [{"name": "all_tracks", "samplers": samplers, "channels": channels}]
    return writer.to_bytes(gltf)
