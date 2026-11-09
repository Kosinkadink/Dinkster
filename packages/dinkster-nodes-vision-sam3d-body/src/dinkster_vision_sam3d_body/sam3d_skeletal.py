"""Animated skinned body meshes, expression morphs and octahedron bones."""

from __future__ import annotations

import importlib
from typing import Any

from .sam3d_animation import (
    bind_skeleton,
    bone_locals_from_globals,
    collect_tracks,
    global_skel_state_from_pose_data,
    quat_sign_fix_per_joint,
    unflip,
)
from .sam3d_glb import (
    GLBWriter,
    compute_normals,
    flat_shade_mesh,
    inverse_bind_matrices,
    make_lit_material,
    rotation_align,
    smooth_time,
)
from .sam3d_rasterizer import build_vertex_colors

np = importlib.import_module("numpy")


def _compact_skin(rig: Any) -> tuple[Any, Any, int]:
    vertices = rig.lbs_vert_indices.cpu().numpy().astype(np.int64)
    indices = rig.lbs_skin_indices.cpu().numpy().astype(np.int64)
    weights = rig.lbs_skin_weights.cpu().numpy()
    count = int(rig.NUM_VERTS)
    joints = np.zeros((count, 8), dtype=np.uint16)
    result = np.zeros((count, 8), dtype=np.float32)
    counts = np.zeros(count, dtype=np.int32)
    if vertices.size:
        order = np.lexsort((-weights, vertices))
        sorted_vertices = vertices[order]
        starts = np.empty(len(order), dtype=bool)
        starts[0] = True
        np.not_equal(sorted_vertices[1:], sorted_vertices[:-1], out=starts[1:])
        positions = np.arange(len(order), dtype=np.int64)
        rank = positions - np.maximum.accumulate(np.where(starts, positions, 0))
        keep = rank < 8
        joints[sorted_vertices[keep], rank[keep]] = indices[order][keep].astype(np.uint16)
        result[sorted_vertices[keep], rank[keep]] = weights[order][keep].astype(np.float32)
        np.minimum(np.bincount(sorted_vertices, minlength=count), 8, out=counts, casting="unsafe")
    sums = result.sum(axis=1, keepdims=True)
    nonzero = sums.squeeze(-1) > 0
    result[nonzero] /= sums[nonzero]
    result[~nonzero, 0] = 1.0
    return joints, result, int(counts.max()) if count else 0


def _rest_vertices(pose: Any, model: Any, shape: Any) -> Any:
    override = pose.get("_skeleton_override")
    if override is not None:
        return np.asarray(override["rest_verts_m"], dtype=np.float32)
    torch = importlib.import_module("torch")
    inner = model.model if hasattr(model, "model") else model
    rig = inner.head_pose.mhr
    parameter = next(rig.parameters())
    device = rig.scale_mean.device if hasattr(rig, "scale_mean") else parameter.device
    shape_tensor = torch.from_numpy(np.ascontiguousarray(shape, dtype=np.float32)).to(device)
    if shape_tensor.ndim == 1:
        shape_tensor = shape_tensor.unsqueeze(0)
    params = torch.zeros(1, 204, device=device, dtype=parameter.dtype)
    expression = torch.zeros(1, 72, device=device, dtype=parameter.dtype)
    vertices, _ = rig(shape_tensor.to(parameter.dtype), params, expression, apply_correctives=False)
    return (vertices[0].cpu().float().numpy() / 100.0).astype(np.float32)


def _bone_mesh(positions: Any, parents: Any, radius: float) -> tuple[Any, ...]:
    base = np.array(
        [[0, 0, 0], [0, 1, 0], [1, 0.1, 0], [-1, 0.1, 0], [0, 0.1, 1], [0, 0.1, -1]],
        dtype=np.float32,
    )
    topology = np.array(
        [[0, 2, 4], [0, 5, 2], [0, 3, 5], [0, 4, 3], [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5]],
        dtype=np.uint32,
    )
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    joints: list[list[int]] = []
    weights: list[list[float]] = []
    children: list[int] = []
    for child, parent_value in enumerate(parents):
        parent = int(parent_value)
        if not 0 <= parent < len(parents) or parent == child:
            continue
        grandparent = int(parents[parent])
        if not 0 <= grandparent < len(parents) or grandparent == parent:
            continue
        head, tail = positions[parent].astype(np.float32), positions[child].astype(np.float32)
        direction = tail - head
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        rotation = rotation_align(np.array([0, 1, 0], dtype=np.float32), direction / length)
        width = max(0.001, min(length * 0.1, radius))
        local = base * np.array([width, length, width], dtype=np.float32)
        offset = len(vertices)
        vertices.extend((local @ rotation.T + head).tolist())
        faces.extend((topology + offset).tolist())
        for vertex in base:
            parent_weight, child_weight = (
                max(0.0, 1.0 - float(vertex[1])),
                max(0.0, float(vertex[1])),
            )
            total = parent_weight + child_weight
            if total > 0:
                parent_weight, child_weight = parent_weight / total, child_weight / total
            joints.append([parent, child, 0, 0])
            weights.append([parent_weight, child_weight, 0.0, 0.0])
            children.append(child)
    v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)
    j = np.asarray(joints, dtype=np.uint16).reshape(-1, 4)
    w = np.asarray(weights, dtype=np.float32).reshape(-1, 4)
    colors = np.asarray(children, dtype=np.int64)[f].reshape(-1)
    return (*flat_shade_mesh(v, f, j, w), colors)


def _bone_colors(positions: Any) -> Any:
    y = positions[:, 1].astype(np.float32)
    value = np.clip((y - float(y.min())) / max(float(y.max()) - float(y.min()), 1e-6), 0, 1)
    return np.stack(
        [
            np.interp(value, [0, 0.35, 0.66, 0.89, 1], [0, 0, 1, 1, 0.5]),
            np.interp(value, [0, 0.125, 0.375, 0.64, 0.91, 1], [0, 0, 1, 1, 0, 0]),
            np.interp(value, [0, 0.11, 0.34, 0.65, 1], [0.5, 1, 1, 0, 0]),
        ],
        axis=-1,
    ).astype(np.float32)


def build_glb_skeletal(
    pose_data: Any,
    model: Any = None,
    *,
    fps: float = 24.0,
    camera_translation: str = "off",
    track_index: int = -1,
    shader: str = "default",
    rainbow_tilt_x_deg: float = 0.0,
    rainbow_tilt_z_deg: float = 0.0,
    person_palette_falloff: float = 0.6,
    bone_smooth_window: int = 0,
    bone_vis: str = "off",
    bone_vis_radius_m: float = 0.04,
    bone_vis_color: str = "white",
    include_body_mesh: bool = True,
) -> bytes:
    frames = pose_data["frames"]
    faces = np.ascontiguousarray(pose_data["faces"], dtype=np.uint32)
    tracks = collect_tracks(pose_data, track_index)
    if not tracks:
        raise ValueError("build_glb_skeletal: no valid tracks in pose_data")
    parents, bind_cm, _, y_down = bind_skeleton(pose_data, model)
    override = pose_data.get("_skeleton_override")
    if override is not None:
        bind = np.asarray(override["bind_global_m"], dtype=np.float32)
    else:
        bind = bind_cm.astype(np.float32).copy()
        bind[:, :3] *= 0.01
    bind_local = bone_locals_from_globals(bind[None], parents)[0]
    if override is not None:
        joints = np.asarray(override["lbs_compact_joints"], dtype=np.uint16)
        weights = np.asarray(override["lbs_compact_weights"], dtype=np.float32)
        max_influences = int(override.get("lbs_compact_max_inf", 4))
        expression_basis = None
    else:
        inner = model.model if hasattr(model, "model") else model
        rig = inner.head_pose.mhr
        joints, weights, max_influences = _compact_skin(rig)
        expression_basis = rig.expr_basis.cpu().numpy() if int(rig.NUM_EXPR) > 0 else None
    writer = GLBWriter()
    nodes: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    skins: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    roots: list[int] = []
    samplers: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    indices = writer.add(faces, "SCALAR", dtype="uint32", target=34963)
    joints0 = writer.add(joints[:, :4], "VEC4", dtype="uint16", target=34962)
    weights0 = writer.add(weights[:, :4], "VEC4", target=34962)
    joints1 = (
        writer.add(joints[:, 4:8], "VEC4", dtype="uint16", target=34962)
        if max_influences > 4
        else None
    )
    weights1 = writer.add(weights[:, 4:8], "VEC4", target=34962) if max_influences > 4 else None
    inverse_bind = writer.add(inverse_bind_matrices(bind), "MAT4")
    morphs: list[int] = []
    if expression_basis is not None:
        for basis in expression_basis.astype(np.float32) * 0.01:
            morphs.append(writer.add(basis, "VEC3", target=34962))

    def animate(node: int, path: str, values: Any, time: int, kind: str) -> None:
        output = writer.add(values, kind)
        samplers.append({"input": time, "output": output, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})

    for track, (person, frame_indices) in enumerate(tracks):
        root_index = len(nodes)
        root: dict[str, Any] = {"name": f"track{track:02d}", "children": []}
        nodes.append(root)
        roots.append(root_index)
        bones = list(range(len(nodes), len(nodes) + len(parents)))
        for joint in range(len(parents)):
            nodes.append(
                {
                    "name": f"bone_{joint:03d}",
                    "translation": bind_local[joint, :3].tolist(),
                    "rotation": bind_local[joint, 3:7].tolist(),
                    "scale": [float(bind_local[joint, 7])] * 3,
                }
            )
        children: list[list[int]] = [[] for _ in parents]
        bone_roots: list[int] = []
        for joint, parent in enumerate(parents):
            if 0 <= int(parent) < len(parents) and parent != joint:
                children[int(parent)].append(bones[joint])
            else:
                bone_roots.append(bones[joint])
        for joint, descendants in enumerate(children):
            if descendants:
                nodes[bones[joint]]["children"] = descendants
        root["children"].extend(bone_roots)
        skin_index = len(skins)
        skins.append(
            {
                "joints": bones,
                "inverseBindMatrices": inverse_bind,
                "skeleton": bone_roots[0] if bone_roots else bones[0],
            }
        )
        mesh_node: int | None = None
        include_bones = bone_vis == "octahedrons"
        if include_body_mesh:
            shape = np.asarray(
                frames[frame_indices[0]][person].get("shape_params", []), dtype=np.float32
            )
            vertices = _rest_vertices(pose_data, model, shape)
            positions = writer.add(vertices, "VEC3", target=34962, bounds=True)
            normals = writer.add(
                compute_normals(vertices, faces), "VEC3", target=34962, bounds=True
            )
            color = None
            canonical = pose_data.get("canonical_colors")
            if shader != "default" and canonical is not None:
                color = build_vertex_colors(
                    canonical, shader, rainbow_tilt_x_deg, rainbow_tilt_z_deg
                )
                falloff = max(0.1, min(1.0, float(person_palette_falloff)))
                pastel = 0.0 if track == 0 else 1.0 - falloff**track
                if pastel > 0:
                    color = color * (1.0 - pastel) + pastel
                color = np.clip(color, 0, 1).astype(np.float32)
            attributes: dict[str, Any] = {
                "POSITION": positions,
                "NORMAL": normals,
                "JOINTS_0": joints0,
                "WEIGHTS_0": weights0,
            }
            if joints1 is not None:
                attributes["JOINTS_1"], attributes["WEIGHTS_1"] = joints1, weights1
            if color is not None:
                attributes["COLOR_0"] = writer.add(color, "VEC3", target=34962, bounds=True)
            primitive: dict[str, Any] = {"attributes": attributes, "indices": indices, "mode": 4}
            if color is not None or include_bones:
                materials.append(make_lit_material(opacity=0.35 if include_bones else 1.0))
                primitive["material"] = len(materials) - 1
            if morphs:
                primitive["targets"] = [{"POSITION": index} for index in morphs]
            mesh: dict[str, Any] = {"primitives": [primitive]}
            if morphs:
                mesh["weights"] = [0.0] * len(morphs)
            meshes.append(mesh)
            mesh_node = len(nodes)
            nodes.append(
                {"name": f"track{track:02d}_mesh", "mesh": len(meshes) - 1, "skin": skin_index}
            )
            root["children"].append(mesh_node)
        if include_bones:
            vertices, normals, topology, bone_joints, bone_weights, color_indices = _bone_mesh(
                bind[:, :3], parents, bone_vis_radius_m
            )
            if len(vertices):
                attributes = {
                    "POSITION": writer.add(vertices, "VEC3", target=34962, bounds=True),
                    "JOINTS_0": None,
                    "WEIGHTS_0": None,
                }
                bone_indices = writer.add(topology, "SCALAR", dtype="uint32", target=34963)
                attributes["JOINTS_0"] = writer.add(
                    bone_joints, "VEC4", dtype="uint16", target=34962
                )
                attributes["WEIGHTS_0"] = writer.add(bone_weights, "VEC4", target=34962)
                attributes["NORMAL"] = writer.add(normals, "VEC3", target=34962, bounds=True)
                if bone_vis_color == "rainbow_y":
                    color = _bone_colors(bind[:, :3])[color_indices].astype(np.float32)
                    attributes["COLOR_0"] = writer.add(color, "VEC3", target=34962, bounds=True)
                primitive = {"attributes": attributes, "indices": bone_indices, "mode": 4}
                if bone_vis_color == "rainbow_y":
                    materials.append(make_lit_material())
                    primitive["material"] = len(materials) - 1
                meshes.append({"primitives": [primitive]})
                nodes.append(
                    {"name": f"track{track:02d}_bones", "mesh": len(meshes) - 1, "skin": skin_index}
                )
                root["children"].append(len(nodes) - 1)
        globals_ = global_skel_state_from_pose_data(
            pose_data, frame_indices, person, len(parents), y_down=y_down
        )
        globals_[..., 3:7] = quat_sign_fix_per_joint(globals_[..., 3:7])
        locals_ = bone_locals_from_globals(globals_, parents)
        rotations = quat_sign_fix_per_joint(locals_[..., 3:7].astype(np.float32))
        sign = np.where((bind_local[:, 3:7] * rotations[0]).sum(axis=-1) < 0, -1.0, 1.0).astype(
            np.float32
        )[:, None]
        rotations[0] *= sign
        rotations = quat_sign_fix_per_joint(rotations)
        if bone_smooth_window > 1 and len(rotations) >= 2:
            rotations = smooth_time(quat_sign_fix_per_joint(rotations), bone_smooth_window)
            rotations = (
                rotations / np.maximum(np.linalg.norm(rotations, axis=-1, keepdims=True), 1e-12)
            ).astype(np.float32)
        rotations = rotations.astype(np.float64)
        rotations = (
            rotations / np.maximum(np.linalg.norm(rotations, axis=-1, keepdims=True), 1e-12)
        ).astype(np.float32)
        time = writer.add(
            np.asarray(frame_indices, dtype=np.float32) / float(fps), "SCALAR", bounds=True
        )
        for joint, node in enumerate(bones):
            scale = np.broadcast_to(locals_[:, joint, 7:8], (len(frame_indices), 3)).astype(
                np.float32
            )
            for path, values, kind in (
                ("translation", locals_[:, joint, :3].astype(np.float32), "VEC3"),
                ("rotation", rotations[:, joint], "VEC4"),
                ("scale", scale, "VEC3"),
            ):
                if (np.ptp(values, axis=0) < 1e-6).all():
                    nodes[node][path] = values[0].tolist()
                else:
                    animate(node, path, values, time, kind)
        if camera_translation != "off":
            camera = np.stack(
                [
                    unflip(np.asarray(frames[t][person]["pred_cam_t"], dtype=np.float32))
                    for t in frame_indices
                ]
            )
            if camera_translation == "centered":
                camera = camera - camera[0:1]
            if (np.ptp(camera, axis=0) < 1e-6).all():
                root["translation"] = camera[0].tolist()
            else:
                animate(root_index, "translation", camera, time, "VEC3")
        if morphs and mesh_node is not None:
            expression = np.stack(
                [
                    np.asarray(frames[t][person]["expr_params"], dtype=np.float32)
                    for t in frame_indices
                ]
            ).astype(np.float32)
            animate(mesh_node, "weights", expression, time, "SCALAR")
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
