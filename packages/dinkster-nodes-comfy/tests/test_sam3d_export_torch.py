"""Animation file hierarchy, channel values and source preservation."""

import asyncio
import copy
import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    FILE_3D_TYPE,
    KIMODO_POSE_DATA_TYPE,
    MHR_POSE_DATA_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_vision_packet,
)
from dinkster_graph import Graph, GraphNode, Link, graph_from_wire, graph_to_wire
from dinkster_nodes_comfy.sam3d_bvh import build_bvh
from dinkster_nodes_comfy.sam3d_export import BuildPoseFile
from dinkster_nodes_comfy.sam3d_openpose_glb import build_glb_openpose
from dinkster_nodes_comfy.sam3d_skeletal import build_glb_skeletal
from scipy.spatial.transform import Rotation

from dinkster.compose import PackSpec, ServingComposer


def _pose(*, branching: bool = False) -> dict[str, Any]:
    parents = np.array([-1, 0, 0 if branching else 1], np.int32)
    positions = np.array([[0.2, 0.3, 0.4], [0.2, 1.3, 0.4], [0.5, 1.7, 0.4]], np.float32)
    bind = np.zeros((3, 8), np.float32)
    bind[:, :3], bind[:, 6:] = positions, 1
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy()
    first = {
        "pred_joint_coords": positions,
        "pred_global_rots": rotations,
        "pred_cam_t": np.array([1, 2, 3], np.float32),
    }
    second = copy.deepcopy(first)
    second["pred_cam_t"] = np.array([4, 7, 9], np.float32)
    moved = copy.deepcopy(first)
    moved["pred_cam_t"] = np.array([1.5, 2.25, 4], np.float32)
    return {
        "frames": [[first, second], [], [moved]],
        "_skeleton_override": {"parents": parents, "bind_global_m": bind},
    }


def _motion(data: bytes) -> tuple[str, Any]:
    text = data.decode("utf-8")
    header, values = text.split("Frame Time: ", 1)
    _, values = values.split("\n", 1)
    return header, np.loadtxt(values.splitlines(), ndmin=2)


@pytest.mark.parametrize(
    ("translation", "units", "expected"),
    [
        ("off", "cm", [[20, 130, 40], [20, 130, 40]]),
        ("absolute", "cm", [[120, -70, -260], [170, -95, -360]]),
        ("centered", "m", [[0.2, 1.3, 0.4], [0.7, 1.05, -0.6]]),
    ],
)
def test_bvh_camera_units_and_frame_selection(
    translation: str, units: str, expected: list[list[float]]
) -> None:
    pose = _pose()
    original = copy.deepcopy(pose)
    data = build_bvh(pose, camera_translation=translation, units=units, fps=30)
    header, motion = _motion(data)
    assert "ROOT Hips\n" in header and "JOINT joint_002\n" in header
    assert "JOINT joint_000" not in header and "Frames: 2\n" in header
    assert "Frame Time: 0.033333\n" in data.decode()
    np.testing.assert_allclose(motion[:, :3], expected, atol=1e-5, rtol=0)
    np.testing.assert_array_equal(motion[:, 3:], np.zeros((2, 6)))
    for frame, old_frame in zip(pose["frames"], original["frames"], strict=True):
        for person, old_person in zip(frame, old_frame, strict=True):
            for key in person:
                np.testing.assert_array_equal(person[key], old_person[key])
    for key in pose["_skeleton_override"]:
        np.testing.assert_array_equal(
            pose["_skeleton_override"][key], original["_skeleton_override"][key]
        )


def test_bvh_external_branching_root_and_selected_track() -> None:
    pose = _pose(branching=True)
    data = build_bvh(pose, track_index=1, camera_translation="absolute", units="m")
    header, motion = _motion(data)
    assert "Frames: 1\n" in header
    assert "JOINT joint_001\n" in header and "JOINT joint_002\n" in header
    np.testing.assert_allclose(motion[:, :3], [[4.2, -6.7, -8.6]], rtol=0, atol=1e-5)
    assert motion.shape == (1, 12)
    # Unlike a world-anchor chain, an external articulated root keeps both limbs.
    assert "OFFSET 0.000000 1.000000 0.000000" in header
    assert "OFFSET 0.300000 1.400000 0.000000" in header


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_bvh_rotation_channels_reconstruct_large_parent_and_child_rotations(axis: str) -> None:
    pose = _pose(branching=True)
    # Above 120 degrees forces the non-trace rotation-to-quaternion branches.
    root = Rotation.from_euler(axis, 170, degrees=True)
    child_local = Rotation.from_euler("ZXY", [23, -31, 47], degrees=True)
    global_matrices = np.stack(
        [root.as_matrix(), (root * child_local).as_matrix(), root.as_matrix()]
    ).astype(np.float32)
    pose["frames"][0][0]["pred_global_rots"] = global_matrices
    pose["frames"][2][0]["pred_global_rots"] = global_matrices
    _, motion = _motion(build_bvh(pose))
    root_decoded = Rotation.from_euler("ZXY", motion[0, 3:6], degrees=True)
    child_decoded = Rotation.from_euler("ZXY", motion[0, 6:9], degrees=True)
    np.testing.assert_allclose(root_decoded.as_matrix(), root.as_matrix(), atol=2e-7, rtol=0)
    np.testing.assert_allclose(
        child_decoded.as_matrix(), child_local.as_matrix(), atol=2e-7, rtol=0
    )


def test_bvh_mhr_bind_and_y_down_coordinates() -> None:
    pose = _pose()
    bind = pose.pop("_skeleton_override")["bind_global_m"]
    for frame in pose["frames"]:
        for person in frame:
            person["pred_joint_coords"][:, 1:3] *= -1

    class Rig:
        NUM_IDENTITY, NUM_EXPR = 45, 72
        skel_joint_parents = torch.tensor([-1, 0, 1], dtype=torch.int32)
        skel_joint_translation_offsets = torch.tensor(
            [[20, 30, 40], [0, 100, 0], [30, 40, 0]], dtype=torch.float32
        )

        def parameters(self) -> Any:
            yield torch.empty(0, dtype=torch.float64)

        def __call__(self, shape: Any, params: Any, expression: Any, **kwargs: Any) -> Any:
            # A non-default dtype checks that zero-pose FK honors the model policy.
            assert shape.dtype == params.dtype == expression.dtype == torch.float64
            state = torch.from_numpy(bind.copy()).unsqueeze(0)
            state[..., :3] *= 100
            return None, state

    model = SimpleNamespace(model=SimpleNamespace(head_pose=SimpleNamespace(mhr=Rig())))
    header, motion = _motion(build_bvh(pose, model))
    assert "JOINT joint_000" not in header
    np.testing.assert_allclose(motion[:, :3], [[20, 130, 40], [20, 130, 40]], atol=1e-5)
    np.testing.assert_array_equal(motion[:, 3:], np.zeros((2, 6)))


def test_bvh_invalid_units_and_missing_track() -> None:
    with pytest.raises(ValueError, match="units must be 'cm' or 'm'"):
        build_bvh(_pose(), units="mm")
    with pytest.raises(ValueError, match="no valid tracks"):
        build_bvh(_pose(), track_index=2)
    pose = _pose()
    pose["frames"] = []
    with pytest.raises(ValueError, match="no valid tracks"):
        build_bvh(pose)


def _glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    magic, version, length = struct.unpack_from("<4sII", data)
    assert (magic, version, length) == (b"glTF", 2, len(data))
    size, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A
    metadata = json.loads(data[20 : 20 + size])
    binary_size, binary_kind = struct.unpack_from("<II", data, 20 + size)
    assert binary_kind == 0x004E4942
    binary = data[28 + size :]
    assert binary_size == len(binary)
    assert all(view["byteOffset"] % 4 == 0 for view in metadata["bufferViews"])
    return metadata, binary


def _accessor(gltf: dict[str, Any], binary: bytes, index: int) -> Any:
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = {5126: "<f4", 5125: "<u4", 5123: "<u2"}[accessor["componentType"]]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[accessor["type"]]
    return np.frombuffer(
        binary, dtype=dtype, count=accessor["count"] * width, offset=view["byteOffset"]
    ).reshape(accessor["count"], width)


def _mesh_pose() -> dict[str, Any]:
    pose = _pose()
    pose["faces"] = np.array([[0, 1, 2]], np.uint32)
    pose["_skeleton_override"].update(
        rest_verts_m=np.array([[-0.5, 0, 0], [0.5, 0, 0], [0, 1, 0]], np.float32),
        lbs_compact_joints=np.zeros((3, 4), np.uint16),
        lbs_compact_weights=np.tile(np.array([1, 0, 0, 0], np.float32), (3, 1)),
    )
    return pose


@pytest.mark.parametrize(
    ("translation", "expected"),
    [("absolute", [[1, -2, -3], [1.5, -2.25, -4]]), ("centered", [[0, 0, 0], [0.5, -0.25, -1]])],
)
def test_skeletal_glb_bind_camera_animation_and_source_preservation(
    translation: str, expected: list[list[float]]
) -> None:
    pose = _mesh_pose()
    # Local translation must rotate back through the parent's animated frame.
    turn = Rotation.from_euler("Z", 37, degrees=True)
    pose["frames"][2][0]["pred_global_rots"][:] = turn.as_matrix()
    original = copy.deepcopy(pose)
    gltf, binary = _glb(
        build_glb_skeletal(pose, camera_translation=translation, fps=20, track_index=0)
    )
    animation = gltf["animations"][0]
    root_channel = next(
        c for c in animation["channels"] if c["target"] == {"node": 0, "path": "translation"}
    )
    sampler = animation["samplers"][root_channel["sampler"]]
    np.testing.assert_array_equal(
        _accessor(gltf, binary, sampler["input"]).ravel(), np.array([0, 0.1], np.float32)
    )
    np.testing.assert_allclose(_accessor(gltf, binary, sampler["output"]), expected, atol=1e-7)
    bind = original["_skeleton_override"]["bind_global_m"]
    matrices = (
        _accessor(gltf, binary, gltf["skins"][0]["inverseBindMatrices"])
        .reshape(-1, 4, 4)
        .transpose(0, 2, 1)
    )
    expected_inverse = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    expected_inverse[:, :3, 3] = -bind[:, :3]
    np.testing.assert_allclose(matrices, expected_inverse, atol=1e-7)
    bone_channel = next(
        c for c in animation["channels"] if c["target"] == {"node": 1, "path": "rotation"}
    )
    quats = _accessor(gltf, binary, animation["samplers"][bone_channel["sampler"]]["output"])
    np.testing.assert_allclose(
        Rotation.from_quat(quats[1]).as_matrix(), turn.as_matrix(), atol=1e-7
    )
    for frame, old in zip(pose["frames"], original["frames"], strict=True):
        for person, old_person in zip(frame, old, strict=True):
            for key in person:
                np.testing.assert_array_equal(person[key], old_person[key])
    for key in pose["_skeleton_override"]:
        np.testing.assert_array_equal(
            pose["_skeleton_override"][key], original["_skeleton_override"][key]
        )


def test_skeletal_glb_bones_have_flat_normals_and_dual_skin() -> None:
    pose = _mesh_pose()
    gltf, binary = _glb(
        build_glb_skeletal(pose, track_index=1, include_body_mesh=False, bone_vis="octahedrons")
    )
    primitive = gltf["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    joints = _accessor(gltf, binary, attributes["JOINTS_0"])
    weights = _accessor(gltf, binary, attributes["WEIGHTS_0"])
    normals = _accessor(gltf, binary, attributes["NORMAL"])
    assert joints.shape == (24, 4)
    np.testing.assert_array_equal(joints[:, :2], np.tile([1, 2], (24, 1)))
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-7)
    assert any(np.allclose(row, [0.9, 0.1, 0, 0]) for row in weights)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-7)
    np.testing.assert_array_equal(normals[::3], normals[1::3])
    assert not np.allclose(normals[0], normals[3])
    assert len(gltf["skins"]) == 1 and "animations" not in gltf


@pytest.mark.parametrize("public_node", (False, True))
def test_skeletal_glb_native_skin_truncation_morphs_and_model_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, public_node: bool
) -> None:
    pose = _mesh_pose()
    override = pose.pop("_skeleton_override")
    for frame in pose["frames"]:
        for person in frame:
            person["pred_joint_coords"][:, 1:3] *= -1
            person["expr_params"] = np.array([0.25, 0.75], np.float32)
            person["shape_params"] = np.array([2.0], np.float32)
    pose["frames"][2][0]["expr_params"] = np.array([0.5, -0.125], np.float32)

    class Rig:
        NUM_IDENTITY, NUM_EXPR, NUM_VERTS = 1, 2, 3
        skel_joint_parents = torch.tensor([-1, 0, 1], dtype=torch.int32)
        skel_joint_translation_offsets = torch.zeros(3, 3)
        lbs_vert_indices = torch.tensor([0] * 9 + [1])
        lbs_skin_indices = torch.tensor([i % 3 for i in range(9)] + [2])
        lbs_skin_weights = torch.tensor(list(range(1, 10)) + [2], dtype=torch.float32)
        expr_basis = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)

        def parameters(self) -> Any:
            yield torch.empty(0, dtype=torch.float64)

        def __call__(self, shape: Any, params: Any, expression: Any, **kwargs: Any) -> Any:
            assert shape.dtype == params.dtype == expression.dtype == torch.float64
            assert not params.any() and not expression.any() and not kwargs["apply_correctives"]
            bind = torch.from_numpy(override["bind_global_m"].copy()).unsqueeze(0)
            bind[..., :3] *= 100
            vertices = torch.from_numpy(override["rest_verts_m"].copy()).unsqueeze(0) * 100
            vertices[..., 0] += shape[0, 0] * 10
            return vertices, bind

    model = SimpleNamespace(model=SimpleNamespace(head_pose=SimpleNamespace(mhr=Rig())))
    if public_node:
        monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
        monkeypatch.setattr(
            "dinkster_nodes_comfy.sam3d_export.initialize_inference_process", lambda: None
        )
        monkeypatch.setitem(
            sys.modules,
            "dinkster_inference.model_management",
            SimpleNamespace(load_model_gpu=lambda model: None),
        )
        value = BuildPoseFile.execute(pose_data=pose, sam3d_body_model=model, track_index=0)[
            "model_3d"
        ]
        data = value["source"].read_bytes()
    else:
        data = build_glb_skeletal(pose, model, track_index=0)
    gltf, binary = _glb(data)
    primitive = gltf["meshes"][0]["primitives"][0]
    a = primitive["attributes"]
    first = _accessor(gltf, binary, a["WEIGHTS_0"])
    second = _accessor(gltf, binary, a["WEIGHTS_1"])
    np.testing.assert_allclose(
        np.concatenate([first[0], second[0]]), np.arange(9, 1, -1) / 44, atol=1e-7
    )
    np.testing.assert_array_equal(_accessor(gltf, binary, a["JOINTS_0"])[0], [2, 1, 0, 2])
    np.testing.assert_array_equal(first[2], [1, 0, 0, 0])
    np.testing.assert_allclose(
        _accessor(gltf, binary, a["POSITION"]), [[-0.3, 0, 0], [0.7, 0, 0], [0.2, 1, 0]], atol=1e-7
    )
    np.testing.assert_allclose(
        _accessor(gltf, binary, primitive["targets"][1]["POSITION"]),
        np.arange(9, 18).reshape(3, 3) / 100,
        atol=1e-8,
    )
    animation = gltf["animations"][0]
    channel = next(c for c in animation["channels"] if c["target"]["path"] == "weights")
    output = animation["samplers"][channel["sampler"]]["output"]
    np.testing.assert_array_equal(
        _accessor(gltf, binary, output).ravel(), [0.25, 0.75, 0.5, -0.125]
    )


def _point_pose() -> dict[str, Any]:
    points = np.arange(210, dtype=np.float32).reshape(70, 3) / 100
    first = {"pred_keypoints_3d": points, "pred_cam_t": np.array([1, 2, 3], np.float32)}
    moved = copy.deepcopy(first)
    moved["pred_keypoints_3d"] += np.array([0.1, 0.2, -0.3], np.float32)
    moved["pred_keypoints_3d"][6] += np.array([0.6, -0.4, 0.2], np.float32)
    moved["pred_cam_t"] += np.array([0.25, 0.5, 1], np.float32)
    return {"frames": [[first], [], [moved]]}


def test_openpose_glb_hand_order_colors_limb_motion_and_missing_frames() -> None:
    pose = _point_pose()
    original = copy.deepcopy(pose)
    gltf, binary = _glb(
        build_glb_openpose(pose, include_hands=True, camera_translation="centered", fps=20)
    )
    nodes = gltf["nodes"]
    indices = {node["name"]: i for i, node in enumerate(nodes)}
    animation = gltf["animations"][0]

    def values(name: str) -> Any:
        channel = next(
            c
            for c in animation["channels"]
            if c["target"] == {"node": indices[name], "path": "translation"}
        )
        sampler = animation["samplers"][channel["sampler"]]
        np.testing.assert_array_equal(
            _accessor(gltf, binary, sampler["input"]).ravel(), np.array([0, 0.1], np.float32)
        )
        return _accessor(gltf, binary, sampler["output"])

    np.testing.assert_allclose(values("openpose_R_thumb1")[0], [0.72, -0.73, -0.74], atol=1e-7)
    np.testing.assert_allclose(values("openpose_L_wrist")[0], [1.86, -1.87, -1.88], atol=1e-7)
    np.testing.assert_allclose(values("track00"), [[0, 0, 0], [0.25, -0.5, -1]], atol=1e-7)
    # Neck (69) and right shoulder (6), not the two neighboring source indices.
    np.testing.assert_allclose(
        values("openpose_limb_Neck_RShoulder")[0], [1.125, -1.135, -1.145], atol=2e-7
    )
    channel = next(
        c
        for c in animation["channels"]
        if c["target"] == {"node": indices["openpose_limb_Neck_RShoulder"], "path": "rotation"}
    )
    rotations = _accessor(gltf, binary, animation["samplers"][channel["sampler"]]["output"])
    axes = [
        (frame[0]["pred_keypoints_3d"][6] - frame[0]["pred_keypoints_3d"][69]) * [1, -1, -1]
        for frame in (pose["frames"][0], pose["frames"][2])
    ]
    np.testing.assert_allclose(
        Rotation.from_quat(rotations[1]).apply(axes[0] / np.linalg.norm(axes[0])),
        axes[1] / np.linalg.norm(axes[1]),
        atol=2e-7,
    )
    primitives = gltf["meshes"][0]["primitives"]
    np.testing.assert_array_equal(
        _accessor(gltf, binary, primitives[2]["attributes"]["COLOR_0"]),
        np.tile([0, 0, 1], (18144, 1)),
    )
    for frame, old in zip(pose["frames"], original["frames"], strict=True):
        for person, old_person in zip(frame, old, strict=True):
            for key in person:
                np.testing.assert_array_equal(person[key], old_person[key])


def test_openpose_glb_weighted_external_rest_and_face_landmark_selection() -> None:
    joints = np.array([[0, 0, 0], [2, 4, 6], [5, 7, 9]], np.float32)
    mapping = np.tile([0, 1], (18, 1))
    bind = np.zeros((3, 8), np.float32)
    bind[:, :3], bind[:, 6:] = joints, 1
    canonical = np.array([[-0.058, 1.660, 0.085], [0.04, 1.673, 0.09], [0, 1.54, 0.11]], np.float32)
    pose = {
        "frames": [[{"pred_joint_coords": joints + 1, "pred_vertices": canonical * [1, -1, -1]}]],
        "_skeleton_override": {
            "bind_global_m": bind,
            "openpose18_joint_indices": mapping,
            "openpose18_joint_weights": np.full(18, 0.25, np.float32),
            "rest_verts_m": canonical,
        },
        "canonical_colors": {"positions": canonical, "face_mask": np.ones(3, bool)},
    }
    original = copy.deepcopy(pose)
    gltf, binary = _glb(build_glb_openpose(pose, face_style="eyes_mouth"))
    nodes = {node["name"]: node for node in gltf["nodes"]}
    np.testing.assert_array_equal(nodes["openpose_Nose"]["translation"], [2.5, 4, 5.5])
    np.testing.assert_allclose(
        nodes["openpose_face_r_eye_outer"]["translation"], canonical[0], atol=1e-7
    )
    # Static sphere positions use weighted REST, not the first animated frame.
    primitive = gltf["meshes"][0]["primitives"][0]
    positions = _accessor(gltf, binary, primitive["attributes"]["POSITION"])
    np.testing.assert_allclose(positions.mean(axis=0), [1.5, 3, 4.5], atol=2e-5)
    face = gltf["meshes"][0]["primitives"][-1]
    np.testing.assert_array_equal(_accessor(gltf, binary, face["attributes"]["COLOR_0"]), 1)
    for key in pose["_skeleton_override"]:
        np.testing.assert_array_equal(
            pose["_skeleton_override"][key], original["_skeleton_override"][key]
        )
    with pytest.raises(ValueError, match="BOTH"):
        pose["_skeleton_override"]["openpose_hand21_r_joint_indices"] = np.tile([0, -1], (21, 1))
        build_glb_openpose(pose, include_hands=True)


def test_scail_glb_neck_stub_open_cylinders_and_smooth_material() -> None:
    points = np.zeros((70, 3), np.float32)
    points[0] = [2, 0, 0]
    points[69] = [0, 0, 0]
    points[10] = [0, 2, 0]
    points[9] = [0, 2, 0]
    pose = {"frames": [[{"pred_keypoints_3d": points}]]}
    gltf, binary = _glb(
        build_glb_openpose(
            pose,
            palette="scail",
            shape="capsule",
            smooth_shade=True,
            material_roughness=0.3,
            material_double_sided=True,
        )
    )
    nodes = {node["name"]: node for node in gltf["nodes"]}
    np.testing.assert_allclose(
        nodes["openpose_Nose"]["translation"], [2 / np.sqrt(13), 3 / np.sqrt(13), 0], atol=1e-7
    )
    assert len(gltf["skins"][0]["joints"]) == 31
    primitive = gltf["meshes"][0]["primitives"][1]
    a = primitive["attributes"]
    vertices = _accessor(gltf, binary, a["POSITION"])
    # Neck to hips, hips to knees, and neck to nose each have two 16-vertex rings.
    assert len(vertices) == 160
    assert len(_accessor(gltf, binary, primitive["indices"])) == 480
    np.testing.assert_allclose(
        np.linalg.norm(_accessor(gltf, binary, a["NORMAL"]), axis=1), 1, atol=1e-7
    )
    material = gltf["materials"][primitive["material"]]
    assert material["doubleSided"] is True
    assert material["pbrMetallicRoughness"]["roughnessFactor"] == 0.3
    assert "animations" not in gltf


def _export_pose() -> dict[str, Any]:
    pose = _mesh_pose()
    pose["image_size"] = (24, 40)
    for frame in pose["frames"]:
        for person in frame:
            person["pred_keypoints_3d"] = np.arange(210, dtype=np.float32).reshape(70, 3) / 100
    return pose


@pytest.mark.parametrize("mode", ("body_mesh", "bones_only", "openpose", "scail", "bvh"))
def test_pose_file_public_bytes_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    pose = _export_pose()
    original = copy.deepcopy(pose)
    options = {"format.mesh_style": mode} if mode != "bvh" else {"format.units": "m"}
    value = BuildPoseFile.execute(
        pose_data=pose,
        format="bvh" if mode == "bvh" else "glb",
        fps=20,
        camera_translation="centered",
        **options,
    )["model_3d"]
    _assert_file(mode, value)
    np.testing.assert_equal(pose, original)


def _assert_file(mode: str, value: Any) -> None:
    assert value["format"] == ("bvh" if mode == "bvh" else "glb")
    data = value["source"].read_bytes()
    if mode == "bvh":
        header, motion = _motion(data)
        assert "Frames: 2\n" in header
        assert "Frame Time: 0.050000\n" in data.decode()
        np.testing.assert_allclose(
            motion[:, :3], [[0.2, 1.3, 0.4], [0.7, 1.05, -0.6]], atol=1e-5, rtol=0
        )
    else:
        gltf, binary = _glb(data)
        assert len(gltf["skins"]) == 2
        if mode in ("body_mesh", "bones_only"):
            assert len(gltf["skins"][0]["joints"]) == 3
            inverse = _accessor(gltf, binary, gltf["skins"][0]["inverseBindMatrices"])
            np.testing.assert_allclose(
                inverse[0].reshape(4, 4).T[:3, 3], [-0.2, -0.3, -0.4], atol=1e-7
            )
        else:
            assert len(gltf["skins"][0]["joints"]) == (31 if mode == "scail" else 35)
            if mode == "scail":
                assert gltf["materials"][0]["doubleSided"] is True
                assert gltf["materials"][0]["pbrMetallicRoughness"]["roughnessFactor"] == 0.3
        animation = gltf["animations"][0]
        channel = next(
            c for c in animation["channels"] if c["target"] == {"node": 0, "path": "translation"}
        )
        sampler = animation["samplers"][channel["sampler"]]
        np.testing.assert_allclose(
            _accessor(gltf, binary, sampler["output"]), [[0, 0, 0], [0.5, -0.25, -1]], atol=1e-7
        )
        np.testing.assert_array_equal(
            _accessor(gltf, binary, sampler["input"]).ravel(), np.array([0, 0.1], np.float32)
        )


@pytest.mark.parametrize("mode", ("body_mesh", "bones_only", "bvh"))
def test_pose_file_requires_model_or_external_rig(mode: str) -> None:
    with pytest.raises(ValueError, match="sam3d_body_model.*_skeleton_override"):
        BuildPoseFile.execute(
            pose_data={"frames": []},
            format="bvh" if mode == "bvh" else "glb",
            **{"format.mesh_style": mode},
        )


class ExportSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "export-fixture.source",
            inputs=(),
            outputs=(
                OutputSpec("mhr", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),
                OutputSpec("kimodo", TypeExpr.concrete(KIMODO_POSE_DATA_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(mhr=_export_pose(), kimodo=_export_pose())


class ExportInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "export-fixture.inspect",
            inputs=(
                *(
                    InputSpec(mode, TypeExpr.concrete(FILE_3D_TYPE))
                    for mode in ("body_mesh", "bones_only", "openpose", "scail", "bvh")
                ),
                InputSpec("pose", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, pose: Any, **files: Any) -> Any:
        for mode, value in files.items():
            _assert_file(mode, value)
        np.testing.assert_equal(materialize_vision_packet(pose), _export_pose())
        return cls.outputs(result="pose file bytes preserved")


@pytest.mark.parametrize("in_process", (True, False))
def test_pose_file_ordinary_hosts_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))
    module = f"export_fixture_{int(in_process)}"
    (tmp_path / f"{module}.py").write_text(
        "from test_sam3d_export_torch import ExportSource, ExportInspect\n"
        "from dinkster_api.v1 import register_inference_types as inference_types, "
        "register_file3d_types, resolver_from_env\n"
        "NODES = (ExportSource, ExportInspect)\n"
        "def register_types(registry):\n"
        "    inference_types(registry)\n"
        "    register_file3d_types(registry, resolver_from_env())\n",
        encoding="ascii",
    )
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(
        '[pack]\nname="export-fixture"\npublisher="dinkster"\nnamespaces=["export-fixture"]\n'
        f'[pack.entry]\nnodes="{module}:NODES"\ntypes="{module}:register_types"\n',
        encoding="ascii",
    )

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        composer = ServingComposer(
            pack_scratch_root=tmp_path / "scratch",
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            },
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest, worker_group="export", group_manifests=(native, fixture)
                    )
                )
            nodes = {"source": GraphNode("export-fixture.source")}
            for index, mode in enumerate(("body_mesh", "bones_only", "openpose", "scail", "bvh")):
                options: dict[str, Any] = {
                    "pose_data": Link("source", "mhr" if index % 2 == 0 else "kimodo"),
                    "fps": 20,
                    "camera_translation": "centered",
                }
                variants = {"format": "bvh" if mode == "bvh" else "glb"}
                if mode == "bvh":
                    options["format.units"] = "m"
                else:
                    variants["format.mesh_style"] = mode
                    if mode in ("body_mesh", "bones_only"):
                        variants["format.mesh_style.bone_vis"] = (
                            "off" if mode == "body_mesh" else "octahedrons"
                        )
                    if mode == "body_mesh":
                        variants["format.mesh_style.shader"] = "default"
                nodes[mode] = GraphNode("BuildPoseFile", options, slot_variants=variants)
            nodes["inspect"] = GraphNode(
                "export-fixture.inspect",
                {
                    **{mode: Link(mode, "model_3d") for mode in nodes if mode != "source"},
                    "pose": Link("source", "mhr"),
                },
            )
            graph = graph_from_wire(graph_to_wire(Graph(nodes=nodes)))
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["result"].resolve() == "pose file bytes preserved"
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
