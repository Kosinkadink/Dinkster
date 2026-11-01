"""Animation file hierarchy, channel values and source preservation."""

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_nodes_comfy.sam3d_bvh import build_bvh
from scipy.spatial.transform import Rotation


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
