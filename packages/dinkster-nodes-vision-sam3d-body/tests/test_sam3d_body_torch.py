"""Temporal pose results and canonical transport through both ordinary hosts."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from dinkster_api.v1 import (
    MHR_POSE_DATA_TYPE,
    SAM3D_BODY_MODEL_TYPE,
    TypeRegistry,
    materialize_vision_packet,
)
from dinkster_vision_sam3d_body import register_types
from dinkster_vision_sam3d_body.sam3d_body import SAM3DBody_Loader, SAM3DBody_Smooth


def test_pack_registers_required_pose_packets_and_preserves_existing_authority() -> None:
    registry = TypeRegistry()
    resident = registry.register(SAM3D_BODY_MODEL_TYPE)
    register_types(registry)
    assert {
        "MHR_POSE_DATA",
        "KIMODO_POSE_DATA",
        "SAM3D_BODY_MODEL",
        "SAM3_TRACK_DATA",
        "LOAD3D_CAMERA",
        "IMAGE",
        "BOUNDING_BOX",
        "FILE_3D",
        "ASSET",
    } <= set(registry.type_ids())
    assert not {"DA3_GEOMETRY", "FACE_LANDMARKS", "DA3_MODEL", "FACE_DETECTION_MODEL"} & set(
        registry.type_ids()
    )
    packet = registry.spec(MHR_POSE_DATA_TYPE)
    register_types(registry)
    assert registry.spec(MHR_POSE_DATA_TYPE) is packet
    assert registry.spec(SAM3D_BODY_MODEL_TYPE) is resident


def test_sam3d_loader_rejects_unbound_file() -> None:
    with pytest.raises(TypeError, match="model_file must be an AssetRef or LocalAssetRef"):
        SAM3DBody_Loader.execute(model_file="model.safetensors")


def _pose() -> dict[str, Any]:
    return {
        "frames": [
            [
                {
                    "pred_vertices": np.array([[x, 2 * x, -x]], dtype=np.float64),
                    "pred_cam_t": np.array([x, 10 + x, -2 + x], dtype=np.float32),
                    "global_rot": np.array([0, 0, 0], dtype=np.float32),
                    "tag": "preserved",
                }
            ]
            for x in (0, 6, 0, 12)
        ],
        "image_size": (3, 5),
        "faces": np.array([[0, 0, 0]], dtype=np.uint32),
        "fps": 24,
    }


def _smooth(pose: Any, **options: Any) -> Any:
    return SAM3DBody_Smooth.execute(
        mhr_pose_data=pose,
        **(
            {"strength": 1.0, "method": "savgol", "window": 3, "rotation_threshold_degrees": 0.0}
            | options
        ),
    )["mhr_pose_data"]


def _assert_result(pose: Any) -> None:
    pose = materialize_vision_packet(pose)
    # Three-point linear least squares has weights 1/3, including repeated endpoints.
    expected = [2, 2, 6, 8]
    for frame, x in zip(pose["frames"], expected, strict=True):
        np.testing.assert_allclose(frame[0]["pred_vertices"], [[x, 2 * x, -x]], atol=1e-6)
        np.testing.assert_allclose(frame[0]["pred_cam_t"], [x, 10 + x, -2 + x], atol=1e-6)
        assert frame[0]["pred_vertices"].dtype == np.float64
        assert frame[0]["tag"] == "preserved"
    assert pose["image_size"] == (3, 5) and pose["fps"] == 24
    np.testing.assert_array_equal(pose["faces"], [[0, 0, 0]])
    assert pose["faces"].dtype == np.uint32


def test_smoothing_values_noops_and_source_preservation() -> None:
    pose = _pose()
    before = copy.deepcopy(pose)
    _assert_result(_smooth(pose))
    assert _smooth(pose, strength=0.0) is pose
    assert _smooth(pose, window=1) is pose
    for actual, original in zip(pose["frames"], before["frames"], strict=True):
        for key in ("pred_vertices", "pred_cam_t", "global_rot"):
            np.testing.assert_array_equal(actual[0][key], original[0][key])
    empty = {**pose, "frames": []}
    assert _smooth(empty) is empty
    singleton = {**pose, "frames": pose["frames"][:1]}
    assert _smooth(singleton) is singleton


def test_fast_spin_backs_off_geometry_but_not_camera() -> None:
    pose = _pose()
    for index, frame in enumerate(pose["frames"]):
        frame[0]["global_rot"][0] = index * np.pi / 2
    result = _smooth(pose, method="gaussian", rotation_threshold_degrees=30.0)
    for actual, original in zip(result["frames"], pose["frames"], strict=True):
        np.testing.assert_array_equal(actual[0]["pred_vertices"], original[0]["pred_vertices"])
    # Gaussian sigma=1 at window3; independent normalized endpoint weight.
    weight = np.exp(-0.5) / (1 + 2 * np.exp(-0.5))
    np.testing.assert_allclose(
        result["frames"][0][0]["pred_cam_t"],
        [6 * weight, 10 + 6 * weight, -2 + 6 * weight],
        atol=1e-6,
    )


def test_wrapped_angle_ranges_and_rotation_projection() -> None:
    pose = _pose()
    angles = [3.0, -3.0, 3.0, -3.0]
    for frame, angle in zip(pose["frames"], angles, strict=True):
        frame[0]["global_rot"][0] = angle
        frame[0]["body_pose_params"] = np.full(126, angle, dtype=np.float32)
        frame[0]["mhr_model_params"] = np.full(132, angle, dtype=np.float32)
        frame[0]["pred_global_rots"] = np.diag([1.0, 2.0, -1.0])[None]
    result = _smooth(pose)
    root = result["frames"][0][0]
    # First window repeats the first angle; only -3 crosses to 2*pi-3.
    unwrapped = (3 + 2 * np.pi) / 3
    linear = 1.0
    np.testing.assert_allclose(root["global_rot"][0], unwrapped, atol=1e-6)
    np.testing.assert_allclose(root["body_pose_params"][:124], unwrapped, atol=1e-6)
    np.testing.assert_allclose(root["body_pose_params"][124:], linear, atol=1e-6)
    np.testing.assert_allclose(root["mhr_model_params"][:3], linear, atol=1e-6)
    np.testing.assert_allclose(root["mhr_model_params"][3:130], unwrapped, atol=1e-6)
    np.testing.assert_allclose(root["mhr_model_params"][130:], linear, atol=1e-6)
    for frame in result["frames"]:
        rotation = frame[0]["pred_global_rots"][0]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-6)


def test_missing_people_fields_even_window_and_partial_strength() -> None:
    pose = _pose()
    pose["frames"][1] = []
    del pose["frames"][2][0]["pred_cam_t"]
    result = _smooth(pose)
    assert result["frames"][1] == []
    np.testing.assert_allclose(result["frames"][2][0]["pred_cam_t"], [4, 14, 2], atol=1e-6)
    # An even requested window normalizes to five; half strength blends with raw values.
    result = _smooth(_pose(), window=4, method="gaussian", strength=0.5)
    weights = np.array([np.exp(-2), np.exp(-0.5), 1, np.exp(-0.5), np.exp(-2)])
    expected = 0.5 * (6 * weights[3]) / weights.sum()
    np.testing.assert_allclose(result["frames"][0][0]["pred_vertices"][0, 0], expected, atol=1e-6)
