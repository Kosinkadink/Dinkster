"""Temporal pose results and canonical transport through both ordinary hosts."""

from __future__ import annotations

import asyncio
import copy
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    MHR_POSE_DATA_TYPE,
    SAM3D_BODY_MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_vision_packet,
)
from dinkster_graph import Graph, GraphNode, Link, graph_from_wire, graph_to_wire
from dinkster_nodes_comfy.sam3d_body import SAM3DBody_Smooth
from dinkster_nodes_comfy.sam3d_expression import SAM3DBody_FaceExpression

from dinkster.compose import PackSpec, ServingComposer


class ExpressionDetector:
    def __init__(self, empty: bool = False, gaps: bool = False) -> None:
        self.empty, self.gaps = empty, gaps
        self.calls: list[Any] = []

    def detect_batch(self, images: Any, *, num_faces: int) -> Any:
        assert len(images) == 1 and images[0].dtype == np.uint8
        marker = int(images[0][0, 0, 0])
        self.calls.append(
            dict(frame_id=marker, image_shape=list(images[0].shape), num_faces=num_faces)
        )
        if self.empty or (self.gaps and marker == 1):
            return [[]]
        return [
            [
                {
                    "bbox_xyxy": np.array([4, 4, 20, 20], dtype=np.float32),
                    "landmarks_xy": np.array([[5, 5], [15, 15]], dtype=np.float32),
                    "blendshapes": {
                        "jawOpen": 0.1,
                        "mouthLowerDownLeft": 0.8,
                        "eyeBlinkLeft": 0.2,
                        "browDownLeft": 0.6,
                    },
                }
            ]
        ]


class ExpressionHead:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def mhr_forward(self, **values: Any) -> Any:
        assert torch.count_nonzero(values["global_trans"]) == 0
        expr = values["expr_params"]
        self.calls.append(dict(count=len(expr), expr_params=expr.tolist()))
        xyz = torch.stack((expr[:, 24], torch.ones(len(expr)), torch.full((len(expr),), 2.0)), -1)
        return (
            xyz[:, None].expand(-1, 3, 3).clone(),
            xyz[:, None].expand(-1, 308, 3).clone(),
            xyz[:, None].expand(-1, 2, 3).clone(),
            None,
            torch.eye(3)[None, None].expand(len(expr), 2, 3, 3).clone(),
        )


class ExpressionModel(torch.nn.Module):
    def __init__(self, empty: bool = False, gaps: bool = False) -> None:
        super().__init__()
        self.face_landmarker = ExpressionDetector(empty, gaps)
        self.head_pose = ExpressionHead()


def expression_pose(count: int, crop: str = "") -> dict[str, Any]:
    kp = np.tile(np.array([6.0, 9.0], dtype=np.float32), (70, 1))
    if crop == "keypoints":
        kp[:5] = [[15, 15], [25, 15], [20, 20], [12, 17], [28, 17]]
    elif crop == "bbox":
        kp[:] = 0
    person = {
        "bbox": np.array([4, 4, 52, 52] if crop else [0, 0, 64, 64], dtype=np.float32),
        "global_rot": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "body_pose_params": np.zeros(124, dtype=np.float32),
        "hand_pose_params": np.zeros(54, dtype=np.float32),
        "shape_params": np.zeros(45, dtype=np.float32),
        "scale_params": np.zeros(28, dtype=np.float32),
        "expr_params": np.full(72, 9, dtype=np.float32),
        "pred_cam_t": np.array([1, 2, 10], dtype=np.float32),
        "focal_length": np.array([10.0], dtype=np.float32),
        "pred_keypoints_2d": kp,
        "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
        "application_tag": "preserved",
    }
    return {
        "frames": [[copy.deepcopy(person)] for _ in range(count)],
        "H": 64,
        "W": 64,
        "faces": np.array([[0, 1, 2]], dtype=np.int64),
    }


def expression_images(count: int) -> Any:
    return (
        torch.arange(max(1, count), dtype=torch.float32)[:, None, None, None]
        .expand(-1, 64, 64, 3)
        .clone()
        / 255
    )


@pytest.mark.parametrize(
    ("count", "crop", "empty", "gaps", "strength", "window"),
    (
        (0, "", False, False, 1.5, 1),
        (6, "", False, True, 1.5, 1),
        (3, "", False, False, 0.0, 1),
        (3, "", True, False, 1.5, 1),
        (3, "keypoints", False, False, 1.5, 1),
        (3, "bbox", False, False, 1.5, 1),
        (29, "", False, False, 1.5, 7),
        (30, "", False, False, 1.5, 7),
    ),
)
def test_expression_mapping_geometry_baseline_crops_and_preservation(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    crop: str,
    empty: bool,
    gaps: bool,
    strength: float,
    window: int,
) -> None:
    """Controlled detector and MHR forward; actual node transforms and projection."""
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    monkeypatch.setattr(management, "load_model_gpu", lambda model: None)
    model = ExpressionModel(empty, gaps)
    pose, images = expression_pose(count, crop), expression_images(count)
    before = copy.deepcopy(pose)
    image_before = images.clone()
    result = SAM3DBody_FaceExpression.execute(
        sam3d_body_model=SimpleNamespace(model=model),
        mhr_pose_data=pose,
        image=images,
        strength=strength,
        mouth_strength=2,
        eye_strength=3,
        brow_strength=4,
        input_threshold=0.2,
        blendshape_smooth_window=window,
    )["mhr_pose_data"]
    torch.testing.assert_close(images, image_before, rtol=0, atol=0)
    for frame, original in zip(pose["frames"], before["frames"], strict=True):
        for key in original[0]:
            if isinstance(original[0][key], np.ndarray):
                np.testing.assert_array_equal(frame[0][key], original[0][key])
    assert result["faces"] is pose["faces"]
    if not count:
        assert result is pose and not model.face_landmarker.calls and not model.head_pose.calls
    elif empty:
        assert not model.head_pose.calls
        for frame in result["frames"]:
            np.testing.assert_array_equal(frame[0]["expr_params"], np.full(72, 9))
    else:
        expected = np.zeros(72, dtype=np.float32)
        if strength and count < 30:
            expected[[0, 24, 54]] = [3.0, 0.675, 2.25]
        for frame in result["frames"]:
            p = frame[0]
            np.testing.assert_allclose(p["expr_params"], expected, rtol=0, atol=5e-7)
            np.testing.assert_allclose(
                p["pred_vertices"][0], [expected[24], -1, -2], rtol=0, atol=1e-7
            )
            assert p["expr_params"].dtype == np.float32 and p["application_tag"] == "preserved"
        if not crop:
            np.testing.assert_allclose(
                result["frames"][0][0]["pred_keypoints_2d"][0],
                [6.25 + 1.25 * expected[24], 8.25],
                rtol=0,
                atol=1e-6,
            )
        else:
            assert all(call["image_shape"][:2] != [64, 64] for call in model.face_landmarker.calls)


def test_expression_smooths_varying_signal_with_even_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    monkeypatch.setattr(management, "load_model_gpu", lambda model: None)

    class VaryingDetector(ExpressionDetector):
        def detect_batch(self, images: Any, *, num_faces: int) -> Any:
            faces = super().detect_batch(images, num_faces=num_faces)
            marker = int(images[0][0, 0, 0])
            faces[0][0]["blendshapes"] = {"jawOpen": (0.2, 0.8, 0.4)[marker]}
            return faces

    model = ExpressionModel()
    model.face_landmarker = VaryingDetector()
    result = SAM3DBody_FaceExpression.execute(
        sam3d_body_model=SimpleNamespace(model=model),
        mhr_pose_data=expression_pose(3),
        image=expression_images(3),
        blendshape_smooth_window=2,
    )["mhr_pose_data"]
    # Three Gaussian taps, endpoint repetition, and even-window normalization to three.
    neighbor = np.exp(-0.5) / (1 + 2 * np.exp(-0.5))
    expected = [0.2 + 0.6 * neighbor, 0.8 - neighbor, 0.4 + 0.4 * neighbor]
    for frame, jaw in zip(result["frames"], expected, strict=True):
        np.testing.assert_allclose(frame[0]["expr_params"][24], jaw, rtol=0, atol=1e-7)
        np.testing.assert_allclose(frame[0]["pred_vertices"][0], [jaw, -1, -2], rtol=0, atol=1e-7)


def test_expression_long_gaps_require_both_endpoints_within_twelve_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    monkeypatch.setattr(management, "load_model_gpu", lambda model: None)

    class SparseDetector(ExpressionDetector):
        def detect_batch(self, images: Any, *, num_faces: int) -> Any:
            marker = int(images[0][0, 0, 0])
            if marker not in (0, 14):
                return [[]]
            result = super().detect_batch(images, num_faces=num_faces)
            result[0][0]["blendshapes"] = {"jawOpen": 0.2 if marker == 0 else 0.9}
            return result

    model = ExpressionModel()
    model.face_landmarker = SparseDetector()
    result = SAM3DBody_FaceExpression.execute(
        sam3d_body_model=SimpleNamespace(model=model),
        mhr_pose_data=expression_pose(15),
        image=expression_images(15),
        blendshape_smooth_window=1,
    )["mhr_pose_data"]
    for index, frame in enumerate(result["frames"]):
        jaw = 9.0 if index in (1, 13) else 0.2 + index * 0.05
        np.testing.assert_allclose(frame[0]["expr_params"][24], jaw, rtol=0, atol=1e-7)


def test_expression_assigns_larger_person_first_without_reusing_faces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    monkeypatch.setattr(management, "load_model_gpu", lambda model: None)
    pose = expression_pose(1)
    small = pose["frames"][0][0]
    small["bbox"] = np.array([0, 0, 70, 128], dtype=np.float32)
    large = copy.deepcopy(small)
    large["bbox"] = np.array([0, 0, 128, 128], dtype=np.float32)
    pose["frames"][0].append(large)
    model = ExpressionModel()
    result = SAM3DBody_FaceExpression.execute(
        sam3d_body_model=SimpleNamespace(model=model),
        mhr_pose_data=pose,
        image=torch.zeros(1, 128, 128, 3),
        blendshape_smooth_window=1,
    )["mhr_pose_data"]
    # The single overlapping face belongs to the larger second person, not input order.
    np.testing.assert_array_equal(result["frames"][0][0]["expr_params"], np.full(72, 9))
    np.testing.assert_allclose(
        result["frames"][0][1]["expr_params"][24], 0.3 * (0.8 - 0.02) / 0.98, rtol=0, atol=1e-7
    )
    assert model.face_landmarker.calls[0]["num_faces"] == 2


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


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sam3d-fixture.source",
            inputs=(),
            outputs=(OutputSpec("pose", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(pose=_pose())


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sam3d-fixture.inspect",
            inputs=(InputSpec("pose", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, pose: Any) -> Any:
        _assert_result(pose)
        return cls.outputs(result="pose values preserved")


class ExpressionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sam3d-fixture.expression-source",
            inputs=(),
            outputs=(
                OutputSpec("model", TypeExpr.concrete(SAM3D_BODY_MODEL_TYPE)),
                OutputSpec("pose", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        initialize_inference_process()
        patcher = importlib.import_module("dinkster_inference.model_patcher").ModelPatcher
        return cls.outputs(
            model=patcher(ExpressionModel(gaps=True), torch.device("cpu"), torch.device("cpu")),
            pose={**expression_pose(6), "image_size": (64, 64)},
            image=expression_images(6),
        )


class ExpressionInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sam3d-fixture.expression-inspect",
            inputs=(InputSpec("pose", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, pose: Any) -> Any:
        pose = materialize_vision_packet(pose)
        assert len(pose["frames"]) == 6 and pose["H"] == pose["W"] == 64
        expected = np.zeros(72, dtype=np.float32)
        expected[[0, 24, 54]] = [3.0, 0.675, 2.25]
        for frame in pose["frames"]:
            value = frame[0]
            np.testing.assert_allclose(value["expr_params"], expected, rtol=0, atol=5e-7)
            np.testing.assert_allclose(
                value["pred_vertices"], [[0.675, -1, -2]] * 3, rtol=0, atol=1e-7
            )
            np.testing.assert_allclose(
                value["pred_face_keypoints_2d"], [[7.09375, 8.25]] * 238, rtol=0, atol=1e-6
            )
            assert value["expr_params"].dtype == value["pred_vertices"].dtype == np.float32
            assert value["application_tag"] == "preserved"
        np.testing.assert_array_equal(pose["faces"], [[0, 1, 2]])
        return cls.outputs(result="expression values preserved")


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("expression", (True, False))
def test_smoothing_ordinary_hosts_twice(tmp_path: Path, in_process: bool, expression: bool) -> None:
    module = f"sam3d_fixture_{int(in_process)}_{int(expression)}"
    (tmp_path / f"{module}.py").write_text(
        "from test_sam3d_body_torch import Source, Inspect, ExpressionSource, ExpressionInspect\n"
        "from dinkster_api.v1 import register_inference_types\n"
        "NODES = (Source, Inspect, ExpressionSource, ExpressionInspect)\n",
        encoding="ascii",
    )
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(
        '[pack]\nname="sam3d-fixture"\npublisher="dinkster"\nnamespaces=["sam3d-fixture"]\n'
        f'[pack.entry]\nnodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
        encoding="ascii",
    )

    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest, worker_group="sam3d", group_manifests=(native, fixture)
                    )
                )
            graph = graph_from_wire(
                graph_to_wire(
                    Graph(
                        nodes={
                            "source": GraphNode(
                                "sam3d-fixture.expression-source"
                                if expression
                                else "sam3d-fixture.source"
                            ),
                            "smooth": GraphNode(
                                "SAM3DBody_FaceExpression" if expression else "SAM3DBody_Smooth",
                                {
                                    "sam3d_body_model": Link("source", "model"),
                                    "mhr_pose_data": Link("source", "pose"),
                                    "image": Link("source", "image"),
                                    "strength": 1.5,
                                    "mouth_strength": 2.0,
                                    "eye_strength": 3.0,
                                    "brow_strength": 4.0,
                                    "input_threshold": 0.2,
                                    "blendshape_smooth_window": 1,
                                }
                                if expression
                                else {
                                    "mhr_pose_data": Link("source", "pose"),
                                    "window": 3,
                                    "rotation_threshold_degrees": 0.0,
                                },
                            ),
                            "inspect": GraphNode(
                                "sam3d-fixture.expression-inspect"
                                if expression
                                else "sam3d-fixture.inspect",
                                {
                                    "pose": Link("smooth", "mhr_pose_data"),
                                },
                            ),
                        }
                    )
                )
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["result"].resolve() == (
                    "expression values preserved" if expression else "pose values preserved"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
