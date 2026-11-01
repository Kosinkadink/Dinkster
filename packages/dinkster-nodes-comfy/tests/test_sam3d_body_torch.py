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
from dinkster_nodes_comfy.sam3d_prediction import SAM3DBody_Predict

from dinkster.compose import PackSpec, ServingComposer


class PredictionHead:
    def __init__(self) -> None:
        self.scale_mean = torch.zeros(1)
        self.num_hand_comps = self.num_scale_comps = self.num_shape_comps = 2
        self.num_face_comps = 72
        self.face_region_rgb = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
        self.mhr = SimpleNamespace(
            NUM_VERTS=4,
            lbs_skin_weights=torch.tensor([0.5, 0.49, 0.3, 0.3, 1.0]),
            lbs_vert_indices=torch.tensor([0, 1, 2, 2, 3]),
            lbs_skin_indices=torch.tensor([0, 0, 0, 0, 1]),
        )

    def faces_np(self) -> Any:
        return np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

    def canonical_vertices(self) -> Any:
        return torch.tensor([[0, 1.5, 0], [0.1, 1.5, 0], [0, 1.6, 0], [0.2, 1.6, 0]])

    def mhr_forward(self, **values: Any) -> Any:
        count = len(values["expr_params"])
        vertices = self.canonical_vertices()[None].repeat(count, 1, 1)
        vertices[:, 0, 2] += values["expr_params"].sum(-1) * 0.001
        if values.get("return_keypoints"):
            return (
                vertices,
                torch.zeros(count, 70, 3),
                torch.tensor([[[0.0, 0, 0], [1, 1, 1]]]).repeat(count, 1, 1),
            )
        return vertices


class PredictionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_size = (6, 8)
        self.head_pose = PredictionHead()
        self.calls: list[Any] = []
        self.memory: list[Any] = []

    def memory_used_forward(self, crops: int, refinement: bool) -> int:
        self.memory.append([crops, refinement])
        return crops * 1000 + int(refinement)

    def run_inference(
        self, images: Any, batch: Any, *, inference_type: str, thresh_wrist_angle: float
    ) -> Any:
        assert thresh_wrist_angle == 1.4
        assert batch["img"].shape[2:] == (3, 8, 6)
        assert all(im.dtype == torch.uint8 and im.device.type == "cpu" for im in images)
        boxes = batch["bbox"][0]
        tokens = torch.tensor([int(im[0, 0, 0]) for im in images]) * 100 + boxes[:, 0]
        shapes = {
            "focal_length": (1,),
            "pred_keypoints_3d": (70, 3),
            "pred_keypoints_2d": (70, 2),
            "pred_vertices": (4, 3),
            "pred_cam_t": (3,),
            "pred_pose_raw": (10,),
            "global_rot": (3,),
            "body_pose": (130,),
            "hand": (4,),
            "scale": (2,),
            "shape": (2,),
            "face": (72,),
            "pred_joint_coords": (2, 3),
            "joint_global_rots": (2, 3, 3),
            "mhr_model_params": (8,),
            "pred_face_keypoints_3d": (238, 3),
            "pred_face_keypoints_2d": (238, 2),
        }
        output = {
            key: tokens.float().reshape(-1, *([1] * len(shape))).expand(-1, *shape).clone()
            for key, shape in shapes.items()
        }
        self.calls.append((len(images), batch["cam_int"].clone(), batch["mask_score"].clone()))
        if inference_type == "full":
            hand = {
                "bbox_center": batch["bbox_center"] + 1,
                "bbox_scale": torch.full_like(batch["bbox_scale"], 2),
            }
            return {"mhr": output}, hand, hand, None, None
        return {"mhr": output}


def prediction_images(count: int = 5) -> Any:
    image = torch.zeros(count, 16, 24, 3)
    image[..., 0] = (torch.arange(count) + 16)[:, None, None] / 255
    image[..., 1] = -0.5
    image[..., 2] = 1.5
    return image


@pytest.mark.parametrize("refine", (False, True))
@pytest.mark.parametrize("selection", ("full", "single", "flat", "ragged", "short", "empty"))
def test_prediction_order_crops_camera_metadata_and_preservation(
    monkeypatch: pytest.MonkeyPatch, selection: str, refine: bool
) -> None:
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    loaded: list[int] = []
    monkeypatch.setattr(
        management,
        "load_models_gpu",
        lambda models, memory_required: loaded.append(memory_required),
    )
    box = {"x": 2, "y": 3, "width": 8, "height": 7}
    other = {**box, "x": 12}
    selections = {
        "full": None,
        "single": box,
        "flat": [box, other],
        "ragged": [[box, other], [other], [], [box, other], [other, box, other]],
        "short": [[box], []],
        "empty": [],
    }
    expected = {
        "full": [[[0, 0, 24, 16]]] * 5,
        "single": [[[2, 3, 10, 10]]] * 5,
        "flat": [[[2, 3, 10, 10], [12, 3, 20, 10]]] * 5,
        "ragged": [
            [[2, 3, 10, 10], [12, 3, 20, 10]],
            [[12, 3, 20, 10]],
            [],
            [[2, 3, 10, 10], [12, 3, 20, 10]],
            [[12, 3, 20, 10], [2, 3, 10, 10], [12, 3, 20, 10]],
        ],
        "short": [[[2, 3, 10, 10]], [], [], [], []],
        "empty": [[[0, 0, 24, 16]]] * 5,
    }[selection]
    model = PredictionModel()
    image = prediction_images()
    before = image.clone()
    detections = copy.deepcopy(selections[selection])
    result = SAM3DBody_Predict.execute(
        sam3d_body_model=SimpleNamespace(model=model),
        image=image,
        bboxes=detections,
        run_hand_refinement=refine,
        fov=90,
        batch_size=4,
    )["mhr_pose_data"]
    assert detections == selections[selection]
    torch.testing.assert_close(image, before, atol=0, rtol=0)
    assert result["image_size"] == (16, 24)
    for index, (frame, boxes) in enumerate(zip(result["frames"], expected, strict=True)):
        assert len(frame) == len(boxes)
        for person, bbox in zip(frame, boxes, strict=True):
            np.testing.assert_array_equal(person["bbox"], bbox)
            np.testing.assert_array_equal(
                person["pred_vertices"], np.full((4, 3), (index + 16) * 100 + bbox[0])
            )
            assert person["pred_vertices"].dtype == np.float32
            assert person["mask"] is None
            if refine:
                x1, y1, x2, y2 = bbox
                np.testing.assert_array_equal(
                    person["lhand_bbox"],
                    [(x1 + x2) / 2, (y1 + y2) / 2, (x1 + x2) / 2 + 2, (y1 + y2) / 2 + 2],
                )
            else:
                assert "lhand_bbox" not in person
    expected_crops = {"full": 4, "single": 4, "flat": 4, "ragged": 4, "short": 1, "empty": 4}[
        selection
    ]
    assert model.memory == [[expected_crops, refine]] and loaded == [expected_crops * 1000 + refine]
    for crops, camera, _ in model.calls:
        assert crops <= 4
        torch.testing.assert_close(camera, torch.tensor([[[8.0, 0, 12], [0, 8, 8], [0, 0, 1]]]))
    np.testing.assert_array_equal(result["faces"], [[0, 1, 2], [1, 3, 2]])
    np.testing.assert_array_equal(result["canonical_colors"]["norm"], [[0.5, 0.5, 1.0]] * 4)
    np.testing.assert_array_equal(
        result["canonical_colors"]["face_mask"], [True, False, False, False]
    )
    np.testing.assert_array_equal(
        result["canonical_colors"]["head_mask"], [True, True, True, False]
    )
    np.testing.assert_array_equal(result["hand_vert_mask"], [True, False, True, False])


def test_prediction_tracking_precedence_empty_fallback_and_zero_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    monkeypatch.setattr(management, "load_models_gpu", lambda *args, **kwargs: None)
    tracker = importlib.import_module("dinkster_inference.ldm.sam3.tracker")
    masks = torch.zeros(5, 2, 16, 24, dtype=torch.uint8)
    masks[:, 0, 2:12, 4:20] = 1
    packed = tracker.pack_masks(masks)
    before = packed.clone()
    for count, track, boxes in (
        (5, {"packed_masks": packed}, [[6, 4, 18, 10], [0, 0, 24, 16]]),
        (5, {"packed_masks": packed[:2]}, [[0, 0, 24, 16]]),
        (5, {"packed_masks": torch.empty(5, 0, 16, 3, dtype=torch.uint8)}, [[0, 0, 24, 16]]),
        (0, None, []),
    ):
        result = SAM3DBody_Predict.execute(
            sam3d_body_model=SimpleNamespace(model=PredictionModel()),
            image=prediction_images(count),
            track_data=track,
            bboxes=None,
            run_hand_refinement=False,
            batch_size=3,
        )["mhr_pose_data"]
        assert len(result["frames"]) == count
        for frame in result["frames"]:
            assert len(frame) == len(boxes)
            for person, bbox in zip(frame, boxes, strict=True):
                np.testing.assert_array_equal(person["bbox"], bbox)
        if count and len(boxes) == 2:
            torch.testing.assert_close(result["frames"][0][0]["mask"], masks[0, 0, :, :, None])
    torch.testing.assert_close(packed, before, rtol=0, atol=0)


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


class PredictionSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sam3d-fixture.prediction-source",
            inputs=(),
            outputs=(
                OutputSpec("model", TypeExpr.concrete(SAM3D_BODY_MODEL_TYPE)),
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        initialize_inference_process()
        patcher = importlib.import_module("dinkster_inference.model_patcher").ModelPatcher
        return cls.outputs(
            model=patcher(PredictionModel(), torch.device("cpu"), torch.device("cpu")),
            image=prediction_images(),
        )


class PredictionInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sam3d-fixture.prediction-inspect",
            inputs=(InputSpec("pose", TypeExpr.concrete(MHR_POSE_DATA_TYPE)),),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, pose: Any) -> Any:
        pose = materialize_vision_packet(pose)
        assert pose["image_size"] == (16, 24)
        for index, frame in enumerate(pose["frames"]):
            assert len(frame) == 1
            np.testing.assert_array_equal(
                frame[0]["pred_vertices"], np.full((4, 3), (index + 16) * 100)
            )
            np.testing.assert_array_equal(frame[0]["bbox"], [0, 0, 24, 16])
            assert frame[0]["pred_vertices"].dtype == np.float32
        np.testing.assert_array_equal(pose["hand_vert_mask"], [True, False, True, False])
        return cls.outputs(result="prediction values preserved")


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


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("prediction", (True, False))
def test_sam3d_ordinary_hosts_twice(tmp_path: Path, in_process: bool, prediction: bool) -> None:
    module = f"sam3d_fixture_{int(in_process)}_{int(prediction)}"
    (tmp_path / f"{module}.py").write_text(
        "from test_sam3d_body_torch import Source, Inspect, PredictionSource, PredictionInspect\n"
        "from dinkster_api.v1 import register_inference_types\n"
        "NODES = (Source, Inspect, PredictionSource, PredictionInspect)\n",
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
                                "sam3d-fixture.prediction-source"
                                if prediction
                                else "sam3d-fixture.source"
                            ),
                            "smooth": GraphNode(
                                "SAM3DBody_Predict" if prediction else "SAM3DBody_Smooth",
                                {
                                    "sam3d_body_model": Link("source", "model"),
                                    "image": Link("source", "image"),
                                    "bboxes": [],
                                    "fov": 90.0,
                                    "batch_size": 4,
                                    "run_hand_refinement": False,
                                }
                                if prediction
                                else {
                                    "mhr_pose_data": Link("source", "pose"),
                                    "window": 3,
                                    "rotation_threshold_degrees": 0.0,
                                },
                            ),
                            "inspect": GraphNode(
                                "sam3d-fixture.prediction-inspect"
                                if prediction
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
                    "prediction values preserved" if prediction else "pose values preserved"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
