"""Face drawing pixels, source preservation and ordinary-host cache behavior."""

from __future__ import annotations

import asyncio
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    FACE_LANDMARKS_TYPE,
    IMAGE_TYPE,
    MASK_TYPE,
    AssetRef,
    AssetVault,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    digest_bytes,
    materialize_image,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link, graph_from_wire, graph_to_wire
from dinkster_nodes_comfy.mediapipe import MediaPipeFaceMask, MediaPipeFaceMeshVisualize
from dinkster_nodes_comfy.mediapipe_detection import (
    LoadMediaPipeFaceLandmarker,
    MediaPipeFaceLandmarker,
)

from dinkster.compose import PackSpec, ServingComposer


def _packet() -> dict[str, Any]:
    def ring(indices: list[int]) -> frozenset[tuple[int, int]]:
        return frozenset(zip(indices, indices[1:] + indices[:1], strict=True))

    landmarks = np.array(
        [[1, 1], [6, 1], [6, 4], [1, 4], [2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float32
    )
    return {
        "frames": [
            [{"landmarks_xy": landmarks}, {"landmarks_xy": landmarks + [4, 0]}],
            [],
            [{"landmarks_xy": landmarks}],
        ],
        "image_size": (7, 11),
        "connection_sets": {
            "face_oval": ring([0, 1, 2, 3]),
            **{part: ring([4, 5, 6, 7]) for part in ("lips", "left_eye", "right_eye", "irises")},
            "left_eyebrow": frozenset({(4, 5)}),
            "right_eyebrow": frozenset({(7, 6)}),
            "nose": frozenset({(0, 6)}),
            "tesselation": frozenset({(0, 2), (1, 3), (0, 99)}),
        },
    }


def _visualize(packet: Any, **options: Any) -> Any:
    return MediaPipeFaceMeshVisualize.execute(
        face_landmarks=packet,
        **{
            "connections": {"connections": "fill"},
            "color": "#d02070",
            "thickness": 2,
            "point_size": 3,
            **options,
        },
    )["IMAGE"]


def _assert_pixels(image: Any, mask: Any) -> None:
    image, mask = materialize_image(image), torch.as_tensor(materialize_latent(mask))
    expected = torch.zeros(3, 7, 11)
    expected[0, 1:5, 1:11] = 1
    expected[2, 1:5, 1:7] = 1
    assert torch.equal(mask, expected)
    assert image.dtype == mask.dtype == torch.float32
    assert image.device.type == mask.device.type == "cpu"
    assert torch.equal(image, expected.unsqueeze(-1) * torch.tensor([208, 32, 112]) / 255)


def test_region_union_fill_and_source_preservation() -> None:
    packet = _packet()
    before = copy.deepcopy(packet)
    mask = MediaPipeFaceMask.execute(face_landmarks=packet, regions={"regions": "all"})["MASK"]
    _assert_pixels(_visualize(packet), mask)
    lips = MediaPipeFaceMask.execute(
        face_landmarks=packet, regions={"regions": "custom", "lips": True}
    )["MASK"]
    expected = torch.zeros(3, 7, 11)
    expected[0, 2:4, 2:4] = expected[0, 2:4, 6:8] = expected[2, 2:4, 2:4] = 1
    assert torch.equal(lips, expected)
    assert torch.equal(
        MediaPipeFaceMask.execute(
            face_landmarks=packet, regions="custom", **{"regions.lips": True}
        )["MASK"],
        expected,
    )
    assert (
        torch.count_nonzero(
            MediaPipeFaceMask.execute(face_landmarks=packet, regions={"regions": "custom"})["MASK"]
        )
        == 0
    )
    for original, actual in zip(before["frames"], packet["frames"], strict=True):
        for a, b in zip(original, actual, strict=True):
            np.testing.assert_array_equal(a["landmarks_xy"], b["landmarks_xy"])


def test_edges_points_background_and_empty_batches() -> None:
    packet = _packet()
    background = torch.linspace(-0.2, 1.2, 4 * 7 * 11 * 4).reshape(4, 7, 11, 4)
    before = background.clone()
    output = _visualize(
        packet,
        image=background,
        thickness=0,
        point_size=0,
        connections={"connections": "all"},
        color="invalid-color",
    )
    expected = before[..., :3].mul(255).add(0.5).clamp(0, 255).to(torch.uint8).float().div(255)
    assert torch.equal(output, expected) and torch.equal(background, before)
    output = _visualize(
        packet,
        image=background,
        thickness=1,
        point_size=0,
        connections={"connections": "custom", "left_eyebrow": True},
        color="invalid-color",
    )
    expected[0, 2, 2:4] = expected[0, 2, 6:8] = expected[2, 2, 2:4] = torch.tensor([0.0, 1.0, 0.0])
    assert torch.equal(output, expected) and torch.equal(background, before)
    for size in (0, 1, 3):
        output = _visualize(
            packet, thickness=0, point_size=size, connections={"connections": "custom"}
        )
        assert torch.count_nonzero(output[1]) == 0
        if size:
            assert torch.equal(output[2, 1, 1], torch.tensor([208, 32, 112]) / 255)
        else:
            assert torch.count_nonzero(output) == 0
    # The out-of-range tessellation edge is ignored, not indexed.
    assert _visualize(packet, connections={"connections": "custom", "tesselation": True}).shape == (
        3,
        7,
        11,
        3,
    )
    empty = {**packet, "frames": []}
    assert MediaPipeFaceMask.execute(face_landmarks=empty, regions={"regions": "all"})[
        "MASK"
    ].shape == (0, 7, 11)
    assert _visualize(empty).shape == (0, 7, 11, 3)


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="face-fixture.source",
            inputs=(),
            outputs=(OutputSpec("faces", TypeExpr.concrete(FACE_LANDMARKS_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(faces=_packet())


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="face-fixture.inspect",
            inputs=(
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec("mask", TypeExpr.concrete(MASK_TYPE)),
            ),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, image: Any, mask: Any) -> Any:
        _assert_pixels(image, mask)
        return cls.outputs(result="face pixels preserved")


@pytest.mark.parametrize("in_process", (True, False))
def test_face_drawing_ordinary_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    module = f"face_fixture_{int(in_process)}"
    (tmp_path / f"{module}.py").write_text(
        "from test_mediapipe_torch import Source, Inspect\n"
        "from dinkster_api.v1 import register_inference_types\n"
        "NODES = (Source, Inspect)\n",
        encoding="ascii",
    )
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(
        '[pack]\nname="face-fixture"\npublisher="dinkster"\nnamespaces=["face-fixture"]\n'
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
                        manifest=manifest, worker_group="face", group_manifests=(native, fixture)
                    )
                )
            graph = graph_from_wire(
                graph_to_wire(
                    Graph(
                        nodes={
                            "source": GraphNode("face-fixture.source"),
                            "visualize": GraphNode(
                                "MediaPipeFaceMeshVisualize",
                                {
                                    "face_landmarks": Link("source", "faces"),
                                    "color": "#d02070",
                                },
                                slot_variants={"connections": "fill"},
                            ),
                            "mask": GraphNode(
                                "MediaPipeFaceMask",
                                {"face_landmarks": Link("source", "faces")},
                                slot_variants={"regions": "all"},
                            ),
                            "inspect": GraphNode(
                                "face-fixture.inspect",
                                {"image": Link("visualize", "IMAGE"), "mask": Link("mask", "MASK")},
                            ),
                        }
                    )
                )
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["result"].resolve() == "face pixels preserved"
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())


def _detection_face(x: float) -> dict[str, Any]:
    return {
        "bbox_xyxy": np.array([x, 2, x + 3, 6], np.float32),
        "landmarks_xy": np.array([[x, 2], [x + 3, 6]], np.float32),
        "landmarks_3d": np.array([[x, 2, -1], [x + 3, 6, 1]], np.float32),
        "blendshapes": {"jawOpen": x / 20},
        "presence": 0.9,
        "score": 0.8,
    }


class DetectionModel:
    canonical_data = {"probe": 7}
    connection_sets = {"nose": frozenset({(0, 1)})}

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def detect_batch(self, images: Any, num_faces: int, score_thresh: float, variant: str) -> Any:
        self.requests.append((len(images), num_faces, score_thresh, variant))
        result = []
        for image in images:
            index = int(image[0, 0, 0])
            assert image.shape == (7, 11, 3) and image.dtype == np.uint8
            assert (image[..., 1] == 23).all() and (image[..., 2] == 51).all()
            faces = [] if index in (0, 2, 17) else [_detection_face(index)]
            if variant == "full" and index == 1:
                faces.append(_detection_face(9))
            result.append(faces[:num_faces] if num_faces else faces)
        return result


def _geometry_probe(face: Any, width: int, height: int, canonical: Any) -> Any:
    assert (width, height) == (11, 7) and canonical == {"probe": 7}
    return np.diag([width, height, face["bbox_xyxy"][0], 1]).astype(np.float32)


@pytest.mark.parametrize("variant", ("short", "full", "both"))
@pytest.mark.parametrize("fallback", ("empty", "previous", "interpolate"))
@pytest.mark.parametrize("cap", (0, 1))
def test_face_detection_batches_selection_fallback_and_source(
    monkeypatch: pytest.MonkeyPatch, variant: str, fallback: str, cap: int
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference.mediapipe.face_geometry",
        SimpleNamespace(transformation_matrix_from_detection=_geometry_probe),
    )
    images = torch.zeros(18, 7, 11, 4)
    images[..., 0] = torch.arange(18)[:, None, None] / 255
    images[..., 1] = 23 / 255
    images[..., 2] = 51 / 255
    images[..., 3] = 0.7
    original = images.clone()
    model = DetectionModel()
    result = MediaPipeFaceLandmarker.execute(
        face_detection_model=model,
        image=images,
        detector_variant=variant,
        num_faces=cap,
        min_confidence=0.37,
        missing_frame_fallback=fallback,
    )
    packet = result["face_landmarks"]
    frames = packet["frames"]
    assert packet["image_size"] == (7, 11) and packet["connection_sets"] == model.connection_sets
    selected_full = variant in ("full", "both") and cap == 0
    assert len(frames[1]) == (2 if selected_full else 1)
    assert frames[1][0]["bbox_xyxy"][0] == 1
    assert frames[3][0]["bbox_xyxy"][0] == 3
    if fallback == "empty":
        assert frames[0] == frames[2] == frames[17] == []
    elif fallback == "previous":
        assert frames[0] == []
        assert frames[2][0]["bbox_xyxy"][0] == 1
        assert frames[17][0]["bbox_xyxy"][0] == 16
    else:
        assert frames[0][0]["bbox_xyxy"][0] == 1
        assert len(frames[2]) == 1
        assert frames[2][0]["bbox_xyxy"][0] == 2
        assert frames[17][0]["bbox_xyxy"][0] == 16
        np.testing.assert_allclose(frames[2][0]["blendshapes"]["jawOpen"], 0.1)
    for faces, boxes in zip(frames, result["bboxes"], strict=True):
        for face, box in zip(faces, boxes, strict=True):
            x = float(face["bbox_xyxy"][0])
            assert box == {
                "x": x,
                "y": 2.0,
                "width": 3.0,
                "height": 4.0,
                "label": "face",
                "score": 0.8,
            }
            np.testing.assert_array_equal(face["transformation_matrix"], np.diag([11, 7, x, 1]))
    variants = ("short", "full") if variant == "both" else (variant,)
    assert model.requests == [(size, cap, 0.37, name) for name in variants for size in (16, 2)]
    assert torch.equal(images, original)


@pytest.mark.parametrize("batch", (0, 3))
@pytest.mark.parametrize("fallback", ("empty", "previous", "interpolate"))
def test_face_detection_empty_frames(
    monkeypatch: pytest.MonkeyPatch, batch: int, fallback: str
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference.mediapipe.face_geometry",
        SimpleNamespace(transformation_matrix_from_detection=_geometry_probe),
    )
    model = SimpleNamespace(
        canonical_data={"probe": 7},
        connection_sets={},
        detect_batch=lambda images, **kwargs: [[] for _ in images],
    )
    result = MediaPipeFaceLandmarker.execute(
        face_detection_model=model,
        image=torch.zeros(batch, 7, 11, 3),
        detector_variant="both",
        missing_frame_fallback=fallback,
    )
    assert result["face_landmarks"]["frames"] == [[] for _ in range(batch)]
    assert result["bboxes"] == [[] for _ in range(batch)]
    assert result["face_landmarks"]["image_size"] == (7, 11)


def test_face_loader_bound_state_variants_and_real_patchers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_api.v1 import initialize_inference_process
    from safetensors.torch import save_file

    initialize_inference_process()

    class Landmarker(torch.nn.Module):
        def __init__(
            self, *, device: Any, dtype: Any, operations: Any, detector_variant: str
        ) -> None:
            super().__init__()
            assert operations is None and dtype == torch.float32
            self.variant = detector_variant
            for name in ("detector", "mesh", "blendshapes"):
                module = torch.nn.Module()
                module.register_buffer("probe", torch.empty(1, device=device, dtype=dtype))
                self.add_module(name, module)

    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference.mediapipe.face_landmarker",
        SimpleNamespace(FaceLandmarker=Landmarker),
    )
    state = {
        "detector_short.probe": torch.tensor([3.0]),
        "detector_full.probe": torch.tensor([7.0]),
        "mesh.probe": torch.tensor([11.0]),
        "blendshapes.probe": torch.tensor([0.25]),
        "canonical_vertices": torch.arange(15, dtype=torch.float32).reshape(5, 3),
        "procrustes_indices": torch.tensor([0, 2, 4]),
        "procrustes_weights": torch.tensor([0.5, 1.0, 2.0]),
    }
    for index, name in enumerate(
        (
            "face_oval",
            "left_eye",
            "right_eye",
            "left_eyebrow",
            "right_eyebrow",
            "lips",
            "irises",
            "nose",
        )
    ):
        state[f"topology.{name}"] = torch.tensor([[index, index + 1]], dtype=torch.int32)
    path = tmp_path / "face.safetensors"
    save_file(state, path)
    data = path.read_bytes()
    vault = AssetVault(tmp_path / "vault")
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    model = LoadMediaPipeFaceLandmarker.execute(
        model_name=AssetRef(digest, path.name, len(data), resolver=vault)
    )["FACE_DETECTION_MODEL"]
    from dinkster_inference.model_patcher import CoreModelPatcher

    for variant, value in (("short", 3), ("full", 7)):
        assert isinstance(model.patchers[variant], CoreModelPatcher)
        assert model.models[variant].state_dict()["detector.probe"].item() == value
        assert model.models[variant].state_dict()["mesh.probe"].item() == 11
        assert model.models[variant].state_dict()["blendshapes.probe"].item() == 0.25
        assert model.models[variant].training is False
    assert model.connection_sets["all"] == frozenset((i, i + 1) for i in range(8))
    np.testing.assert_array_equal(
        model.canonical_data["canonical_vertices"], state["canonical_vertices"].numpy()
    )
    assert path.read_bytes() == data
