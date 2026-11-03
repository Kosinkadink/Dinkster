"""Load the PyTorch face landmarker and detect batched face geometry."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    BOUNDING_BOX_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    FACE_DETECTION_MODEL_TYPE,
    FACE_LANDMARKS_TYPE,
    IMAGE_TYPE,
    AssetRef,
    AssetWidget,
    ComboWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    report_progress,
)

_SEARCH = (
    "face",
    "facial",
    "mediapipe",
    "face landmark",
    "face mesh",
    "blazeface",
    "face detection",
)
_CANONICAL = ("canonical_vertices", "procrustes_indices", "procrustes_weights")
_CONTOURS = ("face_oval", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "lips")


class FaceLandmarkerModel:
    def __init__(self, state_dict: dict[str, Any]) -> None:
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        patcher = importlib.import_module("dinkster_inference.model_patcher")
        storage = importlib.import_module("dinkster_inference.storage")
        landmarker = importlib.import_module("dinkster_inference.mediapipe.face_landmarker")
        self.load_device = management.text_encoder_device()
        offload_device = management.text_encoder_offload_device()
        self.dtype = torch.float32
        base: dict[str, frozenset[tuple[int, int]]] = {
            key.removeprefix("topology."): frozenset(
                (a, b) for a, b in state_dict.pop(key).tolist()
            )
            for key in tuple(state_dict)
            if key.startswith("topology.")
        }
        base["contours"] = frozenset().union(*(base[part] for part in _CONTOURS))
        base["all"] = base["contours"] | base["irises"] | base["nose"]
        self.connection_sets = base
        self.canonical_data = {key: state_dict.pop(key).numpy() for key in _CANONICAL}
        shared = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("mesh.", "blendshapes."))
        }
        self.models: dict[str, Any] = {}
        self.patchers: dict[str, Any] = {}
        for variant in ("short", "full"):
            prefix = f"detector_{variant}."
            sub = dict(shared)
            sub.update(
                {
                    f"detector.{key[len(prefix) :]}": value
                    for key, value in state_dict.items()
                    if key.startswith(prefix)
                }
            )
            model = landmarker.FaceLandmarker(
                device=offload_device, dtype=self.dtype, operations=None, detector_variant=variant
            ).eval()
            model.load_state_dict(sub, strict=False)
            self.models[variant] = model
            self.patchers[variant] = patcher.CoreModelPatcher(
                model,
                load_device=self.load_device,
                offload_device=offload_device,
                size=management.module_size(model),
                fast_disk=storage.state_dict_fast_disk(sub),
            )

    def detect_batch(self, images: Any, num_faces: int, score_thresh: float, variant: str) -> Any:
        management = importlib.import_module("dinkster_inference.model_management")
        management.load_model_gpu(self.patchers[variant])
        return self.models[variant].detect_batch(
            images, num_faces=num_faces, score_thresh=score_thresh
        )


def _copy_face(face: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox_xyxy": face["bbox_xyxy"].copy(),
        "blendshapes": dict(face["blendshapes"]),
        "landmarks_xy": face["landmarks_xy"].copy(),
        "landmarks_3d": face["landmarks_3d"].copy(),
        "presence": face["presence"],
        "score": face["score"],
    }


def _fill_missing_frames(frames: list[list[dict[str, Any]]], mode: str) -> None:
    if mode == "empty":
        return
    valid = [index for index, frame in enumerate(frames) if frame]
    if not valid:
        return
    if mode == "previous":
        last = []
        for index, frame in enumerate(frames):
            if frame:
                last = frame
            elif last:
                frames[index] = [_copy_face(face) for face in last]
        return
    np = importlib.import_module("numpy")
    for index, frame in enumerate(frames):
        if frame:
            continue
        prev = max((v for v in valid if v < index), default=None)
        following = min((v for v in valid if v > index), default=None)
        if prev is None:
            frames[index] = [_copy_face(face) for face in frames[valid[0]]]
        elif following is None:
            frames[index] = [_copy_face(face) for face in frames[prev]]
        else:
            a, b = frames[prev], frames[following]
            centers_a = np.array(
                [
                    (
                        0.5 * (f["bbox_xyxy"][0] + f["bbox_xyxy"][2]),
                        0.5 * (f["bbox_xyxy"][1] + f["bbox_xyxy"][3]),
                    )
                    for f in a
                ]
            )
            centers_b = np.array(
                [
                    (
                        0.5 * (f["bbox_xyxy"][0] + f["bbox_xyxy"][2]),
                        0.5 * (f["bbox_xyxy"][1] + f["bbox_xyxy"][3]),
                    )
                    for f in b
                ]
            )
            distances = np.linalg.norm(centers_a[:, None] - centers_b[None], axis=-1)
            used_a: set[int] = set()
            used_b: set[int] = set()
            t = (index - prev) / (following - prev)
            for _, ia, ib in sorted(
                (distances[ia, ib], ia, ib) for ia in range(len(a)) for ib in range(len(b))
            ):
                if ia in used_a or ib in used_b:
                    continue
                used_a.add(ia)
                used_b.add(ib)
                frames[index].append(
                    {
                        **{
                            key: (1 - t) * a[ia][key] + t * b[ib][key]
                            for key in (
                                "bbox_xyxy",
                                "landmarks_xy",
                                "landmarks_3d",
                                "presence",
                                "score",
                            )
                        },
                        "blendshapes": {
                            key: (1 - t) * a[ia]["blendshapes"][key] + t * b[ib]["blendshapes"][key]
                            for key in a[ia]["blendshapes"]
                        },
                    }
                )


class LoadMediaPipeFaceLandmarker(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "LoadMediaPipeFaceLandmarker",
            display_name="Load Face Detection Model (MediaPipe)",
            category="model/loaders",
            search_terms=_SEARCH,
            inputs=(
                InputSpec(
                    "model_name",
                    TypeExpr.concrete(ASSET_TYPE),
                    widget=AssetWidget(
                        accept=("application/octet-stream",), kind="model/detection"
                    ),
                    doc="Face detection model from models/detection/.",
                ),
            ),
            outputs=(
                OutputSpec("FACE_DETECTION_MODEL", TypeExpr.concrete(FACE_DETECTION_MODEL_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, *, model_name: Any) -> Mapping[str, object]:
        if not isinstance(model_name, (AssetRef, LocalAssetRef)):
            raise TypeError("model_name must be an AssetRef or LocalAssetRef")
        path = str(model_name.local_path())
        initialize_inference_process()
        state = importlib.import_module("dinkster_inference.utils").load_torch_file(
            path, safe_load=True
        )
        return cls.outputs(FACE_DETECTION_MODEL=FaceLandmarkerModel(state))


class MediaPipeFaceLandmarker(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "MediaPipeFaceLandmarker",
            display_name="Detect Face Landmarks (MediaPipe)",
            category="image/detection",
            search_terms=_SEARCH,
            description="Detects facial landmarks using MediaPipe model.",
            inputs=(
                InputSpec("face_detection_model", TypeExpr.concrete(FACE_DETECTION_MODEL_TYPE)),
                InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
                InputSpec(
                    "detector_variant",
                    TypeExpr.concrete(CORE_COMBO),
                    default="short",
                    widget=ComboWidget(("short", "full", "both")),
                ),
                InputSpec(
                    "num_faces",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=0, max=16, step=1),
                ),
                InputSpec(
                    "min_confidence",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                    advanced=True,
                ),
                InputSpec(
                    "missing_frame_fallback",
                    TypeExpr.concrete(CORE_COMBO),
                    default="empty",
                    widget=ComboWidget(("empty", "previous", "interpolate")),
                    advanced=True,
                ),
            ),
            outputs=(
                OutputSpec(
                    "face_landmarks",
                    TypeExpr.concrete(FACE_LANDMARKS_TYPE),
                    display_name="face_landmarks",
                ),
                OutputSpec("bboxes", TypeExpr.concrete(BOUNDING_BOX_TYPE)),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        face_detection_model: Any,
        image: Any,
        detector_variant: str = "short",
        num_faces: int = 1,
        min_confidence: float = 0.5,
        missing_frame_fallback: str = "empty",
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        geometry = importlib.import_module("dinkster_inference.mediapipe.face_geometry")
        image = materialize_image(image)
        images = image[..., :3].mul(255.0).add_(0.5).clamp_(0, 255).to(torch.uint8).cpu().numpy()
        batch, height, width = images.shape[:3]
        both = detector_variant == "both"
        total = batch * (2 if both else 1)
        completed = 0

        def run(variant: str) -> list[list[dict[str, Any]]]:
            nonlocal completed
            result: list[list[dict[str, Any]]] = []
            for start in range(0, batch, 16):
                end = min(start + 16, batch)
                result.extend(
                    face_detection_model.detect_batch(
                        [images[i] for i in range(start, end)],
                        num_faces=int(num_faces),
                        score_thresh=float(min_confidence),
                        variant=variant,
                    )
                )
                completed += end - start
                report_progress(completed, total)
            return result

        if both:
            short, full = run("short"), run("full")
            frames = [short[i] if len(short[i]) >= len(full[i]) else full[i] for i in range(batch)]
        else:
            frames = run(detector_variant)
        _fill_missing_frames(frames, missing_frame_fallback)
        bboxes: list[list[dict[str, Any]]] = []
        for frame in frames:
            boxes: list[dict[str, Any]] = []
            for face in frame:
                face["transformation_matrix"] = geometry.transformation_matrix_from_detection(
                    face, width, height, face_detection_model.canonical_data
                )
                x1, y1, x2, y2 = (float(v) for v in face["bbox_xyxy"])
                boxes.append(
                    {
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "label": "face",
                        "score": float(face["score"]),
                    }
                )
            bboxes.append(boxes)
        return cls.outputs(
            face_landmarks={
                "frames": frames,
                "image_size": (height, width),
                "connection_sets": face_detection_model.connection_sets,
            },
            bboxes=bboxes,
        )
