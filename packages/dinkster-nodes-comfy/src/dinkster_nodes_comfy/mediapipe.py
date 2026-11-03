"""MediaPipe face landmark visualization and region masks."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    COLOR_TYPE,
    CORE_BOOLEAN,
    CORE_INT,
    FACE_LANDMARKS_TYPE,
    IMAGE_TYPE,
    MASK_TYPE,
    ColorWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_vision_packet,
)

_CONTOURS = ("face_oval", "lips", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow")
_ALL_PARTS = (*_CONTOURS, "irises", "nose")
_CUSTOM_FEATURES = (*((part, True) for part in _ALL_PARTS), ("tesselation", False))
_MASK_REGIONS = ("face_oval", "lips", "left_eye", "right_eye", "irises")


def _runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    initialize_inference_process()
    return (
        importlib.import_module("torch"),
        importlib.import_module("numpy"),
        importlib.import_module("PIL.Image"),
        importlib.import_module("PIL.ImageDraw"),
        importlib.import_module("PIL.ImageColor"),
        importlib.import_module("dinkster_inference.model_management"),
    )


def _ordered_rings(edges: Any) -> list[list[int]]:
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    visited: set[int] = set()
    rings: list[list[int]] = []
    for start in adjacency:
        if start in visited:
            continue
        ring = [start]
        visited.add(start)
        previous, current = -1, start
        while True:
            following = next((v for v in adjacency[current] if v != previous), None)
            if following is None or following == start:
                break
            ring.append(following)
            visited.add(following)
            previous, current = current, following
        rings.append(ring)
    return rings


class MediaPipeFaceMeshVisualize(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MediaPipeFaceMeshVisualize",
            display_name="Visualize Face Landmarks (MediaPipe)",
            category="image/detection",
            description="Draws face landmarks mesh on the input image.",
            search_terms=(
                "face",
                "facial",
                "mediapipe",
                "face landmark",
                "face mesh",
                "blazeface",
                "face detection",
                "visualize",
            ),
            inputs=(
                InputSpec("face_landmarks", TypeExpr.concrete(FACE_LANDMARKS_TYPE)),
                InputSpec(
                    "image",
                    TypeExpr.concrete(IMAGE_TYPE),
                    required=False,
                    doc="If not connected, a black canvas will be used.",
                ),
                InputSpec(
                    "color", TypeExpr.concrete(COLOR_TYPE), default="#00ff00", widget=ColorWidget()
                ),
                InputSpec(
                    "thickness",
                    TypeExpr.concrete(CORE_INT),
                    default=1,
                    widget=NumberWidget(min=0, max=8, step=1),
                    doc="Edge line thickness in pixels. 0 disables edge drawing.",
                ),
                InputSpec(
                    "point_size",
                    TypeExpr.concrete(CORE_INT),
                    default=2,
                    widget=NumberWidget(min=0, max=16, step=1),
                    doc="Landmark dot radius in pixels. 0 disables point drawing.",
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "connections",
                    (
                        DynamicComboOption("all"),
                        DynamicComboOption("fill"),
                        DynamicComboOption(
                            "custom",
                            tuple(
                                InputSpec(
                                    feature,
                                    TypeExpr.concrete(CORE_BOOLEAN),
                                    default=default,
                                    doc=f"Draw the '{feature}' connection set.",
                                )
                                for feature, default in _CUSTOM_FEATURES
                            ),
                        ),
                    ),
                    doc="'all' = oval+eyes+brows+lips+irises+nose. 'fill' = solid face_oval "
                    "polygon (silhouette mask). 'custom' = toggle each feature individually "
                    "(including 'tesselation', the full 2547-edge wireframe).",
                ),
            ),
            input_order=(
                "face_landmarks",
                "image",
                "connections",
                "color",
                "thickness",
                "point_size",
            ),
            outputs=(OutputSpec("IMAGE", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        face_landmarks: Any,
        connections: str | Mapping[str, Any],
        color: str,
        thickness: int,
        point_size: int,
        image: Any = None,
        **inputs: Any,
    ) -> Any:
        torch, np, image_module, image_draw, image_color, management = _runtime()
        if isinstance(connections, str):
            connections = {
                "connections": connections,
                **{key.removeprefix("connections."): value for key, value in inputs.items()},
            }
        packet = materialize_vision_packet(face_landmarks)
        sets = packet["connection_sets"]
        selected = connections["connections"]
        rings = None
        edges: frozenset[tuple[int, int]]
        if selected == "fill":
            rings = _ordered_rings(sets["face_oval"])
            edges = frozenset()
        elif selected == "custom":
            edges = frozenset[tuple[int, int]]().union(
                *(sets[part] for part, _ in _CUSTOM_FEATURES if connections.get(part, False))
            )
        else:
            edges = frozenset[tuple[int, int]]().union(*(sets[part] for part in _ALL_PARTS))
        try:
            rgb = image_color.getrgb(color)[:3]
        except ValueError:
            rgb = (0, 255, 0)
        frames = packet["frames"]
        if image is None:
            height, width = packet["image_size"]
            images = np.zeros((len(frames), height, width, 3), dtype=np.uint8)
        else:
            images = (
                materialize_image(image)[..., :3]
                .mul(255.0)
                .add_(0.5)
                .clamp_(0, 255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
        out = np.empty_like(images)
        thick, size = int(thickness), int(point_size)
        for index, background in enumerate(images):
            faces: list[Any] = frames[index] if index < len(frames) else []
            if not faces or (rings is None and not (thick > 0 and edges) and size <= 0):
                out[index] = background
                continue
            canvas = image_module.fromarray(background)
            draw = image_draw.Draw(canvas)
            for face in faces:
                landmarks = face["landmarks_xy"]
                if rings is not None:
                    for ring in rings:
                        draw.polygon(
                            [(float(landmarks[i, 0]), float(landmarks[i, 1])) for i in ring],
                            fill=rgb,
                        )
                    continue
                if thick > 0:
                    for a, b in edges:
                        if a < len(landmarks) and b < len(landmarks):
                            draw.line(
                                [
                                    (float(landmarks[a, 0]), float(landmarks[a, 1])),
                                    (float(landmarks[b, 0]), float(landmarks[b, 1])),
                                ],
                                fill=rgb,
                                width=thick,
                            )
                if size == 1:
                    draw.point(landmarks.flatten().tolist(), fill=rgb)
                elif size > 1:
                    radius = size * 0.5
                    for x, y in landmarks:
                        draw.ellipse(
                            (
                                float(x) - radius,
                                float(y) - radius,
                                float(x) + radius,
                                float(y) + radius,
                            ),
                            fill=rgb,
                        )
            out[index] = np.asarray(canvas)
        result = (
            torch.from_numpy(out)
            .to(device=management.intermediate_device(), dtype=management.intermediate_dtype())
            .div_(255.0)
        )
        return cls.outputs(IMAGE=result)


class MediaPipeFaceMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MediaPipeFaceMask",
            display_name="Draw Face Mask (MediaPipe)",
            category="image/detection",
            description="Draws a mask from face landmarks.",
            search_terms=(
                "face",
                "facial",
                "mediapipe",
                "face mask",
                "blazeface",
                "face detection",
                "visualize",
            ),
            inputs=(InputSpec("face_landmarks", TypeExpr.concrete(FACE_LANDMARKS_TYPE)),),
            combos=(
                DynamicComboSpec(
                    "regions",
                    (
                        DynamicComboOption("all"),
                        DynamicComboOption(
                            "custom",
                            tuple(
                                InputSpec(
                                    region,
                                    TypeExpr.concrete(CORE_BOOLEAN),
                                    default=region == "face_oval",
                                    doc=f"Include the '{region}' region in the mask.",
                                )
                                for region in _MASK_REGIONS
                            ),
                        ),
                    ),
                    doc="'all' = union of face_oval+lips+eyes+irises (which collapses to "
                    "face_oval since it encloses the rest). 'custom' = toggle each region "
                    "individually for combos like lips+eyes.",
                ),
            ),
            input_order=("face_landmarks", "regions"),
            outputs=(OutputSpec("MASK", TypeExpr.concrete(MASK_TYPE)),),
        )

    @classmethod
    def execute(
        cls, *, face_landmarks: Any, regions: str | Mapping[str, Any], **inputs: Any
    ) -> Any:
        torch, np, image_module, image_draw, _, management = _runtime()
        if isinstance(regions, str):
            regions = {
                "regions": regions,
                **{key.removeprefix("regions."): value for key, value in inputs.items()},
            }
        packet = materialize_vision_packet(face_landmarks)
        sets = packet["connection_sets"]
        picked = (
            [region for region in _MASK_REGIONS if regions.get(region, False)]
            if regions["regions"] == "custom"
            else _MASK_REGIONS
        )
        rings = [ring for region in picked for ring in _ordered_rings(sets[region])]
        height, width = packet["image_size"]
        frames = packet["frames"]
        masks = np.zeros((len(frames), height, width), dtype=np.uint8)
        for index, faces in enumerate(frames):
            if not faces:
                continue
            canvas = image_module.new("L", (width, height), 0)
            draw = image_draw.Draw(canvas)
            for face in faces:
                landmarks = face["landmarks_xy"]
                for ring in rings:
                    draw.polygon(
                        [(float(landmarks[i, 0]), float(landmarks[i, 1])) for i in ring], fill=255
                    )
            masks[index] = np.asarray(canvas)
        result = (
            torch.from_numpy(masks)
            .to(device=management.intermediate_device(), dtype=management.intermediate_dtype())
            .div_(255.0)
        )
        return cls.outputs(MASK=result)
