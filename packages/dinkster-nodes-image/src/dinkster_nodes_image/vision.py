"""Stable schemas for model-backed vision capabilities.

Each node executes only through an installed vision provider pack; these
owner schemas keep the graph contract stable while providers come and go.
"""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

from .geometry import IMAGE
from .support import combo_input as _combo
from .support import number_input as _number
from .types import DETECTION_TYPE

MASK = TypeExpr.concrete("dinkster.mask")
DETECTION = TypeExpr.concrete(DETECTION_TYPE)
DETECTIONS = TypeExpr.list_of(DETECTION)
MASKS = TypeExpr.list_of(MASK)
COMBO = TypeExpr.concrete(CORE_COMBO)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
INT = TypeExpr.concrete(CORE_INT)
STRING = TypeExpr.concrete(CORE_STRING)
_PROMPT_MODES = ("comma-separated", "literal")
_RESULT_LIMIT_MODES = ("count", "slice-stop")

DETECT_PROVIDER_CHOICE = "dinkster.detection.detect.providers"
SEGMENT_PROVIDER_CHOICE = "dinkster.detection.segment.providers"
TEXT_SEGMENT_PROVIDER_CHOICE = "dinkster.detection.segment_text.providers"
MATTE_PROVIDER_CHOICE = "dinkster.image.matte.providers"
TRACK_PROVIDER_CHOICE = "dinkster.detection.track.providers"


def _provider_input(choice: str) -> InputSpec:
    return InputSpec(
        "provider",
        COMBO,
        required=False,
        widget=ComboWidget(remote_route=f"/api/choices/{choice}"),
        hidden=True,
    )


def _model_input(options: tuple[ComboOption, ...]) -> InputSpec:
    return InputSpec(
        "model",
        COMBO,
        required=False,
        default="auto",
        widget=ComboWidget(options=(ComboOption("auto", "Automatic"), *options)),
    )


class DetectObjects(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.detection.detect",
            display_name="Detect Objects",
            category="image/detection",
            description=(
                "Finds objects in each image with a detection model, producing "
                "labeled and scored detections. The prompt names the classes or "
                "phrases to find; providers that detect a fixed class set use it "
                "as a label filter, and an empty prompt keeps every class. Prompt "
                "mode controls whether commas separate phrases or remain literal."
            ),
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec("prompt", STRING, required=False, default="", widget=StringWidget()),
                _combo("prompt_mode", _PROMPT_MODES, "comma-separated", advanced=True),
                _number("min_score", FLOAT, 0.5, step=0.01),
                _number("max_results", INT, -1, step=1),
                _combo("result_limit_mode", _RESULT_LIMIT_MODES, "count", advanced=True),
                _model_input(
                    (
                        ComboOption("sam-3.1", "SAM 3.1"),
                        ComboOption("detr-resnet-50", "DETR ResNet-50"),
                        ComboOption("rtdetr-v4-x-hgnet", "RT-DETR v4 x-HGNet"),
                    )
                ),
                _provider_input(DETECT_PROVIDER_CHOICE),
            ),
            outputs=(
                OutputSpec("detections", DETECTIONS),
                OutputSpec("count", INT),
            ),
            search_terms=(
                "object detection",
                "GroundingDINO",
                "YOLO",
                "RTDETR",
                "Florence2",
                "text prompted detection",
            ),
        )

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("object detection requires an installed provider")


class SegmentDetections(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.detection.segment",
            display_name="Segment Detections",
            category="image/detection",
            description=(
                "Segments each detection's object with a promptable segmentation "
                "model, using the detection boxes as prompts. Returns the same "
                "detections carrying full-frame soft masks, plus the masks as a "
                "list."
            ),
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec("detections", DETECTIONS),
                _model_input(
                    (
                        ComboOption("sam-3.1", "SAM 3.1"),
                        ComboOption("efficient-sam-ti", "EfficientSAM-Ti"),
                    )
                ),
                _provider_input(SEGMENT_PROVIDER_CHOICE),
            ),
            outputs=(
                OutputSpec("detections", DETECTIONS),
                OutputSpec("masks", MASKS),
            ),
            search_terms=(
                "SAM",
                "segment anything",
                "SAMDetector",
                "box prompt segmentation",
            ),
        )

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("prompted segmentation requires an installed provider")


class SegmentByText(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.detection.segment_text",
            display_name="Segment Objects by Text",
            category="image/detection",
            description=(
                "Finds and segments each object named by the prompt in one model "
                "operation. Returns labeled and scored detections carrying "
                "full-frame masks, plus the masks as a list. Prompt mode controls "
                "whether commas separate phrases or remain literal."
            ),
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec("prompt", STRING, required=False, default="", widget=StringWidget()),
                _combo("prompt_mode", _PROMPT_MODES, "comma-separated", advanced=True),
                _number("min_score", FLOAT, 0.5, minimum=0.0, maximum=1.0, step=0.01),
                _provider_input(TEXT_SEGMENT_PROVIDER_CHOICE),
            ),
            outputs=(
                OutputSpec("detections", DETECTIONS),
                OutputSpec("masks", MASKS),
            ),
            search_terms=(
                "text prompted segmentation",
                "GroundingDINO SAM",
                "Florence2 SAM",
                "segment by phrase",
            ),
        )

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("text-prompted segmentation requires an installed provider")


class ImageMatte(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.image.matte",
            display_name="Image Matte",
            category="image/matte",
            description=(
                "Estimates a foreground alpha matte for each image with a "
                "matting model. The mask is the soft foreground coverage; "
                "combine it with the input through alpha join or compositing "
                "for background removal."
            ),
            inputs=(
                InputSpec("image", IMAGE),
                _provider_input(MATTE_PROVIDER_CHOICE),
            ),
            outputs=(
                OutputSpec(
                    "mask", MASK, preview=True, mask_polarity="coverage", mask_semantic="alpha"
                ),
            ),
            search_terms=(
                "background removal",
                "BiRefNet",
                "rembg",
                "alpha matte",
                "remove background",
            ),
        )

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("foreground matting requires an installed provider")


class TrackObjects(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.detection.track",
            display_name="Track Objects",
            category="image/detection",
            description=(
                "Tracks objects across a batch of video frames with a video "
                "segmentation model. First-frame detections or masks prompt the "
                "objects; each tracked object yields one mask batched across every "
                "frame, plus their combined union."
            ),
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec("detections", DETECTIONS, required=False),
                InputSpec("initial_masks", MASK, required=False),
                _provider_input(TRACK_PROVIDER_CHOICE),
            ),
            outputs=(
                OutputSpec("masks", MASKS),
                OutputSpec("combined", MASK, preview=True),
            ),
            search_terms=(
                "video object tracking",
                "SAM3 video track",
                "video segmentation",
                "track anything",
            ),
        )

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("video object tracking requires an installed provider")


def vision_choices() -> dict[str, tuple[str, ...]]:
    return {
        DETECT_PROVIDER_CHOICE: (),
        SEGMENT_PROVIDER_CHOICE: (),
        TEXT_SEGMENT_PROVIDER_CHOICE: (),
        MATTE_PROVIDER_CHOICE: (),
        TRACK_PROVIDER_CHOICE: (),
    }


VISION_NODES: tuple[type[Node], ...] = (
    DetectObjects,
    SegmentDetections,
    SegmentByText,
    ImageMatte,
    TrackObjects,
)


__all__ = [
    "DETECT_PROVIDER_CHOICE",
    "MATTE_PROVIDER_CHOICE",
    "SEGMENT_PROVIDER_CHOICE",
    "TEXT_SEGMENT_PROVIDER_CHOICE",
    "TRACK_PROVIDER_CHOICE",
    "VISION_NODES",
    "DetectObjects",
    "ImageMatte",
    "SegmentByText",
    "SegmentDetections",
    "TrackObjects",
    "vision_choices",
]
