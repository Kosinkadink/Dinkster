"""Stable node shape and boundary codecs for the SAM 3.1 provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dinkster_api.v1 import (
    ABSENT,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    DETECTION_TYPE,
    REGION_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    TypeRegistry,
    coerce_detection,
    coerce_region,
    decode_detection,
    decode_image_array,
    decode_region,
    detection_meta,
    encode_detection,
    encode_image_array,
    encode_region,
    image_array_fingerprint,
    image_array_meta,
    image_input,
    mask_array_meta,
    prepare_image_array_encoding,
    region_meta,
)

IMAGE_TYPE = "dinkster.image"
MASK_TYPE = "dinkster.mask"
IMAGE = TypeExpr.concrete(IMAGE_TYPE)
MASK = TypeExpr.concrete(MASK_TYPE)
MASKS = TypeExpr.list_of(TypeExpr.concrete(MASK_TYPE))
DETECTIONS = TypeExpr.list_of(TypeExpr.concrete(DETECTION_TYPE))
COMBO = TypeExpr.concrete(CORE_COMBO)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
INT = TypeExpr.concrete(CORE_INT)
STRING = TypeExpr.concrete(CORE_STRING)
DETECT_PROVIDER_CHOICE = "dinkster.detection.detect.providers"
SEGMENT_PROVIDER_CHOICE = "dinkster.detection.segment.providers"
TEXT_SEGMENT_PROVIDER_CHOICE = "dinkster.detection.segment_text.providers"
TRACK_PROVIDER_CHOICE = "dinkster.detection.track.providers"
_PROMPT_MODES = ("comma-separated", "literal")
_RESULT_LIMIT_MODES = ("count", "slice-stop")


def _prompt_mode_input() -> InputSpec:
    return InputSpec(
        "prompt_mode",
        COMBO,
        required=False,
        default="comma-separated",
        widget=ComboWidget(options=_PROMPT_MODES),
        advanced=True,
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
                _prompt_mode_input(),
                InputSpec(
                    "min_score",
                    FLOAT,
                    required=False,
                    default=0.5,
                    widget=NumberWidget(step=0.01),
                ),
                InputSpec(
                    "max_results",
                    INT,
                    required=False,
                    default=-1,
                    widget=NumberWidget(step=1),
                ),
                InputSpec(
                    "result_limit_mode",
                    COMBO,
                    required=False,
                    default="count",
                    widget=ComboWidget(options=_RESULT_LIMIT_MODES),
                    advanced=True,
                ),
                InputSpec(
                    "model",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(
                        options=(
                            ComboOption("auto", "Automatic"),
                            ComboOption("sam-3.1", "SAM 3.1"),
                            ComboOption("detr-resnet-50", "DETR ResNet-50"),
                            ComboOption("rtdetr-v4-x-hgnet", "RT-DETR v4 x-HGNet"),
                        )
                    ),
                ),
                InputSpec(
                    "provider",
                    COMBO,
                    required=False,
                    widget=ComboWidget(remote_route=f"/api/choices/{DETECT_PROVIDER_CHOICE}"),
                    hidden=True,
                ),
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
    def execute(
        cls,
        *,
        image: object,
        prompt: str = "",
        prompt_mode: str = "comma-separated",
        min_score: float = 0.5,
        max_results: int = -1,
        result_limit_mode: str = "count",
        model: str = "auto",
        provider: str = "",
    ) -> Mapping[str, object]:
        del model, provider
        from .model import execute_detect

        detections = execute_detect(
            image,
            prompt=prompt,
            prompt_mode=prompt_mode,
            min_score=min_score,
            max_results=max_results,
            result_limit_mode=result_limit_mode,
        )
        return cls.outputs(detections=detections, count=len(detections))


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
                InputSpec(
                    "model",
                    COMBO,
                    required=False,
                    default="auto",
                    widget=ComboWidget(
                        options=(
                            ComboOption("auto", "Automatic"),
                            ComboOption("sam-3.1", "SAM 3.1"),
                            ComboOption("efficient-sam-ti", "EfficientSAM-Ti"),
                        )
                    ),
                ),
                InputSpec(
                    "provider",
                    COMBO,
                    required=False,
                    widget=ComboWidget(remote_route=f"/api/choices/{SEGMENT_PROVIDER_CHOICE}"),
                    hidden=True,
                ),
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
    def execute(
        cls,
        *,
        image: object,
        detections: Sequence[object],
        model: str = "auto",
        provider: str = "",
    ) -> Mapping[str, object]:
        del model, provider
        from .model import execute_segment

        segmented, masks = execute_segment(image, detections)
        return cls.outputs(detections=segmented, masks=masks)


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
                _prompt_mode_input(),
                InputSpec(
                    "min_score",
                    FLOAT,
                    required=False,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "provider",
                    COMBO,
                    required=False,
                    widget=ComboWidget(remote_route=f"/api/choices/{TEXT_SEGMENT_PROVIDER_CHOICE}"),
                    hidden=True,
                ),
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
    def execute(
        cls,
        *,
        image: object,
        provider: str,
        prompt: str = "",
        prompt_mode: str = "comma-separated",
        min_score: float = 0.5,
    ) -> Mapping[str, object]:
        del provider
        from .model import execute_text_segment

        detections, masks = execute_text_segment(
            image,
            prompt=prompt,
            prompt_mode=prompt_mode,
            min_score=min_score,
        )
        return cls.outputs(detections=detections, masks=masks)


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
                InputSpec(
                    "provider",
                    COMBO,
                    required=False,
                    widget=ComboWidget(remote_route=f"/api/choices/{TRACK_PROVIDER_CHOICE}"),
                    hidden=True,
                ),
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
    def execute(
        cls,
        *,
        image: object,
        provider: str,
        detections: object = ABSENT,
        initial_masks: object = ABSENT,
    ) -> Mapping[str, object]:
        del provider
        from .model import execute_track

        masks, combined = execute_track(image, detections, initial_masks=initial_masks)
        return cls.outputs(masks=masks, combined=combined)


def _register_array(registry: TypeRegistry, type_id: str) -> None:
    if type_id not in registry:
        registry.register(
            type_id,
            encode=encode_image_array,
            decode=decode_image_array,
            prepare_buffer_encoding=prepare_image_array_encoding,
            fingerprint=image_array_fingerprint(type_id),
            meta=mask_array_meta if type_id == MASK_TYPE else image_array_meta,
            input_convert=image_input,
        )


def register_types(registry: TypeRegistry) -> None:
    _register_array(registry, IMAGE_TYPE)
    _register_array(registry, MASK_TYPE)
    if REGION_TYPE not in registry:
        registry.register(
            REGION_TYPE,
            encode=encode_region,
            decode=decode_region,
            coerce=coerce_region,
            meta=region_meta,
        )
    if DETECTION_TYPE not in registry:
        registry.register(
            DETECTION_TYPE,
            encode=encode_detection,
            decode=decode_detection,
            coerce=coerce_detection,
            meta=detection_meta,
        )


SAM31_PROVIDER_NODES: tuple[type[Node], ...] = (
    DetectObjects,
    SegmentDetections,
    SegmentByText,
    TrackObjects,
)


__all__ = [
    "SAM31_PROVIDER_NODES",
    "DetectObjects",
    "SegmentByText",
    "SegmentDetections",
    "TrackObjects",
    "register_types",
]
