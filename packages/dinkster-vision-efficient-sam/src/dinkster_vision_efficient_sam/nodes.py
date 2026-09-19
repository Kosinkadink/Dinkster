"""Stable node shape and boundary codecs for the EfficientSAM provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dinkster_api.v1 import (
    CORE_COMBO,
    DETECTION_TYPE,
    REGION_TYPE,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    coerce_detection,
    coerce_region,
    decode_detection,
    decode_image_array,
    decode_image_array_buffer,
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
MASKS = TypeExpr.list_of(TypeExpr.concrete(MASK_TYPE))
DETECTIONS = TypeExpr.list_of(TypeExpr.concrete(DETECTION_TYPE))
COMBO = TypeExpr.concrete(CORE_COMBO)
PROVIDER_CHOICE = "dinkster.detection.segment.providers"


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
                    widget=ComboWidget(remote_route=f"/api/choices/{PROVIDER_CHOICE}"),
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


def _register_array(registry: TypeRegistry, type_id: str) -> None:
    if type_id not in registry:
        registry.register(
            type_id,
            encode=encode_image_array,
            decode=decode_image_array,
            decode_buffer=decode_image_array_buffer,
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


EFFICIENT_SAM_PROVIDER_NODES: tuple[type[Node], ...] = (SegmentDetections,)


__all__ = ["EFFICIENT_SAM_PROVIDER_NODES", "SegmentDetections", "register_types"]
