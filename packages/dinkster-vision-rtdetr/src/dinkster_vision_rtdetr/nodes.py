"""Stable node shape and boundary codecs for the RT-DETR provider."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
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
    prepare_image_array_encoding,
    region_meta,
)

IMAGE_TYPE = "dinkster.image"
IMAGE = TypeExpr.concrete(IMAGE_TYPE)
DETECTIONS = TypeExpr.list_of(TypeExpr.concrete(DETECTION_TYPE))
COMBO = TypeExpr.concrete(CORE_COMBO)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
INT = TypeExpr.concrete(CORE_INT)
STRING = TypeExpr.concrete(CORE_STRING)
PROVIDER_CHOICE = "dinkster.detection.detect.providers"
_PROMPT_MODES = ("comma-separated", "literal")
_RESULT_LIMIT_MODES = ("count", "slice-stop")


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
                InputSpec(
                    "prompt_mode",
                    COMBO,
                    required=False,
                    default="comma-separated",
                    widget=ComboWidget(options=_PROMPT_MODES),
                    advanced=True,
                ),
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
                    widget=ComboWidget(remote_route=f"/api/choices/{PROVIDER_CHOICE}"),
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


def register_types(registry: TypeRegistry) -> None:
    if IMAGE_TYPE not in registry:
        registry.register(
            IMAGE_TYPE,
            encode=encode_image_array,
            decode=decode_image_array,
            prepare_buffer_encoding=prepare_image_array_encoding,
            fingerprint=image_array_fingerprint(IMAGE_TYPE),
            meta=image_array_meta,
        )
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


RTDETR_PROVIDER_NODES: tuple[type[Node], ...] = (DetectObjects,)


__all__ = ["RTDETR_PROVIDER_NODES", "DetectObjects", "register_types"]
