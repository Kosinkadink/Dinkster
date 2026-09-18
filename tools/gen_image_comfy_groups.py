"""Generate maintained ComfyUI vision group translations.

The source schemas are inert interface snapshots transcribed from the pinned
ComfyUI revision. The generator never imports ComfyUI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dinkster_schema import (
    SCHEMA_WIRE_VERSION,
    BooleanWidget,
    ComboWidget,
    ComfyAliasConfidence,
    ComfyAliasSource,
    ComfyGroupEdge,
    ComfyGroupNode,
    ComfyGroupPattern,
    ComfyGroupRecord,
    ComfyGroupRegistry,
    ComfyGroupSource,
    ComfyGroupSourceSchema,
    InputSpec,
    MappingSource,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    ReplacementCase,
    ReplacementPredicate,
    ReplacementRule,
    StringWidget,
    TypeExpr,
    comfy_group_registry_to_wire,
)

COMFY_REVISION = "c67885b1"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages" / "dinkster-nodes-image" / "comfy-groups.json"

IMAGE = TypeExpr.concrete("comfy.IMAGE")
MASK = TypeExpr.concrete("comfy.MASK")
MODEL = TypeExpr.concrete("comfy.MODEL")
CLIP = TypeExpr.concrete("comfy.CLIP")
CONDITIONING = TypeExpr.concrete("comfy.CONDITIONING")
VAE = TypeExpr.concrete("comfy.VAE")
SAM3_TRACK_DATA = TypeExpr.concrete("comfy.SAM3_TRACK_DATA")
BACKGROUND_REMOVAL = TypeExpr.concrete("comfy.BACKGROUND_REMOVAL")
BOUNDING_BOX = TypeExpr.concrete("comfy.BOUNDING_BOX")
BOOLEAN = TypeExpr.concrete("core.boolean")
COMBO = TypeExpr.concrete("core.combo")
FLOAT = TypeExpr.concrete("core.float")
INT = TypeExpr.concrete("core.int")
STRING = TypeExpr.concrete("core.string")

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def _combo(id: str, values: tuple[str, ...], default: str) -> InputSpec:
    return InputSpec(id, COMBO, default=default, widget=ComboWidget(values))


def _source(node_class: str, node_type: str) -> ComfyAliasSource:
    return ComfyAliasSource(
        pack="comfy-core",
        node_class=node_class,
        node_type=node_type,
        revision=COMFY_REVISION,
    )


def _snapshot(schema: NodeSchema) -> ComfyGroupSourceSchema:
    return ComfyGroupSourceSchema(schema, SCHEMA_WIRE_VERSION)


UNET_LOADER = NodeSchema(
    node_type="comfy.UNETLoader",
    inputs=(
        _combo(
            "unet_name",
            (
                "rt_detr_v4-x-hgnet_fp16.safetensors",
                "rt_detr_v4-x-hgnet_fp32.safetensors",
            ),
            "rt_detr_v4-x-hgnet_fp32.safetensors",
        ),
        _combo(
            "weight_dtype",
            ("default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"),
            "default",
        ),
    ),
    outputs=(OutputSpec("model", MODEL),),
    aliases=("UNETLoader",),
)

RTDETR_DETECT = NodeSchema(
    node_type="comfy.RTDETR_detect",
    inputs=(
        InputSpec("model", MODEL),
        InputSpec("image", IMAGE),
        InputSpec(
            "threshold",
            FLOAT,
            default=0.5,
            widget=NumberWidget(step=0.01),
        ),
        _combo("class_name", ("all", *COCO_CLASSES), "all"),
        InputSpec("max_detections", INT, default=100, widget=NumberWidget(step=1)),
    ),
    outputs=(OutputSpec("bboxes", BOUNDING_BOX),),
    aliases=("RTDETR_detect",),
)

CHECKPOINT_LOADER = NodeSchema(
    node_type="comfy.CheckpointLoaderSimple",
    inputs=(
        _combo(
            "ckpt_name",
            ("sam3.1_multiplex_fp16.safetensors",),
            "sam3.1_multiplex_fp16.safetensors",
        ),
    ),
    outputs=(
        OutputSpec("model", MODEL),
        OutputSpec("clip", CLIP),
        OutputSpec("vae", VAE),
    ),
    aliases=("CheckpointLoaderSimple",),
)

CLIP_TEXT_ENCODE = NodeSchema(
    node_type="comfy.CLIPTextEncode",
    inputs=(
        InputSpec("clip", CLIP),
        InputSpec("text", STRING, default="", widget=StringWidget(multiline=True)),
    ),
    outputs=(OutputSpec("conditioning", CONDITIONING),),
    aliases=("CLIPTextEncode",),
)

SAM3_DETECT = NodeSchema(
    node_type="comfy.SAM3_Detect",
    inputs=(
        InputSpec("model", MODEL),
        InputSpec("image", IMAGE),
        InputSpec("conditioning", CONDITIONING, required=False),
        InputSpec(
            "bboxes",
            BOUNDING_BOX,
            required=False,
            default={"x": 0, "y": 0, "width": 512, "height": 512},
            force_input=True,
        ),
        InputSpec(
            "positive_coords",
            STRING,
            required=False,
            widget=StringWidget(),
            force_input=True,
        ),
        InputSpec(
            "negative_coords",
            STRING,
            required=False,
            widget=StringWidget(),
            force_input=True,
        ),
        InputSpec(
            "threshold",
            FLOAT,
            default=0.5,
            widget=NumberWidget(min=0.0, max=1.0, step=0.01),
        ),
        InputSpec(
            "refine_iterations",
            INT,
            default=2,
            widget=NumberWidget(min=0, max=5),
        ),
        InputSpec(
            "individual_masks",
            BOOLEAN,
            default=False,
            widget=BooleanWidget(label_on="true", label_off="false"),
        ),
    ),
    outputs=(OutputSpec("masks", MASK), OutputSpec("bboxes", BOUNDING_BOX)),
    aliases=("SAM3_Detect",),
)

SAM3_VIDEO_TRACK = NodeSchema(
    node_type="comfy.SAM3_VideoTrack",
    inputs=(
        InputSpec("images", IMAGE),
        InputSpec("model", MODEL),
        InputSpec("initial_mask", MASK, required=False),
        InputSpec("conditioning", CONDITIONING, required=False),
        InputSpec(
            "detection_threshold",
            FLOAT,
            default=0.5,
            widget=NumberWidget(min=0.0, max=1.0, step=0.01),
        ),
        InputSpec(
            "max_objects",
            INT,
            default=4,
            widget=NumberWidget(min=0, max=64),
        ),
        InputSpec("detect_interval", INT, default=1, widget=NumberWidget(min=1)),
    ),
    outputs=(OutputSpec("track_data", SAM3_TRACK_DATA),),
    aliases=("SAM3_VideoTrack",),
)

SAM3_TRACK_TO_MASK = NodeSchema(
    node_type="comfy.SAM3_TrackToMask",
    inputs=(
        InputSpec("track_data", SAM3_TRACK_DATA),
        InputSpec("object_indices", STRING, default="", widget=StringWidget()),
    ),
    outputs=(OutputSpec("masks", MASK),),
    aliases=("SAM3_TrackToMask",),
)

LOAD_BACKGROUND = NodeSchema(
    node_type="comfy.LoadBackgroundRemovalModel",
    inputs=(_combo("bg_removal_name", ("birefnet.safetensors",), "birefnet.safetensors"),),
    outputs=(OutputSpec("bg_model", BACKGROUND_REMOVAL),),
    aliases=("LoadBackgroundRemovalModel",),
)

REMOVE_BACKGROUND = NodeSchema(
    node_type="comfy.RemoveBackground",
    inputs=(
        InputSpec("bg_removal_model", BACKGROUND_REMOVAL),
        InputSpec("image", IMAGE),
    ),
    outputs=(OutputSpec("mask", MASK),),
    aliases=("RemoveBackground",),
)


def _rtdetr_record(model_name: str, suffix: str) -> tuple[NodeSchema, ComfyGroupRecord]:
    group_type = f"comfy-group.comfy-core.rtdetr-detect-{suffix}"
    group = NodeSchema(
        node_type=group_type,
        inputs=(
            InputSpec("image", IMAGE),
            InputSpec(
                "threshold",
                FLOAT,
                default=0.5,
                widget=NumberWidget(step=0.01),
            ),
            _combo("class_name", ("all", *COCO_CLASSES), "all"),
            InputSpec("max_detections", INT, default=100, widget=NumberWidget(step=1)),
        ),
        outputs=(OutputSpec("bboxes", BOUNDING_BOX),),
    )
    pattern = ComfyGroupPattern(
        group_type=group_type,
        anchor="detect",
        nodes=(
            ("loader", ComfyGroupNode(_source("UNETLoader", UNET_LOADER.node_type), "active")),
            (
                "detect",
                ComfyGroupNode(_source("RTDETR_detect", RTDETR_DETECT.node_type), "active"),
            ),
        ),
        edges=(ComfyGroupEdge("loader:model", "detect:model"),),
        inputs=(("image", "detect:image"),),
        parameters=(
            ("threshold", "detect:threshold"),
            ("class_name", "detect:class_name"),
            ("max_detections", "detect:max_detections"),
        ),
        constants=(("loader:unet_name", model_name), ("loader:weight_dtype", "default")),
        outputs=(("bboxes", "detect:bboxes"),),
    )

    def case(prompt: MappingSource, when: ReplacementPredicate | None = None) -> ReplacementCase:
        return ReplacementCase.build(
            "dinkster.detection.detect",
            when=when,
            inputs={
                "image": MappingSource.copy("image"),
                "prompt": prompt,
                "min_score": MappingSource.copy("threshold"),
                "max_results": MappingSource.copy("max_detections"),
                "result_limit_mode": MappingSource.constant("slice-stop"),
                "provider": MappingSource.constant("dinkster-vision-rtdetr"),
            },
            outputs={"detections": "bboxes"},
        )

    replacement = ReplacementRule(
        from_type=group_type,
        note=(
            "The native RT-DETR provider preserves the source model, COCO label filtering, "
            "score threshold, descending score order, and per-frame result limits."
        ),
        cases=(
            case(
                MappingSource.constant(""),
                ReplacementPredicate.value_equals("class_name", "all"),
            ),
            case(MappingSource.copy("class_name")),
        ),
    )
    return group, ComfyGroupRecord(
        id=f"comfy_group:comfy-core/rtdetr-detect-{suffix}",
        mapping_kind="op",
        carrier="dinkster.detection.detect",
        source=ComfyGroupSource("comfy-core", f"rtdetr-detect-{suffix}", COMFY_REVISION),
        pattern=pattern,
        replacement=replacement,
        confidence=ComfyAliasConfidence(
            "grouped",
            ("tests/test_image_comfy_groups.py::test_rtdetr_groups_preserve_detection_parameters",),
        ),
    )


def _background_record() -> tuple[NodeSchema, ComfyGroupRecord]:
    group_type = "comfy-group.comfy-core.remove-background-birefnet"
    group = NodeSchema(
        node_type=group_type,
        inputs=(InputSpec("image", IMAGE),),
        outputs=(OutputSpec("mask", MASK),),
    )
    return group, ComfyGroupRecord(
        id="comfy_group:comfy-core/remove-background-birefnet",
        mapping_kind="op",
        carrier="dinkster.image.matte",
        source=ComfyGroupSource("comfy-core", "remove-background-birefnet", COMFY_REVISION),
        pattern=ComfyGroupPattern(
            group_type=group_type,
            anchor="remove",
            nodes=(
                (
                    "loader",
                    ComfyGroupNode(
                        _source("LoadBackgroundRemovalModel", LOAD_BACKGROUND.node_type),
                        "active",
                    ),
                ),
                (
                    "remove",
                    ComfyGroupNode(
                        _source("RemoveBackground", REMOVE_BACKGROUND.node_type), "active"
                    ),
                ),
            ),
            edges=(ComfyGroupEdge("loader:bg_model", "remove:bg_removal_model"),),
            inputs=(("image", "remove:image"),),
            parameters=(),
            constants=(("loader:bg_removal_name", "birefnet.safetensors"),),
            outputs=(("mask", "remove:mask"),),
        ),
        replacement=ReplacementRule(
            from_type=group_type,
            cases=(
                ReplacementCase.build(
                    "dinkster.image.matte",
                    inputs={
                        "image": MappingSource.copy("image"),
                        "provider": MappingSource.constant("dinkster-vision-birefnet"),
                    },
                    outputs={"mask": "mask"},
                ),
            ),
        ),
        confidence=ComfyAliasConfidence(
            "grouped",
            ("tests/test_image_comfy_groups.py::test_background_group_selects_birefnet",),
        ),
    )


def _sam3_text_detection_record() -> tuple[NodeSchema, ComfyGroupRecord]:
    group_type = "comfy-group.comfy-core.sam3-text-detection"
    group = NodeSchema(
        node_type=group_type,
        inputs=(
            InputSpec("image", IMAGE),
            InputSpec("text", STRING, default="", widget=StringWidget(multiline=True)),
            InputSpec(
                "threshold",
                FLOAT,
                default=0.5,
                widget=NumberWidget(min=0.0, max=1.0, step=0.01),
            ),
        ),
        outputs=(OutputSpec("bboxes", BOUNDING_BOX),),
    )
    return group, ComfyGroupRecord(
        id="comfy_group:comfy-core/sam3-text-detection",
        mapping_kind="op",
        carrier="dinkster.detection.detect",
        source=ComfyGroupSource("comfy-core", "sam3-text-detection", COMFY_REVISION),
        pattern=ComfyGroupPattern(
            group_type=group_type,
            anchor="detect",
            nodes=(
                (
                    "loader",
                    ComfyGroupNode(
                        _source("CheckpointLoaderSimple", CHECKPOINT_LOADER.node_type), "active"
                    ),
                ),
                (
                    "encode",
                    ComfyGroupNode(_source("CLIPTextEncode", CLIP_TEXT_ENCODE.node_type), "active"),
                ),
                (
                    "detect",
                    ComfyGroupNode(_source("SAM3_Detect", SAM3_DETECT.node_type), "active"),
                ),
            ),
            edges=(
                ComfyGroupEdge("loader:model", "detect:model"),
                ComfyGroupEdge("loader:clip", "encode:clip"),
                ComfyGroupEdge("encode:conditioning", "detect:conditioning"),
            ),
            inputs=(("image", "detect:image"),),
            parameters=(("text", "encode:text"), ("threshold", "detect:threshold")),
            constants=(
                ("loader:ckpt_name", "sam3.1_multiplex_fp16.safetensors"),
                ("detect:refine_iterations", 2),
                ("detect:individual_masks", False),
            ),
            outputs=(("bboxes", "detect:bboxes"),),
            disconnected=(
                "detect:bboxes",
                "detect:positive_coords",
                "detect:negative_coords",
            ),
        ),
        replacement=ReplacementRule(
            from_type=group_type,
            note=(
                "Text-only SAM 3 detection preserves the pinned model, threshold, "
                "frame-major box order, and one result per text prompt."
            ),
            cases=(
                ReplacementCase.build(
                    "dinkster.detection.detect",
                    inputs={
                        "image": MappingSource.copy("image"),
                        "prompt": MappingSource.copy("text"),
                        "prompt_mode": MappingSource.constant("literal"),
                        "min_score": MappingSource.copy("threshold"),
                        "max_results": MappingSource.constant(1),
                        "provider": MappingSource.constant("dinkster-vision-sam31"),
                    },
                    outputs={"detections": "bboxes"},
                ),
            ),
        ),
        confidence=ComfyAliasConfidence(
            "grouped",
            (
                "tests/test_image_comfy_groups.py::test_sam3_text_detection_preserves_supported_inputs",
            ),
        ),
    )


def _sam3_video_tracking_record() -> tuple[NodeSchema, ComfyGroupRecord]:
    group_type = "comfy-group.comfy-core.sam3-video-track-initial-mask"
    group = NodeSchema(
        node_type=group_type,
        inputs=(InputSpec("images", IMAGE), InputSpec("initial_mask", MASK, required=False)),
        outputs=(OutputSpec("masks", MASK),),
    )
    return group, ComfyGroupRecord(
        id="comfy_group:comfy-core/sam3-video-track-initial-mask",
        mapping_kind="op",
        carrier="dinkster.detection.track",
        source=ComfyGroupSource("comfy-core", "sam3-video-track-initial-mask", COMFY_REVISION),
        pattern=ComfyGroupPattern(
            group_type=group_type,
            anchor="track",
            nodes=(
                (
                    "loader",
                    ComfyGroupNode(
                        _source("CheckpointLoaderSimple", CHECKPOINT_LOADER.node_type), "active"
                    ),
                ),
                (
                    "track",
                    ComfyGroupNode(
                        _source("SAM3_VideoTrack", SAM3_VIDEO_TRACK.node_type), "active"
                    ),
                ),
                (
                    "to_mask",
                    ComfyGroupNode(
                        _source("SAM3_TrackToMask", SAM3_TRACK_TO_MASK.node_type), "active"
                    ),
                ),
            ),
            edges=(
                ComfyGroupEdge("loader:model", "track:model"),
                ComfyGroupEdge("track:track_data", "to_mask:track_data"),
            ),
            inputs=(
                ("images", "track:images"),
                ("initial_mask", "track:initial_mask"),
            ),
            parameters=(),
            constants=(
                ("loader:ckpt_name", "sam3.1_multiplex_fp16.safetensors"),
                ("track:detection_threshold", 0.5),
                ("track:max_objects", 4),
                ("track:detect_interval", 1),
                ("to_mask:object_indices", ""),
            ),
            outputs=(("masks", "to_mask:masks"),),
            disconnected=("track:conditioning",),
        ),
        replacement=ReplacementRule(
            from_type=group_type,
            note=(
                "Initial-mask SAM 3 tracking preserves frame order, object masks, "
                "source dimensions, and the all-object mask union."
            ),
            cases=(
                ReplacementCase.build(
                    "dinkster.detection.track",
                    inputs={
                        "image": MappingSource.copy("images"),
                        "initial_masks": MappingSource.copy("initial_mask"),
                        "provider": MappingSource.constant("dinkster-vision-sam31"),
                    },
                    outputs={"combined": "masks"},
                ),
            ),
        ),
        confidence=ComfyAliasConfidence(
            "grouped",
            ("tests/test_image_comfy_groups.py::test_sam3_video_tracking_preserves_initial_masks",),
        ),
    )


def build_registry() -> ComfyGroupRegistry:
    fp16_group, fp16_record = _rtdetr_record("rt_detr_v4-x-hgnet_fp16.safetensors", "fp16")
    sam3_group, sam3_record = _sam3_text_detection_record()
    sam3_video_group, sam3_video_record = _sam3_video_tracking_record()
    background_group, background_record = _background_record()
    return ComfyGroupRegistry(
        source_schemas=tuple(
            _snapshot(schema)
            for schema in (
                UNET_LOADER,
                RTDETR_DETECT,
                CHECKPOINT_LOADER,
                CLIP_TEXT_ENCODE,
                SAM3_DETECT,
                SAM3_VIDEO_TRACK,
                SAM3_TRACK_TO_MASK,
                LOAD_BACKGROUND,
                REMOVE_BACKGROUND,
            )
        ),
        group_schemas=tuple(
            _snapshot(schema)
            for schema in (fp16_group, sam3_group, sam3_video_group, background_group)
        ),
        records=(fp16_record, sam3_record, sam3_video_record, background_record),
    )


def main() -> None:
    content = (json.dumps(comfy_group_registry_to_wire(build_registry()), indent=2) + "\n").encode()
    OUT.write_bytes(content)
    print(f"{OUT}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
