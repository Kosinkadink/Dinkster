"""Torch-free native Wan image-conditioning interface contracts."""

import subprocess
import sys

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    MODEL_TYPE,
    TRACKS_TYPE,
    VAE_TYPE,
    WAN_CAMERA_EMBEDDING_TYPE,
    NumberWidget,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_workers import load_manifest


def test_wan_image_schemas_and_cold_native_discovery() -> None:
    optional_names = {
        "WanImageToVideo": ["clip_vision_output", "start_image", "ref_pad_image"],
        "WanFirstLastFrameToVideo": [
            "clip_vision_start_image",
            "clip_vision_end_image",
            "start_image",
            "end_image",
        ],
        "WanFunInpaintToVideo": ["clip_vision_output", "start_image", "end_image"],
    }
    for name, optional in optional_names.items():
        schema = next(n.schema() for n in COMFY_NODES if n.schema().node_type == name)
        assert schema.category == (
            "model/conditioning/wan/fun inpaint"
            if name == "WanFunInpaintToVideo"
            else "model/conditioning/wan"
        )
        assert [i.id for i in schema.inputs] == [
            "positive",
            "negative",
            "vae",
            "width",
            "height",
            "length",
            "batch_size",
            *optional,
        ]
        assert [i.type for i in schema.inputs[:7]] == [
            TypeExpr.concrete(t)
            for t in (
                CONDITIONING_TYPE,
                CONDITIONING_TYPE,
                VAE_TYPE,
                CORE_INT,
                CORE_INT,
                CORE_INT,
                CORE_INT,
            )
        ]
        assert [i.default for i in schema.inputs[3:7]] == [832, 480, 81, 1]
        assert [i.widget for i in schema.inputs[3:7]] == [
            NumberWidget(min=16, max=16384, step=16),
            NumberWidget(min=16, max=16384, step=16),
            NumberWidget(min=1, max=16384, step=4),
            NumberWidget(min=1, max=4096),
        ]
        assert all(i.required for i in schema.inputs[:7])
        assert all(not i.required for i in schema.inputs[7:])
        assert [i.type for i in schema.inputs[7:]] == [
            TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE if n.startswith("clip_") else IMAGE_TYPE)
            for n in optional
        ]
        assert [o.id for o in schema.outputs] == ["positive", "negative", "latent"]
        assert [o.type for o in schema.outputs] == [
            TypeExpr.concrete(t) for t in (CONDITIONING_TYPE, CONDITIONING_TYPE, LATENT_TYPE)
        ]
        assert schema_from_wire(schema_to_wire(schema)) == schema
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert tuple(n.schema().node_type for n in COMFY_NODES) == manifest.namespaces
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy import COMFY_NODES; "
            "assert {'WanImageToVideo','WanFirstLastFrameToVideo','WanFunInpaintToVideo'} "
            "<= {n.schema().node_type for n in COMFY_NODES}; "
            "assert 'torch' not in sys.modules; assert 'dinkster_inference' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_wan_control_schemas_are_native_and_keep_declared_order() -> None:
    cases = {
        "WanFunControlToVideo": (
            "fun control",
            ["clip_vision_output", "start_image", "control_video"],
            ["positive", "negative", "latent"],
        ),
        "Wan22FunControlToVideo": (
            "fun control",
            ["ref_image", "control_video"],
            ["positive", "negative", "latent"],
        ),
        "WanVaceToVideo": (
            "vace",
            ["strength", "control_video", "control_masks", "reference_image"],
            ["positive", "negative", "latent", "trim_latent"],
        ),
        "WanPhantomSubjectToVideo": (
            "phantom subject",
            ["images"],
            ["positive", "negative_text", "negative_img_text", "latent"],
        ),
    }
    for name, (category, extra, outputs) in cases.items():
        schema = next(n.schema() for n in COMFY_NODES if n.schema().node_type == name)
        assert schema.category == f"model/conditioning/wan/{category}"
        assert [i.id for i in schema.inputs] == [
            "positive",
            "negative",
            "vae",
            "width",
            "height",
            "length",
            "batch_size",
            *extra,
        ]
        assert [i.default for i in schema.inputs[3:7]] == [832, 480, 81, 1]
        for input_ in schema.inputs[7:]:
            assert input_.required == (input_.id == "strength")
            if input_.id == "control_masks":
                assert input_.type == TypeExpr.concrete(MASK_TYPE)
        assert [o.id for o in schema.outputs] == outputs
        assert schema_from_wire(schema_to_wire(schema)) == schema
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert tuple(n.schema().node_type for n in COMFY_NODES) == manifest.namespaces


def test_scail_schema_preserves_interleaved_optional_inputs() -> None:
    schema = next(n.schema() for n in COMFY_NODES if n.schema().node_type == "WanSCAILToVideo")
    assert schema.category == "model/conditioning/wan/scail"
    assert [i.id for i in schema.inputs] == [
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "pose_video",
        "pose_video_mask",
        "replacement_mode",
        "pose_strength",
        "pose_start",
        "pose_end",
        "reference_image",
        "reference_image_mask",
        "clip_vision_output",
        "video_frame_offset",
        "previous_frame_count",
        "previous_frames",
    ]
    assert [i.id for i in schema.inputs if not i.required] == [
        "pose_video",
        "pose_video_mask",
        "replacement_mode",
        "reference_image",
        "reference_image_mask",
        "clip_vision_output",
        "previous_frames",
    ]
    assert [i.default for i in schema.inputs[3:7]] == [512, 896, 81, 1]
    assert [o.id for o in schema.outputs] == [
        "positive",
        "negative",
        "latent",
        "video_frame_offset",
    ]
    assert schema_from_wire(schema_to_wire(schema)) == schema


def test_track_schema_preserves_required_image_and_string_tracks() -> None:
    schema = next(n.schema() for n in COMFY_NODES if n.schema().node_type == "WanTrackToVideo")
    assert schema.category == "model/conditioning/wan/move"
    assert [i.id for i in schema.inputs] == [
        "positive",
        "negative",
        "vae",
        "tracks",
        "width",
        "height",
        "length",
        "batch_size",
        "temperature",
        "topk",
        "start_image",
        "clip_vision_output",
    ]
    assert [i.default for i in schema.inputs[3:10]] == ["[]", 832, 480, 81, 1, 220.0, 2]
    assert [i.id for i in schema.inputs if not i.required] == ["clip_vision_output"]
    assert schema.inputs[8].advanced and schema.inputs[9].advanced
    assert [o.id for o in schema.outputs] == ["positive", "negative", "latent"]
    assert schema_from_wire(schema_to_wire(schema)) == schema


def test_animation_schemas_preserve_interleaved_inputs_and_cache_options() -> None:
    suffixes = {
        "WanAnimateToVideo": [
            "clip_vision_output",
            "reference_image",
            "face_video",
            "pose_video",
            "continue_motion_max_frames",
            "background_video",
            "character_mask",
            "continue_motion",
            "video_frame_offset",
        ],
        "WanAnimate2ToVideo": [
            "reference_image",
            "pose_video",
            "clip_vision_output",
            "positive_pose",
            "clip_vision_output_pose",
            "continue_motion",
            "video_frame_offset",
            "pose_strength",
            "pose_start_percent",
            "pose_end_percent",
            "reference_image_strength",
        ],
    }
    for name, suffix in suffixes.items():
        schema = next(n.schema() for n in COMFY_NODES if n.schema().node_type == name)
        assert [i.id for i in schema.inputs] == [
            "positive",
            "negative",
            "vae",
            "width",
            "height",
            "length",
            "batch_size",
            *suffix,
        ]
        assert schema.inputs[6].widget == NumberWidget(min=1, max=4096)
        assert [o.id for o in schema.outputs] == [
            "positive",
            "negative",
            "latent",
            "trim_latent",
            "trim_image",
            "video_frame_offset",
        ]
        assert schema_from_wire(schema_to_wire(schema)) == schema
    cache = next(n.schema() for n in COMFY_NODES if n.schema().node_type == "WanAnimate2Cache")
    assert [i.id for i in cache.inputs] == ["model", "device", "dtype"]
    assert [i.default for i in cache.inputs[1:]] == ["cpu", "default"]
    assert schema_from_wire(schema_to_wire(cache)) == cache


def test_advanced_wan_native_outputs_use_canonical_portable_types() -> None:
    conditioning = (
        ("positive", CONDITIONING_TYPE),
        ("negative", CONDITIONING_TYPE),
        ("latent", LATENT_TYPE),
    )
    cases = {
        "WanCameraEmbedding": (
            ("camera_embedding", WAN_CAMERA_EMBEDDING_TYPE),
            ("width", CORE_INT),
            ("height", CORE_INT),
            ("length", CORE_INT),
        ),
        "WanCameraImageToVideo": conditioning,
        "WanMoveTracksFromCoords": (("TRACKS", TRACKS_TYPE), ("track_length", CORE_INT)),
        "GenerateTracks": (("TRACKS", TRACKS_TYPE), ("track_length", CORE_INT)),
        "WanMoveConcatTrack": (("TRACKS", TRACKS_TYPE),),
        "WanMoveTrackToVideo": conditioning,
        "WanMoveVisualizeTracks": (("IMAGE", IMAGE_TYPE),),
        "WanSoundImageToVideo": conditioning,
        "WanSoundImageToVideoExtend": conditioning,
        "WanHuMoImageToVideo": conditioning,
        "WanInfiniteTalkToVideo": (("model", MODEL_TYPE), *conditioning, ("trim_image", CORE_INT)),
        "SCAIL2ColoredMask": (
            ("pose_video_mask", IMAGE_TYPE),
            ("reference_image_mask", IMAGE_TYPE),
        ),
    }
    schemas = {node.schema().node_type: node.schema() for node in COMFY_NODES}
    for name, outputs in cases.items():
        schema = schemas[name]
        assert [(output.id, output.type) for output in schema.outputs] == [
            (name, TypeExpr.concrete(type_id)) for name, type_id in outputs
        ]
        assert schema_from_wire(schema_to_wire(schema)) == schema
