from __future__ import annotations

import asyncio
import inspect
import sys
import tomllib
from pathlib import Path

from dinkster_model_wan import WAN_MODEL_NODE_IDS, WAN_MODEL_NODES
from dinkster_schema import AssetWidget, BooleanWidget, ComboWidget, NumberWidget, build_schemas
from dinkster_workers import load_manifest

from dinkster.compose import PackSpec, ServingComposer, default_pack_spec

PACKAGE = Path(__file__).parent.parent / "packages" / "dinkster-model-wan"
MANIFEST = PACKAGE / "dinkster-pack.toml"
GENERATION_MANIFEST = (
    Path(__file__).parent.parent / "packages" / "dinkster-nodes-generation" / "dinkster-pack.toml"
)


def test_wan_model_manifest_owns_the_native_animate_schemas() -> None:
    manifest = load_manifest(MANIFEST)

    assert manifest.name == "dinkster-model-wan"
    assert manifest.schema_only == ()
    assert WAN_MODEL_NODE_IDS == (
        "dinkster.empty_ar_video_latent",
        "dinkster.sampler_ar_video",
        "dinkster.ar_video_i2v",
        "dinkster.wan22_animate_to_video",
        "dinkster.wan21_animate2_to_video",
        "dinkster.wan21_scail_to_video",
        "dinkster.load_wan21_uni3c",
        "dinkster.apply_wan21_uni3c",
        "dinkster.load_wan_s2v_audio_encoder",
        "dinkster.encode_wan_s2v_audio",
        "dinkster.wan22_s2v",
        "dinkster.wan22_s2v_extend",
        "dinkster.wan21_humo",
        "dinkster.wan_infinite_talk_to_video",
        "dinkster.encode_wandancer_audio",
        "dinkster.wan22_dancer_video",
        "dinkster.wandancer_pad_keyframes",
        "dinkster.wandancer_pad_keyframe_list",
    )
    assert manifest.executes == ()
    assert manifest.capabilities == ()
    assert [(item.pack, item.version) for item in manifest.dependencies] == [
        ("dinkster-nodes-media-io", "<1,>=0.0.1")
    ]
    assert [(item.id, item.version) for item in manifest.requirements.capabilities] == [
        ("dinkster.generation.schemas", "<2,>=1.0.0")
    ]
    assert {(item.registry, item.id) for item in manifest.requirements.registry} == {
        ("dinkster.model-families", "dinkster.wan21"),
        ("dinkster.model-families", "dinkster.wan22"),
        ("dinkster.samplers", "dinkster.ar_video"),
        ("dinkster.samplers", "dinkster.uni_pc"),
        ("dinkster.schedulers", "dinkster.simple"),
    }


def test_wan_model_package_does_not_import_the_generation_owner() -> None:
    metadata = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["dependencies"] == [
        "dinkster-api",
        "dinkster-inference",
        "dinkster-inference-torch",
        "numpy>=1.26",
        "scipy>=1.11",
    ]
    assert not any(
        "dinkster_nodes_generation" in path.read_text(encoding="utf-8")
        for path in (PACKAGE / "src").rglob("*.py")
    )


def test_wan_animate_schema_uses_only_published_native_boundaries() -> None:
    schemas = build_schemas(WAN_MODEL_NODES)
    assert tuple(schemas) == WAN_MODEL_NODE_IDS
    empty = schemas["dinkster.empty_ar_video_latent"]
    assert {item.id: item.type.types for item in empty.inputs} == {
        "width": ("core.int",),
        "height": ("core.int",),
        "length": ("core.int",),
        "batch_size": ("core.int",),
    }
    assert {item.id: item.type.types for item in empty.outputs} == {"latent": ("dinkster.latent",)}
    sampler = schemas["dinkster.sampler_ar_video"]
    assert {item.id: item.type.types for item in sampler.inputs} == {
        "num_frame_per_block": ("core.int",)
    }
    assert {item.id: item.type.types for item in sampler.outputs} == {
        "sampler": ("dinkster.sampler",)
    }
    i2v = schemas["dinkster.ar_video_i2v"]
    assert {item.id: item.type.types for item in i2v.inputs} == {
        "model": ("dinkster.model",),
        "vae": ("dinkster.vae",),
        "start_image": ("dinkster.image",),
        "width": ("core.int",),
        "height": ("core.int",),
        "length": ("core.int",),
        "batch_size": ("core.int",),
    }
    assert {item.id: item.type.types for item in i2v.outputs} == {
        "model": ("dinkster.model",),
        "latent": ("dinkster.latent",),
    }
    schema = schemas["dinkster.wan22_animate_to_video"]

    assert schema.display_name == "WanAnimateToVideo"
    assert schema.aliases == ()
    assert {item.id: item.type.types for item in schema.inputs} == {
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "model": ("dinkster.model",),
        "vae": ("dinkster.vae",),
        "width": ("core.int",),
        "height": ("core.int",),
        "length": ("core.int",),
        "batch_size": ("core.int",),
        "reference_image": ("dinkster.image",),
        "face_video": ("dinkster.image",),
        "pose_video": ("dinkster.image",),
        "continue_motion_max_frames": ("core.int",),
        "background_video": ("dinkster.image",),
        "character_mask": ("dinkster.mask",),
        "continue_motion": ("dinkster.image",),
        "video_frame_offset": ("core.int",),
    }
    assert {item.id: item.type.types for item in schema.outputs} == {
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "latent": ("dinkster.latent",),
        "trim_latent": ("core.int",),
        "trim_image": ("core.int",),
        "video_frame_offset": ("core.int",),
    }
    assert not any(
        type_id.startswith("comfy.")
        for socket in (*schema.inputs, *schema.outputs)
        for type_id in socket.type.types
    )

    inputs = {item.id: item for item in schema.inputs}
    assert inputs["width"].widget == NumberWidget(min=16, max=16384, step=16)
    assert inputs["height"].widget == NumberWidget(min=16, max=16384, step=16)
    assert inputs["length"].widget == NumberWidget(min=1, max=16384, step=4)
    assert inputs["batch_size"].widget == NumberWidget(min=1, max=4096, step=1)
    assert inputs["continue_motion_max_frames"].widget == NumberWidget(min=1, max=16384, step=4)
    assert inputs["video_frame_offset"].widget == NumberWidget(min=0, max=16384, step=1)

    animate2 = schemas["dinkster.wan21_animate2_to_video"]
    assert animate2.display_name == "WanAnimate2ToVideo"
    assert animate2.aliases == ()
    assert {item.id: item.type.types for item in animate2.inputs} == {
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "model": ("dinkster.model",),
        "vae": ("dinkster.vae",),
        "width": ("core.int",),
        "height": ("core.int",),
        "length": ("core.int",),
        "batch_size": ("core.int",),
        "video_frame_offset": ("core.int",),
        "pose_strength": ("core.float",),
        "pose_start_percent": ("core.float",),
        "pose_end_percent": ("core.float",),
        "reference_image_strength": ("core.float",),
        "reference_image": ("dinkster.image",),
        "pose_video": ("dinkster.image",),
        "positive_pose": ("dinkster.conditioning",),
        "continue_motion": ("dinkster.image",),
    }
    assert {item.id: item.type.types for item in animate2.outputs} == {
        item.id: item.type.types for item in schema.outputs
    }
    animate2_inputs = {item.id: item for item in animate2.inputs}
    assert animate2_inputs["pose_strength"].widget == NumberWidget(min=0.0, max=10.0, step=0.01)
    assert animate2_inputs["pose_start_percent"].widget == NumberWidget(min=0.0, max=1.0, step=0.01)
    assert animate2_inputs["pose_end_percent"].widget == NumberWidget(min=0.0, max=1.0, step=0.01)
    assert animate2_inputs["reference_image_strength"].widget == NumberWidget(
        min=0.0, max=10.0, step=0.01
    )

    scail = schemas["dinkster.wan21_scail_to_video"]
    assert scail.display_name == "WanSCAILToVideo"
    assert scail.aliases == ()
    assert {item.id: item.type.types for item in scail.inputs} == {
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "model": ("dinkster.model",),
        "vae": ("dinkster.vae",),
        "width": ("core.int",),
        "height": ("core.int",),
        "length": ("core.int",),
        "batch_size": ("core.int",),
        "pose_strength": ("core.float",),
        "pose_start_percent": ("core.float",),
        "pose_end_percent": ("core.float",),
        "video_frame_offset": ("core.int",),
        "previous_frame_count": ("core.int",),
        "replacement_mode": ("core.boolean",),
        "reference_image": ("dinkster.image",),
        "pose_video": ("dinkster.image",),
        "pose_video_mask": ("dinkster.image",),
        "reference_image_mask": ("dinkster.image",),
        "previous_frames": ("dinkster.image",),
    }
    assert {item.id: item.type.types for item in scail.outputs} == {
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "latent": ("dinkster.latent",),
        "video_frame_offset": ("core.int",),
    }
    scail_inputs = {item.id: item for item in scail.inputs}
    assert scail_inputs["width"].widget == NumberWidget(min=32, max=16384, step=32)
    assert scail_inputs["height"].widget == NumberWidget(min=32, max=16384, step=32)
    assert scail_inputs["replacement_mode"].widget == BooleanWidget(
        label_on="replacement", label_off="animation"
    )
    assert scail_inputs["reference_image"].required is False

    loader = schemas["dinkster.load_wan21_uni3c"]
    assert {item.id: item.type.types for item in loader.inputs} == {
        "model_patch": ("dinkster.asset",)
    }
    assert loader.inputs[0].widget == AssetWidget(
        accept=("application/octet-stream",), kind="model/patch"
    )
    assert {item.id: item.type.types for item in loader.outputs} == {
        "patch": ("dinkster.wan21_uni3c",)
    }

    apply = schemas["dinkster.apply_wan21_uni3c"]
    assert {item.id: item.type.types for item in apply.inputs} == {
        "model": ("dinkster.model",),
        "patch": ("dinkster.wan21_uni3c",),
        "vae": ("dinkster.vae",),
        "render_video": ("dinkster.image",),
        "strength": ("core.float",),
        "start_percent": ("core.float",),
        "end_percent": ("core.float",),
    }
    assert {item.id: item.type.types for item in apply.outputs} == {"model": ("dinkster.model",)}
    apply_inputs = {item.id: item for item in apply.inputs}
    assert apply_inputs["strength"].widget == NumberWidget(min=-10.0, max=10.0, step=0.01)
    assert apply_inputs["start_percent"].widget == NumberWidget(min=0.0, max=1.0, step=0.01)
    assert apply_inputs["end_percent"].widget == NumberWidget(min=0.0, max=1.0, step=0.01)

    audio_loader = schemas["dinkster.load_wan_s2v_audio_encoder"]
    assert audio_loader.aliases == ("AudioEncoderLoader",)
    assert audio_loader.display_name == "Load Wan Audio Encoder"
    assert {"humo", "whisper"}.issubset(audio_loader.search_terms)
    assert audio_loader.inputs[0].widget == AssetWidget(
        accept=("application/octet-stream",), kind="model/audio-encoder"
    )
    assert audio_loader.outputs[0].type.types == ("dinkster.audio_encoder",)
    audio_encode = schemas["dinkster.encode_wan_s2v_audio"]
    assert audio_encode.aliases == ("AudioEncoderEncode",)
    assert audio_encode.display_name == "Encode Wan Audio"
    assert {"humo", "whisper"}.issubset(audio_encode.search_terms)
    assert {item.id: item.type.types for item in audio_encode.inputs} == {
        "audio_encoder": ("dinkster.audio_encoder",),
        "audio": ("dinkster.audio",),
    }
    assert audio_encode.outputs[0].type.types == ("dinkster.audio_encoder_output",)

    s2v = schemas["dinkster.wan22_s2v"]
    assert s2v.aliases == ("WanSoundImageToVideo",)
    assert {item.id: item.type.types for item in s2v.inputs} == {
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "vae": ("dinkster.vae",),
        "width": ("core.int",),
        "height": ("core.int",),
        "length": ("core.int",),
        "batch_size": ("core.int",),
        "audio_encoder_output": ("dinkster.audio_encoder_output",),
        "ref_image": ("dinkster.image",),
        "control_video": ("dinkster.image",),
        "ref_motion": ("dinkster.image",),
    }
    assert {item.id: item.type.types for item in s2v.outputs} == {
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "latent": ("dinkster.latent",),
    }
    s2v_extend = schemas["dinkster.wan22_s2v_extend"]
    assert s2v_extend.aliases == ("WanSoundImageToVideoExtend",)
    assert {item.id: item.type.types for item in s2v_extend.inputs} == {
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "vae": ("dinkster.vae",),
        "length": ("core.int",),
        "video_latent": ("dinkster.latent",),
        "audio_encoder_output": ("dinkster.audio_encoder_output",),
        "ref_image": ("dinkster.image",),
        "control_video": ("dinkster.image",),
    }
    humo = schemas["dinkster.wan21_humo"]
    assert humo.aliases == ("WanHuMoImageToVideo",)
    assert {item.id: item.type.types for item in humo.inputs} == {
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "vae": ("dinkster.vae",),
        "width": ("core.int",),
        "height": ("core.int",),
        "length": ("core.int",),
        "batch_size": ("core.int",),
        "audio_encoder_output": ("dinkster.audio_encoder_output",),
        "ref_image": ("dinkster.image",),
    }
    assert {item.id: item.type.types for item in humo.outputs} == {
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "latent": ("dinkster.latent",),
    }
    infinite_talk = schemas["dinkster.wan_infinite_talk_to_video"]
    assert infinite_talk.aliases == ("WanInfiniteTalkToVideo",)
    assert {item.id: item.type.types for item in infinite_talk.inputs} == {
        "mode": ("core.combo",),
        "model": ("dinkster.model",),
        "model_patch": ("comfy.MODEL_PATCH",),
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "vae": ("dinkster.vae",),
        "width": ("core.int",),
        "height": ("core.int",),
        "length": ("core.int",),
        "audio_encoder_output_1": ("dinkster.audio_encoder_output",),
        "motion_frame_count": ("core.int",),
        "audio_scale": ("core.float",),
        "start_image": ("dinkster.image",),
        "previous_frames": ("dinkster.image",),
        "audio_encoder_output_2": ("dinkster.audio_encoder_output",),
        "mask_1": ("dinkster.mask",),
        "mask_2": ("dinkster.mask",),
    }
    assert {item.id: item.type.types for item in infinite_talk.outputs} == {
        "model": ("dinkster.model",),
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "latent": ("dinkster.latent",),
        "trim_image": ("core.int",),
    }
    infinite_inputs = {item.id: item for item in infinite_talk.inputs}
    assert infinite_inputs["mode"].widget == ComboWidget(options=("single_speaker", "two_speakers"))
    assert infinite_inputs["model_patch"].type.types == ("comfy.MODEL_PATCH",)

    dancer_audio = schemas["dinkster.encode_wandancer_audio"]
    assert dancer_audio.aliases == ("WanDancerEncodeAudio",)
    assert {item.id: item.type.types for item in dancer_audio.inputs} == {
        "audio": ("dinkster.audio",),
        "video_frames": ("core.int",),
        "audio_inject_scale": ("core.float",),
    }
    assert {item.id: item.type.types for item in dancer_audio.outputs} == {
        "audio_encoder_output": ("dinkster.audio_encoder_output",),
        "fps_string": ("core.string",),
    }
    dancer_audio_inputs = {item.id: item for item in dancer_audio.inputs}
    assert dancer_audio_inputs["video_frames"].widget == NumberWidget(min=1, max=16384, step=4)
    assert dancer_audio_inputs["audio_inject_scale"].widget == NumberWidget(
        min=0.0, max=10.0, step=0.01
    )

    dancer_video = schemas["dinkster.wan22_dancer_video"]
    assert dancer_video.aliases == ("WanDancerVideo",)
    assert {item.id: item.type.types for item in dancer_video.inputs} == {
        "model": ("dinkster.model",),
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "vae": ("dinkster.vae",),
        "width": ("core.int",),
        "height": ("core.int",),
        "length": ("core.int",),
        "batch_size": ("core.int",),
        "start_image": ("dinkster.image",),
        "mask": ("dinkster.mask",),
        "reference_image": ("dinkster.image",),
        "audio_encoder_output": ("dinkster.audio_encoder_output",),
    }
    assert {item.id: item.type.types for item in dancer_video.outputs} == {
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "latent": ("dinkster.latent",),
    }
    dancer_video_inputs = {item.id: item for item in dancer_video.inputs}
    assert not dancer_video_inputs["start_image"].required
    assert not dancer_video_inputs["mask"].required
    assert not dancer_video_inputs["reference_image"].required
    assert not dancer_video_inputs["audio_encoder_output"].required

    dancer_pad = schemas["dinkster.wandancer_pad_keyframes"]
    assert dancer_pad.aliases == ("WanDancerPadKeyframes",)
    assert {item.id: item.type.types for item in dancer_pad.inputs} == {
        "images": ("dinkster.image",),
        "segment_length": ("core.int",),
        "segment_index": ("core.int",),
        "audio": ("dinkster.audio",),
    }
    assert {item.id: item.type.types for item in dancer_pad.outputs} == {
        "keyframes_sequence": ("dinkster.image",),
        "keyframes_mask": ("dinkster.mask",),
        "audio_segment": ("dinkster.audio",),
    }
    dancer_pad_list = schemas["dinkster.wandancer_pad_keyframe_list"]
    assert dancer_pad_list.aliases == ("WanDancerPadKeyframesList",)
    assert {item.id: item.type.types for item in dancer_pad_list.inputs} == {
        "images": ("dinkster.image",),
        "segment_length": ("core.int",),
        "num_segments": ("core.int",),
        "audio": ("dinkster.audio",),
    }
    assert {item.id: item.type.cardinality() for item in dancer_pad_list.outputs} == {
        "keyframes_sequence": "list",
        "keyframes_mask": "list",
        "audio_segment": "list",
    }


def test_wan_animate_schema_publishes_with_its_own_provider() -> None:
    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(default_pack_spec("dinkster-nodes-foundation"))
            await composer.add_pack(default_pack_spec("dinkster-nodes-media-io"))
            await composer.add_pack(
                PackSpec(GENERATION_MANIFEST, trust_reserved=True, in_process=True)
            )
            delta = await composer.add_pack(
                PackSpec(MANIFEST, trust_reserved=True, in_process=True)
            )
            assert tuple(delta.schemas) == WAN_MODEL_NODE_IDS
            for node_id in WAN_MODEL_NODE_IDS:
                assert node_id in composer.composition.schemas
                assert composer.composition.node_packs[node_id] == "dinkster-model-wan"
            assert composer.incomplete_generation_removals()["dinkster-model-wan"] == ()
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_wan_animate_body_keeps_torch_provider_lazy() -> None:
    assert "dinkster_model_wan.provider" not in sys.modules
    signature = inspect.signature(WAN_MODEL_NODES[0].execute)
    assert tuple(signature.parameters) == ("width", "height", "length", "batch_size")
    assert tuple(inspect.signature(WAN_MODEL_NODES[1].execute).parameters) == (
        "num_frame_per_block",
    )
    assert tuple(inspect.signature(WAN_MODEL_NODES[2].execute).parameters) == (
        "model",
        "vae",
        "start_image",
        "width",
        "height",
        "length",
        "batch_size",
    )
    signature = inspect.signature(WAN_MODEL_NODES[3].execute)
    assert tuple(signature.parameters) == (
        "positive",
        "negative",
        "model",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "continue_motion_max_frames",
        "video_frame_offset",
        "reference_image",
        "face_video",
        "pose_video",
        "background_video",
        "character_mask",
        "continue_motion",
    )
    animate2_signature = inspect.signature(WAN_MODEL_NODES[4].execute)
    assert tuple(animate2_signature.parameters) == (
        "positive",
        "negative",
        "model",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "video_frame_offset",
        "pose_strength",
        "pose_start_percent",
        "pose_end_percent",
        "reference_image_strength",
        "reference_image",
        "pose_video",
        "positive_pose",
        "continue_motion",
    )
    scail_signature = inspect.signature(WAN_MODEL_NODES[5].execute)
    assert tuple(scail_signature.parameters) == (
        "positive",
        "negative",
        "model",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "pose_strength",
        "pose_start_percent",
        "pose_end_percent",
        "video_frame_offset",
        "previous_frame_count",
        "replacement_mode",
        "reference_image",
        "pose_video",
        "pose_video_mask",
        "reference_image_mask",
        "previous_frames",
    )
    assert tuple(inspect.signature(WAN_MODEL_NODES[6].execute).parameters) == ("model_patch",)
    assert tuple(inspect.signature(WAN_MODEL_NODES[7].execute).parameters) == (
        "model",
        "patch",
        "vae",
        "render_video",
        "strength",
        "start_percent",
        "end_percent",
    )
    assert tuple(inspect.signature(WAN_MODEL_NODES[8].execute).parameters) == ("audio_encoder",)
    assert tuple(inspect.signature(WAN_MODEL_NODES[9].execute).parameters) == (
        "audio_encoder",
        "audio",
    )
    assert tuple(inspect.signature(WAN_MODEL_NODES[10].execute).parameters) == (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "audio_encoder_output",
        "ref_image",
        "control_video",
        "ref_motion",
    )
    assert tuple(inspect.signature(WAN_MODEL_NODES[11].execute).parameters) == (
        "positive",
        "negative",
        "vae",
        "length",
        "video_latent",
        "audio_encoder_output",
        "ref_image",
        "control_video",
    )
    assert tuple(inspect.signature(WAN_MODEL_NODES[12].execute).parameters) == (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "audio_encoder_output",
        "ref_image",
    )
    assert tuple(inspect.signature(WAN_MODEL_NODES[13].execute).parameters) == (
        "mode",
        "model",
        "model_patch",
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "audio_encoder_output_1",
        "motion_frame_count",
        "audio_scale",
        "start_image",
        "previous_frames",
        "audio_encoder_output_2",
        "mask_1",
        "mask_2",
    )
    assert tuple(inspect.signature(WAN_MODEL_NODES[14].execute).parameters) == (
        "audio",
        "video_frames",
        "audio_inject_scale",
    )
    assert tuple(inspect.signature(WAN_MODEL_NODES[15].execute).parameters) == (
        "model",
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "start_image",
        "mask",
        "reference_image",
        "audio_encoder_output",
    )
    assert tuple(inspect.signature(WAN_MODEL_NODES[16].execute).parameters) == (
        "images",
        "segment_length",
        "segment_index",
        "audio",
    )
    assert tuple(inspect.signature(WAN_MODEL_NODES[17].execute).parameters) == (
        "images",
        "segment_length",
        "num_segments",
        "audio",
    )
