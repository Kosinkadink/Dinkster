"""Wan model nodes."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    AssetWidget,
    BooleanWidget,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

ASSET = TypeExpr.concrete("dinkster.asset")
MODEL = TypeExpr.concrete("dinkster.model")
VAE = TypeExpr.concrete("dinkster.vae")
CONDITIONING = TypeExpr.concrete("dinkster.conditioning")
LATENT = TypeExpr.concrete("dinkster.latent")
IMAGE = TypeExpr.concrete("dinkster.image")
MASK = TypeExpr.concrete("dinkster.mask")
SAMPLER = TypeExpr.concrete("dinkster.sampler")
AUDIO = TypeExpr.concrete("dinkster.audio")
AUDIO_ENCODER = TypeExpr.concrete("dinkster.audio_encoder")
AUDIO_ENCODER_OUTPUT = TypeExpr.concrete("dinkster.audio_encoder_output")
MODEL_PATCH = TypeExpr.concrete("comfy.MODEL_PATCH")
WAN21_UNI3C = TypeExpr.concrete("dinkster.wan21_uni3c")
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
COMBO = TypeExpr.concrete(CORE_COMBO)
STRING = TypeExpr.concrete(CORE_STRING)
IMAGE_LIST = TypeExpr.list_of(IMAGE)
MASK_LIST = TypeExpr.list_of(MASK)
AUDIO_LIST = TypeExpr.list_of(AUDIO)


class EmptyARVideoLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_ar_video_latent",
            display_name="Empty AR Video Latent",
            category="model/latent/autoregressive",
            inputs=(
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=8192, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=8192, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=1024, step=4)),
                InputSpec("batch_size", INT, default=1, widget=NumberWidget(min=1, max=64, step=1)),
            ),
            outputs=(OutputSpec("latent", LATENT),),
        )

    @classmethod
    def execute(
        cls, *, width: int, height: int, length: int, batch_size: int
    ) -> Mapping[str, object]:
        from .provider import execute_empty_ar_video_latent

        return execute_empty_ar_video_latent(
            width=width, height=height, length=length, batch_size=batch_size
        )


class SamplerARVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.sampler_ar_video",
            display_name="Sampler AR Video",
            category="model/sampling/samplers",
            inputs=(
                InputSpec(
                    "num_frame_per_block",
                    INT,
                    default=1,
                    widget=NumberWidget(min=1, max=64, step=1),
                ),
            ),
            outputs=(OutputSpec("sampler", SAMPLER),),
        )

    @classmethod
    def execute(cls, *, num_frame_per_block: int) -> Mapping[str, object]:
        from .provider import execute_sampler_ar_video

        return execute_sampler_ar_video(num_frame_per_block=num_frame_per_block)


class ARVideoI2V(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.ar_video_i2v",
            display_name="AR Video I2V",
            category="model/conditioning/autoregressive",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("vae", VAE),
                InputSpec("start_image", IMAGE),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=8192, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=8192, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=1024, step=4)),
                InputSpec("batch_size", INT, default=1, widget=NumberWidget(min=1, max=64, step=1)),
            ),
            outputs=(OutputSpec("model", MODEL), OutputSpec("latent", LATENT)),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        vae: object,
        start_image: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
    ) -> Mapping[str, object]:
        from .provider import execute_ar_video_i2v

        return execute_ar_video_i2v(
            model=model,
            vae=vae,
            start_image=start_image,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
        )


class LoadWan21Uni3C(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_wan21_uni3c",
            display_name="Load Wan 2.1 Uni3C",
            category="model/loaders/wan",
            inputs=(
                InputSpec(
                    "model_patch",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/patch",
                    ),
                ),
            ),
            outputs=(OutputSpec("patch", WAN21_UNI3C),),
            search_terms=("wan", "wan 2.1", "uni3c", "control"),
        )

    @classmethod
    def execute(cls, model_patch: object) -> Mapping[str, object]:
        from .provider import execute_load_wan21_uni3c

        return execute_load_wan21_uni3c(model_patch=model_patch)


class ApplyWan21Uni3C(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.apply_wan21_uni3c",
            display_name="Apply Wan 2.1 Uni3C",
            category="model/control/wan",
            description="Applies a Uni3C patch to base Wan 2.1 14B using an RGB render video.",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("patch", WAN21_UNI3C),
                InputSpec("vae", VAE),
                InputSpec("render_video", IMAGE),
                InputSpec(
                    "strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "start_percent",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "end_percent",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", MODEL),),
            search_terms=("wan", "wan 2.1", "uni3c", "control", "render video"),
        )

    @classmethod
    def execute(
        cls,
        model: object,
        patch: object,
        vae: object,
        render_video: object,
        strength: float = 1.0,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
    ) -> Mapping[str, object]:
        from .provider import execute_apply_wan21_uni3c

        return execute_apply_wan21_uni3c(
            model=model,
            patch=patch,
            vae=vae,
            render_video=render_video,
            strength=strength,
            start_percent=start_percent,
            end_percent=end_percent,
        )


class Wan22AnimateToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan22_animate_to_video",
            display_name="WanAnimateToVideo",
            category="model/conditioning/wan/animate",
            description=(
                "Builds Wan 2.2 Animate conditioning and latent geometry from reference, "
                "pose, face, background, mask, and continuation inputs."
            ),
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("model", MODEL),
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=77, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec("reference_image", IMAGE, required=False, default=None),
                InputSpec("face_video", IMAGE, required=False, default=None),
                InputSpec("pose_video", IMAGE, required=False, default=None),
                InputSpec(
                    "continue_motion_max_frames",
                    INT,
                    default=5,
                    widget=NumberWidget(min=1, max=16384, step=4),
                    advanced=True,
                ),
                InputSpec("background_video", IMAGE, required=False, default=None),
                InputSpec("character_mask", MASK, required=False, default=None),
                InputSpec("continue_motion", IMAGE, required=False, default=None),
                InputSpec(
                    "video_frame_offset",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, max=16384, step=1),
                    advanced=True,
                ),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
                OutputSpec("trim_latent", INT),
                OutputSpec("trim_image", INT),
                OutputSpec("video_frame_offset", INT),
            ),
            search_terms=("wan", "wan 2.2", "animate", "pose", "face", "video"),
        )

    @classmethod
    def execute(
        cls,
        positive: object,
        negative: object,
        model: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        continue_motion_max_frames: int,
        video_frame_offset: int,
        reference_image: object | None = None,
        face_video: object | None = None,
        pose_video: object | None = None,
        background_video: object | None = None,
        character_mask: object | None = None,
        continue_motion: object | None = None,
    ) -> Mapping[str, object]:
        from .provider import execute_wan22_animate_to_video

        return execute_wan22_animate_to_video(
            positive=positive,
            negative=negative,
            model=model,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            continue_motion_max_frames=continue_motion_max_frames,
            video_frame_offset=video_frame_offset,
            reference_image=reference_image,
            face_video=face_video,
            pose_video=pose_video,
            background_video=background_video,
            character_mask=character_mask,
            continue_motion=continue_motion,
        )


class Wan21Animate2ToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan21_animate2_to_video",
            display_name="WanAnimate2ToVideo",
            category="model/conditioning/wan/animate",
            description=(
                "Builds Wan 2.1 Animate2 conditioning and latent geometry from reference, "
                "pose, and continuation inputs."
            ),
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("model", MODEL),
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec(
                    "video_frame_offset",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, max=16384, step=1),
                    advanced=True,
                ),
                InputSpec(
                    "pose_strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "pose_start_percent",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "pose_end_percent",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "reference_image_strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
                InputSpec("reference_image", IMAGE, required=False, default=None),
                InputSpec("pose_video", IMAGE, required=False, default=None),
                InputSpec("positive_pose", CONDITIONING, required=False, default=None),
                InputSpec("continue_motion", IMAGE, required=False, default=None),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
                OutputSpec("trim_latent", INT),
                OutputSpec("trim_image", INT),
                OutputSpec("video_frame_offset", INT),
            ),
            search_terms=("wan", "wan 2.1", "animate2", "pose", "video"),
        )

    @classmethod
    def execute(
        cls,
        positive: object,
        negative: object,
        model: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        video_frame_offset: int,
        pose_strength: float,
        pose_start_percent: float,
        pose_end_percent: float,
        reference_image_strength: float,
        reference_image: object | None = None,
        pose_video: object | None = None,
        positive_pose: object | None = None,
        continue_motion: object | None = None,
    ) -> Mapping[str, object]:
        from .provider import execute_wan21_animate2_to_video

        return execute_wan21_animate2_to_video(
            positive=positive,
            negative=negative,
            model=model,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            video_frame_offset=video_frame_offset,
            pose_strength=pose_strength,
            pose_start_percent=pose_start_percent,
            pose_end_percent=pose_end_percent,
            reference_image_strength=reference_image_strength,
            reference_image=reference_image,
            pose_video=pose_video,
            positive_pose=positive_pose,
            continue_motion=continue_motion,
        )


class Wan21SCAILToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan21_scail_to_video",
            display_name="WanSCAILToVideo",
            category="model/conditioning/wan/scail",
            description=(
                "Builds Wan 2.1 SCAIL or SCAIL2 reference, pose, identity-mask, "
                "and continuation conditioning."
            ),
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("model", MODEL),
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=512, widget=NumberWidget(min=32, max=16384, step=32)
                ),
                InputSpec(
                    "height", INT, default=896, widget=NumberWidget(min=32, max=16384, step=32)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec(
                    "pose_strength",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "pose_start_percent",
                    FLOAT,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "pose_end_percent",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.01),
                ),
                InputSpec(
                    "video_frame_offset",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, max=16384, step=1),
                    advanced=True,
                ),
                InputSpec(
                    "previous_frame_count",
                    INT,
                    default=5,
                    widget=NumberWidget(min=1, max=16384, step=4),
                    advanced=True,
                ),
                InputSpec(
                    "replacement_mode",
                    BOOLEAN,
                    default=False,
                    widget=BooleanWidget(label_on="replacement", label_off="animation"),
                ),
                InputSpec("reference_image", IMAGE, required=False, default=None),
                InputSpec("pose_video", IMAGE, required=False, default=None),
                InputSpec("pose_video_mask", IMAGE, required=False, default=None),
                InputSpec("reference_image_mask", IMAGE, required=False, default=None),
                InputSpec("previous_frames", IMAGE, required=False, default=None),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
                OutputSpec("video_frame_offset", INT),
            ),
            search_terms=(
                "wan",
                "wan 2.1",
                "scail",
                "scail2",
                "character replacement",
                "pose",
                "video",
            ),
        )

    @classmethod
    def execute(
        cls,
        positive: object,
        negative: object,
        model: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        pose_strength: float,
        pose_start_percent: float,
        pose_end_percent: float,
        video_frame_offset: int,
        previous_frame_count: int,
        replacement_mode: bool,
        reference_image: object | None = None,
        pose_video: object | None = None,
        pose_video_mask: object | None = None,
        reference_image_mask: object | None = None,
        previous_frames: object | None = None,
    ) -> Mapping[str, object]:
        from .provider import execute_wan21_scail_to_video

        return execute_wan21_scail_to_video(
            positive=positive,
            negative=negative,
            model=model,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            pose_strength=pose_strength,
            pose_start_percent=pose_start_percent,
            pose_end_percent=pose_end_percent,
            video_frame_offset=video_frame_offset,
            previous_frame_count=previous_frame_count,
            replacement_mode=replacement_mode,
            reference_image=reference_image,
            pose_video=pose_video,
            pose_video_mask=pose_video_mask,
            reference_image_mask=reference_image_mask,
            previous_frames=previous_frames,
        )


class LoadWanS2VAudioEncoder(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_wan_s2v_audio_encoder",
            display_name="Load Wan Audio Encoder",
            category="model/loaders/wan",
            inputs=(
                InputSpec(
                    "audio_encoder",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/audio-encoder",
                    ),
                ),
            ),
            outputs=(OutputSpec("audio_encoder", AUDIO_ENCODER),),
            aliases=("AudioEncoderLoader",),
            search_terms=("wan", "s2v", "humo", "wav2vec2", "whisper", "audio"),
        )

    @classmethod
    def execute(cls, *, audio_encoder: object) -> Mapping[str, object]:
        from .provider import execute_load_wav2vec2_audio_encoder

        return execute_load_wav2vec2_audio_encoder(audio_encoder=audio_encoder)


class EncodeWanS2VAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.encode_wan_s2v_audio",
            display_name="Encode Wan Audio",
            category="model/conditioning/wan/sound",
            inputs=(
                InputSpec("audio_encoder", AUDIO_ENCODER),
                InputSpec("audio", AUDIO),
            ),
            outputs=(OutputSpec("audio_encoder_output", AUDIO_ENCODER_OUTPUT),),
            aliases=("AudioEncoderEncode",),
            search_terms=("wan", "s2v", "humo", "wav2vec2", "whisper", "audio"),
        )

    @classmethod
    def execute(cls, *, audio_encoder: object, audio: object) -> Mapping[str, object]:
        from .provider import execute_encode_wav2vec2_audio

        return execute_encode_wav2vec2_audio(audio_encoder=audio_encoder, audio=audio)


class Wan22S2V(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan22_s2v",
            display_name="Wan 2.2 Sound Image to Video",
            category="model/conditioning/wan/sound",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=77, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec(
                    "audio_encoder_output", AUDIO_ENCODER_OUTPUT, required=False, default=None
                ),
                InputSpec("ref_image", IMAGE, required=False, default=None),
                InputSpec("control_video", IMAGE, required=False, default=None),
                InputSpec("ref_motion", IMAGE, required=False, default=None),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("WanSoundImageToVideo",),
            search_terms=("wan", "wan 2.2", "s2v", "sound", "audio"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        audio_encoder_output: object | None = None,
        ref_image: object | None = None,
        control_video: object | None = None,
        ref_motion: object | None = None,
    ) -> Mapping[str, object]:
        from .provider import execute_wan22_s2v

        return execute_wan22_s2v(
            positive=positive,
            negative=negative,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            audio_encoder_output=audio_encoder_output,
            ref_image=ref_image,
            control_video=control_video,
            ref_motion=ref_motion,
        )


class Wan22S2VExtend(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan22_s2v_extend",
            display_name="Wan 2.2 Sound Image to Video Extend",
            category="model/conditioning/wan/sound",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec("length", INT, default=77, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec("video_latent", LATENT),
                InputSpec(
                    "audio_encoder_output", AUDIO_ENCODER_OUTPUT, required=False, default=None
                ),
                InputSpec("ref_image", IMAGE, required=False, default=None),
                InputSpec("control_video", IMAGE, required=False, default=None),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("WanSoundImageToVideoExtend",),
            search_terms=("wan", "wan 2.2", "s2v", "sound", "audio", "extend"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        length: int,
        video_latent: object,
        audio_encoder_output: object | None = None,
        ref_image: object | None = None,
        control_video: object | None = None,
    ) -> Mapping[str, object]:
        from .provider import execute_wan22_s2v_extend

        return execute_wan22_s2v_extend(
            positive=positive,
            negative=negative,
            vae=vae,
            length=length,
            video_latent=video_latent,
            audio_encoder_output=audio_encoder_output,
            ref_image=ref_image,
            control_video=control_video,
        )


class Wan21Humo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan21_humo",
            display_name="Wan 2.1 HuMo Image to Video",
            category="model/conditioning/wan/humo",
            inputs=(
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=97, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec(
                    "audio_encoder_output", AUDIO_ENCODER_OUTPUT, required=False, default=None
                ),
                InputSpec("ref_image", IMAGE, required=False, default=None),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("WanHuMoImageToVideo",),
            search_terms=("wan", "wan 2.1", "humo", "human motion", "audio"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        audio_encoder_output: object | None = None,
        ref_image: object | None = None,
    ) -> Mapping[str, object]:
        from .provider import execute_wan21_humo

        return execute_wan21_humo(
            positive=positive,
            negative=negative,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            audio_encoder_output=audio_encoder_output,
            ref_image=ref_image,
        )


class WanInfiniteTalkToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan_infinite_talk_to_video",
            display_name="Wan InfiniteTalk Image to Video",
            category="model/conditioning/wan/infinite talk",
            inputs=(
                InputSpec(
                    "mode",
                    COMBO,
                    default="single_speaker",
                    widget=ComboWidget(options=("single_speaker", "two_speakers")),
                ),
                InputSpec("model", MODEL),
                InputSpec("model_patch", MODEL_PATCH),
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec("length", INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)),
                InputSpec("audio_encoder_output_1", AUDIO_ENCODER_OUTPUT),
                InputSpec(
                    "motion_frame_count",
                    INT,
                    default=9,
                    widget=NumberWidget(min=1, max=33, step=1),
                ),
                InputSpec(
                    "audio_scale",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
                InputSpec("start_image", IMAGE, required=False, default=None),
                InputSpec("previous_frames", IMAGE, required=False, default=None),
                InputSpec(
                    "audio_encoder_output_2", AUDIO_ENCODER_OUTPUT, required=False, default=None
                ),
                InputSpec("mask_1", MASK, required=False, default=None),
                InputSpec("mask_2", MASK, required=False, default=None),
            ),
            outputs=(
                OutputSpec("model", MODEL),
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
                OutputSpec("trim_image", INT),
            ),
            aliases=("WanInfiniteTalkToVideo",),
            search_terms=("wan", "wan 2.1", "infinite talk", "multitalk", "audio"),
        )

    @classmethod
    def execute(
        cls,
        *,
        mode: str,
        model: object,
        model_patch: object,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        audio_encoder_output_1: object,
        motion_frame_count: int,
        audio_scale: float,
        start_image: object | None = None,
        previous_frames: object | None = None,
        audio_encoder_output_2: object | None = None,
        mask_1: object | None = None,
        mask_2: object | None = None,
    ) -> Mapping[str, object]:
        from .provider import execute_wan_infinite_talk_to_video

        return execute_wan_infinite_talk_to_video(
            mode=mode,
            model=model,
            model_patch=model_patch,
            positive=positive,
            negative=negative,
            vae=vae,
            width=width,
            height=height,
            length=length,
            audio_encoder_output_1=audio_encoder_output_1,
            motion_frame_count=motion_frame_count,
            audio_scale=audio_scale,
            start_image=start_image,
            previous_frames=previous_frames,
            audio_encoder_output_2=audio_encoder_output_2,
            mask_1=mask_1,
            mask_2=mask_2,
        )


class EncodeWanDancerAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.encode_wandancer_audio",
            display_name="Encode WanDancer Audio",
            category="model/conditioning/wan/dancer",
            inputs=(
                InputSpec("audio", AUDIO),
                InputSpec(
                    "video_frames",
                    INT,
                    default=149,
                    widget=NumberWidget(min=1, max=16384, step=4),
                ),
                InputSpec(
                    "audio_inject_scale",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
            ),
            outputs=(
                OutputSpec("audio_encoder_output", AUDIO_ENCODER_OUTPUT),
                OutputSpec("fps_string", STRING),
            ),
            aliases=("WanDancerEncodeAudio",),
            search_terms=("wan", "wan 2.2", "wandancer", "dance", "music", "audio"),
        )

    @classmethod
    def execute(
        cls,
        *,
        audio: object,
        video_frames: int,
        audio_inject_scale: float,
    ) -> Mapping[str, object]:
        from .provider import execute_encode_wandancer_audio

        return execute_encode_wandancer_audio(
            audio=audio,
            video_frames=video_frames,
            audio_inject_scale=audio_inject_scale,
        )


class Wan22DancerVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wan22_dancer_video",
            display_name="WanDancer Video",
            category="model/conditioning/wan/dancer",
            inputs=(
                InputSpec("model", MODEL),
                InputSpec("positive", CONDITIONING),
                InputSpec("negative", CONDITIONING),
                InputSpec("vae", VAE),
                InputSpec(
                    "width", INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "length", INT, default=149, widget=NumberWidget(min=1, max=16384, step=4)
                ),
                InputSpec(
                    "batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096, step=1)
                ),
                InputSpec("start_image", IMAGE, required=False, default=None),
                InputSpec("mask", MASK, required=False, default=None),
                InputSpec("reference_image", IMAGE, required=False, default=None),
                InputSpec(
                    "audio_encoder_output", AUDIO_ENCODER_OUTPUT, required=False, default=None
                ),
            ),
            outputs=(
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("latent", LATENT),
            ),
            aliases=("WanDancerVideo",),
            search_terms=("wan", "wan 2.2", "wandancer", "dance", "music", "video"),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        start_image: object | None = None,
        mask: object | None = None,
        reference_image: object | None = None,
        audio_encoder_output: object | None = None,
    ) -> Mapping[str, object]:
        from .provider import execute_wan22_dancer_video

        return execute_wan22_dancer_video(
            model=model,
            positive=positive,
            negative=negative,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            start_image=start_image,
            mask=mask,
            reference_image=reference_image,
            audio_encoder_output=audio_encoder_output,
        )


class WanDancerPadKeyframes(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wandancer_pad_keyframes",
            display_name="WanDancer Pad Keyframes",
            category="image/video",
            inputs=(
                InputSpec("images", IMAGE),
                InputSpec(
                    "segment_length",
                    INT,
                    default=149,
                    widget=NumberWidget(min=1, max=10000, step=1),
                ),
                InputSpec(
                    "segment_index",
                    INT,
                    default=0,
                    widget=NumberWidget(min=0, max=100, step=1),
                ),
                InputSpec("audio", AUDIO),
            ),
            outputs=(
                OutputSpec("keyframes_sequence", IMAGE),
                OutputSpec("keyframes_mask", MASK),
                OutputSpec("audio_segment", AUDIO),
            ),
            aliases=("WanDancerPadKeyframes",),
            search_terms=("wan", "wandancer", "dance", "keyframe", "segment"),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        segment_length: int,
        segment_index: int,
        audio: object,
    ) -> Mapping[str, object]:
        from .provider import execute_wandancer_pad_keyframes

        return execute_wandancer_pad_keyframes(
            images=images,
            segment_length=segment_length,
            segment_index=segment_index,
            audio=audio,
        )


class WanDancerPadKeyframeList(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.wandancer_pad_keyframe_list",
            display_name="WanDancer Pad Keyframe List",
            category="image/video",
            inputs=(
                InputSpec("images", IMAGE),
                InputSpec(
                    "segment_length",
                    INT,
                    default=149,
                    widget=NumberWidget(min=1, max=10000, step=1),
                ),
                InputSpec(
                    "num_segments",
                    INT,
                    default=1,
                    widget=NumberWidget(min=1, max=100, step=1),
                ),
                InputSpec("audio", AUDIO),
            ),
            outputs=(
                OutputSpec("keyframes_sequence", IMAGE_LIST),
                OutputSpec("keyframes_mask", MASK_LIST),
                OutputSpec("audio_segment", AUDIO_LIST),
            ),
            aliases=("WanDancerPadKeyframesList",),
            search_terms=("wan", "wandancer", "dance", "keyframe", "segment", "list"),
        )

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        segment_length: int,
        num_segments: int,
        audio: object,
    ) -> Mapping[str, object]:
        from .provider import execute_wandancer_pad_keyframe_list

        return execute_wandancer_pad_keyframe_list(
            images=images,
            segment_length=segment_length,
            num_segments=num_segments,
            audio=audio,
        )


WAN_MODEL_NODES: tuple[type[Node], ...] = (
    EmptyARVideoLatent,
    SamplerARVideo,
    ARVideoI2V,
    Wan22AnimateToVideo,
    Wan21Animate2ToVideo,
    Wan21SCAILToVideo,
    LoadWan21Uni3C,
    ApplyWan21Uni3C,
    LoadWanS2VAudioEncoder,
    EncodeWanS2VAudio,
    Wan22S2V,
    Wan22S2VExtend,
    Wan21Humo,
    WanInfiniteTalkToVideo,
    EncodeWanDancerAudio,
    Wan22DancerVideo,
    WanDancerPadKeyframes,
    WanDancerPadKeyframeList,
)
WAN_MODEL_NODE_IDS = tuple(node.schema().node_type for node in WAN_MODEL_NODES)

__all__ = ["WAN_MODEL_NODE_IDS", "WAN_MODEL_NODES"]
