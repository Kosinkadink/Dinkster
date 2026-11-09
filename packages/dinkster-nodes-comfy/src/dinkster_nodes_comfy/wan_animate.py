"""Native Wan animation conditioning nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_clip_vision_output,
    materialize_image,
)

from .conditioning import conditioning_set_values

_CLIP_VISION_OUTPUT = TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)
_CATEGORY = "model/conditioning/wan/animate"


def _number(
    name: str,
    type_: TypeExpr,
    default: float | int,
    minimum: float | int,
    maximum: float | int,
    step: float | int | None,
    doc: str = "",
) -> InputSpec:
    return InputSpec(
        name,
        type_,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
        doc=doc,
    )


def _upscale(image: Any, width: int, height: int, mode: str) -> Any:
    utils = importlib.import_module("dinkster_inference.utils")
    return utils.common_upscale(image.movedim(-1, 1), width, height, mode, "center").movedim(1, -1)


class WanAnimateToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanAnimateToVideo",
            category=_CATEGORY,
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                _number("width", _INT, 832, 16, 16384, 16),
                _number("height", _INT, 480, 16, 16384, 16),
                _number("length", _INT, 77, 1, 16384, 4),
                _number("batch_size", _INT, 1, 1, 4096, None),
                InputSpec("clip_vision_output", _CLIP_VISION_OUTPUT, required=False),
                InputSpec("reference_image", _IMAGE, required=False),
                InputSpec("face_video", _IMAGE, required=False),
                InputSpec("pose_video", _IMAGE, required=False),
                _number("continue_motion_max_frames", _INT, 5, 1, 16384, 4),
                InputSpec("background_video", _IMAGE, required=False),
                InputSpec("character_mask", _MASK, required=False),
                InputSpec("continue_motion", _IMAGE, required=False),
                _number(
                    "video_frame_offset",
                    _INT,
                    0,
                    0,
                    16384,
                    1,
                    "The amount of frames to seek in all the input videos. Used for generating "
                    "longer videos by chunk. Connect to the video_frame_offset output of the "
                    "previous node for extending a video.",
                ),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec("latent", _LATENT, display_name="latent"),
                OutputSpec("trim_latent", _INT, display_name="trim_latent"),
                OutputSpec("trim_image", _INT, display_name="trim_image"),
                OutputSpec("video_frame_offset", _INT, display_name="video_frame_offset"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        continue_motion_max_frames: int,
        video_frame_offset: int,
        reference_image: object | None = None,
        clip_vision_output: object | None = None,
        face_video: object | None = None,
        pose_video: object | None = None,
        continue_motion: object | None = None,
        background_video: object | None = None,
        character_mask: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        latent_length = ((length - 1) // 4) + 1
        trim_latent = 0
        reference = (
            torch.zeros((1, height, width, 3))
            if reference_image is None
            else materialize_image(reference_image)
        )
        concat_latent_image = vae.encode(
            _upscale(reference[:length], width, height, "area")[:, :, :, :3]
        )
        mask = torch.zeros(
            (1, 4, *concat_latent_image.shape[-3:]),
            device=concat_latent_image.device,
            dtype=concat_latent_image.dtype,
        )
        trim_latent += concat_latent_image.shape[2]
        ref_motion_latent_length = 0
        motion = None if continue_motion is None else materialize_image(continue_motion)
        if motion is None:
            image = torch.ones((length, height, width, 3)) * 0.5
        else:
            motion = motion[-continue_motion_max_frames:]
            video_frame_offset = max(0, video_frame_offset - motion.shape[0])
            motion = _upscale(motion[-length:], width, height, "area")
            image = (
                torch.ones(
                    (length, height, width, motion.shape[-1]),
                    device=motion.device,
                    dtype=motion.dtype,
                )
                * 0.5
            )
            image[: motion.shape[0]] = motion
            ref_motion_latent_length += ((motion.shape[0] - 1) // 4) + 1
        if clip_vision_output is not None:
            values = {"clip_vision_output": materialize_clip_vision_output(clip_vision_output)}
            positive, negative = (
                conditioning_set_values(positive, values),
                conditioning_set_values(negative, values),
            )
        pose = None if pose_video is None else materialize_image(pose_video)
        if pose is not None:
            pose = None if pose.shape[0] <= video_frame_offset else pose[video_frame_offset:]
        if pose is not None:
            pose = _upscale(pose[:length], width, height, "area")
            if pose.shape[0] < length:
                pose = torch.cat((pose, *(pose[-1:],) * (length - pose.shape[0])), dim=0)
            values = {"pose_video_latent": vae.encode(pose[:, :, :, :3])}
            positive, negative = (
                conditioning_set_values(positive, values),
                conditioning_set_values(negative, values),
            )
        face = None if face_video is None else materialize_image(face_video)
        if face is not None:
            face = None if face.shape[0] <= video_frame_offset else face[video_frame_offset:]
        if face is not None:
            face = (
                _upscale(face[:length], 512, 512, "area").movedim(-1, 1).movedim(0, 1).unsqueeze(0)
                * 2.0
                - 1.0
            )
            positive = conditioning_set_values(positive, {"face_video_pixels": face})
            negative = conditioning_set_values(negative, {"face_video_pixels": face * 0.0 - 1.0})
        ref_images_num = max(0, ref_motion_latent_length * 4 - 3)
        background = None if background_video is None else materialize_image(background_video)
        if background is not None and background.shape[0] > video_frame_offset:
            background = _upscale(background[video_frame_offset:][:length], width, height, "area")
            if background.shape[0] > ref_images_num:
                image[ref_images_num : background.shape[0]] = background[ref_images_num:]
        mask_refmotion = torch.ones(
            (1, 1, latent_length * 4, *concat_latent_image.shape[-2:]),
            device=mask.device,
            dtype=mask.dtype,
        )
        if motion is not None:
            mask_refmotion[:, :, : ref_motion_latent_length * 4] = 0.0
        character = None if character_mask is None else materialize_image(character_mask)
        if character is not None and (
            character.shape[0] > video_frame_offset or character.shape[0] == 1
        ):
            if character.shape[0] == 1:
                character = character.repeat((length,) + (1,) * (character.ndim - 1))
            else:
                character = character[video_frame_offset:]
            if character.ndim == 3:
                character = character.unsqueeze(1).movedim(0, 1)
            if character.ndim == 4:
                character = character.unsqueeze(1)
            utils = importlib.import_module("dinkster_inference.utils")
            character = utils.common_upscale(
                character[:, :, :length],
                concat_latent_image.shape[-1],
                concat_latent_image.shape[-2],
                "nearest-exact",
                "center",
            )
            if character.shape[2] > ref_images_num:
                mask_refmotion[:, :, ref_images_num : character.shape[2]] = character[
                    :, :, ref_images_num:
                ]
        concat_latent_image = torch.cat(
            (concat_latent_image, vae.encode(image[:, :, :, :3])), dim=2
        )
        mask_refmotion = mask_refmotion.view(
            1,
            mask_refmotion.shape[2] // 4,
            4,
            mask_refmotion.shape[3],
            mask_refmotion.shape[4],
        ).transpose(1, 2)
        values = {
            "concat_latent_image": concat_latent_image,
            "concat_mask": torch.cat((mask, mask_refmotion), dim=2),
        }
        positive, negative = (
            conditioning_set_values(positive, values),
            conditioning_set_values(negative, values),
        )
        latent = torch.zeros(
            (batch_size, 16, latent_length + trim_latent, height // 8, width // 8),
            device=management.intermediate_device(),
        )
        return cls.outputs(
            positive=positive,
            negative=negative,
            latent={"samples": latent},
            trim_latent=trim_latent,
            trim_image=max(0, ref_motion_latent_length * 4 - 3),
            video_frame_offset=video_frame_offset + length,
        )


class WanAnimate2ToVideo(Node):
    CONTINUE_MOTION_FRAMES = 1

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanAnimate2ToVideo",
            category=_CATEGORY,
            description=(
                "Animate a character in a reference image using a video, effectively replicating "
                "the facial animation, body motion and hands gesture from the video."
            ),
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                _number("width", _INT, 832, 16, 16384, 16, "Output video width in pixels."),
                _number("height", _INT, 480, 16, 16384, 16, "Output video height in pixels."),
                _number("length", _INT, 81, 1, 16384, 4, "Number of frames to generate."),
                _number(
                    "batch_size",
                    _INT,
                    1,
                    1,
                    4096,
                    None,
                    "Number of videos to generate simultaneously.",
                ),
                InputSpec(
                    "reference_image", _IMAGE, required=False, doc="The character to animate."
                ),
                InputSpec(
                    "pose_video",
                    _IMAGE,
                    required=False,
                    doc="The video whose motion is transferred to the reference character.",
                ),
                InputSpec(
                    "clip_vision_output",
                    _CLIP_VISION_OUTPUT,
                    required=False,
                    doc="CLIP vision of the reference image.",
                ),
                InputSpec(
                    "positive_pose",
                    _CONDITIONING,
                    required=False,
                    doc=(
                        "Prompt for the pose-video branch, describing the motion rather than the "
                        "character. Defaults to positive. Used for both the cond and uncond passes."
                    ),
                ),
                InputSpec(
                    "clip_vision_output_pose",
                    _CLIP_VISION_OUTPUT,
                    required=False,
                    doc=(
                        "CLIP vision of the pose video's first frame. Defaults to "
                        "clip_vision_output."
                    ),
                ),
                InputSpec(
                    "continue_motion",
                    _IMAGE,
                    required=False,
                    doc="Previous motion sequence to continue from for temporal consistency.",
                ),
                _number(
                    "video_frame_offset",
                    _INT,
                    0,
                    0,
                    16384,
                    1,
                    "Frames to seek into the pose video. Connect to the video_frame_offset "
                    "output of the previous node when extending.",
                ),
                _number(
                    "pose_strength",
                    _FLOAT,
                    1.0,
                    0.0,
                    10.0,
                    0.01,
                    "Scales the pose video's influence on the motion. 1.0 is the trained "
                    "behavior; below weakens adherence, above amplifies. 0.0 mutes it but does "
                    "not fully remove it.",
                ),
                _number(
                    "pose_start_percent",
                    _FLOAT,
                    0.0,
                    0.0,
                    1.0,
                    0.01,
                    "Sampling percent at which the pose influence starts. Outside the window "
                    "the pose branch is skipped entirely, which also speeds those steps up.",
                ),
                _number(
                    "pose_end_percent",
                    _FLOAT,
                    1.0,
                    0.0,
                    1.0,
                    0.01,
                    "Sampling percent at which the pose influence ends. Motion is mostly "
                    "established early, so e.g. 0.7 can loosen fine detail while keeping the "
                    "choreography.",
                ),
                _number(
                    "reference_image_strength",
                    _FLOAT,
                    1.0,
                    0.0,
                    10.0,
                    0.01,
                    "Scales how strongly generated frames attend to the reference image's "
                    "latent frame. Below 1.0 loosens identity/appearance adherence (e.g. to let "
                    "the prompt restyle), above tightens it against drift.",
                ),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec("latent", _LATENT, display_name="latent"),
                OutputSpec(
                    "trim_latent",
                    _INT,
                    display_name="trim_latent",
                    doc="Number of latent frames that should be trimmed before decoding.",
                ),
                OutputSpec(
                    "trim_image",
                    _INT,
                    display_name="trim_image",
                    doc="Number of overlapping image frames when extending a video.",
                ),
                OutputSpec(
                    "video_frame_offset",
                    _INT,
                    display_name="video_frame_offset",
                    doc="Frames to seek into the pose video.",
                ),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        video_frame_offset: int,
        reference_image: object | None = None,
        pose_video: object | None = None,
        clip_vision_output: object | None = None,
        positive_pose: Sequence[Sequence[Any]] | None = None,
        clip_vision_output_pose: object | None = None,
        continue_motion: object | None = None,
        pose_strength: float = 1.0,
        pose_start_percent: float = 0.0,
        pose_end_percent: float = 1.0,
        reference_image_strength: float = 1.0,
    ) -> Mapping[str, object]:
        if pose_start_percent > pose_end_percent:
            raise ValueError(
                f"pose_start_percent ({pose_start_percent}) must not be greater than "
                f"pose_end_percent ({pose_end_percent})."
            )
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        latent_length = ((length - 1) // 4) + 1
        reference = (
            torch.zeros((1, height, width, 3))
            if reference_image is None
            else materialize_image(reference_image)
        )
        ref_latent = vae.encode(_upscale(reference[:1], width, height, "area")[:, :, :, :3])
        trim_latent = ref_latent.shape[2]
        ref_motion_latent_length = 0
        motion = None if continue_motion is None else materialize_image(continue_motion)
        if motion is None:
            image = torch.ones((length, height, width, 3)) * 0.5
        else:
            motion = motion[-cls.CONTINUE_MOTION_FRAMES :]
            video_frame_offset = max(0, video_frame_offset - motion.shape[0])
            motion = _upscale(motion[-length:], width, height, "area")
            image = (
                torch.ones(
                    (length, height, width, motion.shape[-1]),
                    device=motion.device,
                    dtype=motion.dtype,
                )
                * 0.5
            )
            image[: motion.shape[0]] = motion
            ref_motion_latent_length += ((motion.shape[0] - 1) // 4) + 1
        concat_latent_image = torch.cat((ref_latent, vae.encode(image[:, :, :, :3])), dim=2)
        mask = torch.ones(
            (1, 1, latent_length + trim_latent, height // 8, width // 8),
            device=concat_latent_image.device,
            dtype=concat_latent_image.dtype,
        )
        mask[:, :, : trim_latent + ref_motion_latent_length] = 0.0
        values = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        positive, negative = (
            conditioning_set_values(positive, values),
            conditioning_set_values(negative, values),
        )
        clip = (
            None
            if clip_vision_output is None
            else materialize_clip_vision_output(clip_vision_output)
        )
        if clip is not None:
            positive, negative = (
                conditioning_set_values(positive, {"clip_vision_output": clip}),
                conditioning_set_values(negative, {"clip_vision_output": clip}),
            )
        if reference_image_strength != 1.0:
            positive, negative = (
                conditioning_set_values(positive, {"reference_strength": reference_image_strength}),
                conditioning_set_values(negative, {"reference_strength": reference_image_strength}),
            )
        pose_values: dict[str, object] = {}
        pose = None if pose_video is None else materialize_image(pose_video)
        if pose is not None:
            if pose.shape[0] <= video_frame_offset:
                raise ValueError(
                    f"pose_video has {pose.shape[0]} frames but video_frame_offset is "
                    f"{video_frame_offset} -- nothing left to read."
                )
            pose = _upscale(pose[video_frame_offset:][:length], width, height, "area")
            if pose.shape[0] < length:
                pose = torch.cat((pose, *(pose[-1:],) * (length - pose.shape[0])), dim=0)
            pose_values["pose_video_latent"] = vae.encode(pose[:, :, :, :3])
        pose_clip = (
            clip_vision_output_pose if clip_vision_output_pose is not None else clip_vision_output
        )
        if pose_clip is not None:
            pose_values["clip_vision_output_pose"] = materialize_clip_vision_output(pose_clip)
        pose_cond = positive_pose if positive_pose is not None else positive
        if pose_cond:
            pose_values["cross_attn_pose"] = pose_cond[0][0]
        if pose_strength != 1.0:
            pose_values["pose_strength"] = pose_strength
        if pose_start_percent > 0.0 or pose_end_percent < 1.0:

            def windowed(conditioning: Sequence[Sequence[Any]]) -> list[list[Any]]:
                parts = conditioning_set_values(
                    conditioning,
                    {
                        **pose_values,
                        "start_percent": pose_start_percent,
                        "end_percent": pose_end_percent,
                    },
                )
                if pose_start_percent > 0.0:
                    parts += conditioning_set_values(
                        conditioning, {"start_percent": 0.0, "end_percent": pose_start_percent}
                    )
                if pose_end_percent < 1.0:
                    parts += conditioning_set_values(
                        conditioning, {"start_percent": pose_end_percent, "end_percent": 1.0}
                    )
                return parts

            positive, negative = windowed(positive), windowed(negative)
        else:
            positive, negative = (
                conditioning_set_values(positive, pose_values),
                conditioning_set_values(negative, pose_values),
            )
        latent = torch.zeros(
            (batch_size, 16, latent_length + trim_latent, height // 8, width // 8),
            device=management.intermediate_device(),
        )
        return cls.outputs(
            positive=positive,
            negative=negative,
            latent={"samples": latent},
            trim_latent=trim_latent,
            trim_image=max(0, ref_motion_latent_length * 4 - 3),
            video_frame_offset=video_frame_offset + length,
        )


class WanAnimate2Cache(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanAnimate2Cache",
            category=_CATEGORY,
            description=(
                "Caches the pose-video's per-block activations so it runs once instead of on "
                "every sampling step. Roughly halves generation time Tradeoff is ~12.5 GB of "
                "system RAM at 480x832/81 frames in bf16 (scales with resolution and length). "
                "With context windows each window is cached separately, so RAM scales with the "
                "window count; use the static_standard schedule, as uniform schedules shift the "
                "windows every step and nothing ever recurs to hit the cache."
            ),
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "device",
                    _COMBO,
                    default="cpu",
                    widget=ComboWidget(("cpu", "gpu")),
                    doc=(
                        "Where to keep the cache. cpu (RAM) is the safe choice, the cache will not "
                        "fit in VRAM alongside the model at typical sizes. gpu (VRAM) can be "
                        "faster "
                        "if it fits."
                    ),
                ),
                InputSpec(
                    "dtype",
                    _COMBO,
                    default="default",
                    widget=ComboWidget(("default", "int8", "int4")),
                    doc=(
                        "Storage precision. default stores the activations in the model's compute "
                        "dtype. int8 halves the cache, int4 quarters it, convrot is used to retain "
                        "accuracy."
                    ),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(cls, *, model: Any, device: str, dtype: str = "default") -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        model_animate2 = importlib.import_module("dinkster_inference.ldm.wan.model_animate2")
        patcher_extension = importlib.import_module("dinkster_inference.patcher_extension")
        store = management.get_torch_device() if device == "gpu" else torch.device("cpu")
        cache = model_animate2.PoseBranchCache(store_device=store, dtype=dtype)
        patched = model.clone()
        patched.model_options.setdefault("transformer_options", {})["animate2_cache"] = cache

        def cleanup_callback(_patcher: Any) -> None:
            cache.free()

        patched.add_callback(patcher_extension.CallbacksMP.ON_CLEANUP, cleanup_callback)
        return cls.outputs(MODEL=patched)


WAN_ANIMATE_NODES = (WanAnimateToVideo, WanAnimate2ToVideo, WanAnimate2Cache)
