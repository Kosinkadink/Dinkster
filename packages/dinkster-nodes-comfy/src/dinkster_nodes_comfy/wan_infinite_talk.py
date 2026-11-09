"""Wan InfiniteTalk audio conditioning and model patches."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    MODEL_PATCH_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_clip_vision_output,
    materialize_image,
    materialize_latent,
)

from .conditioning import conditioning_set_values

_INT = TypeExpr.concrete(CORE_INT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_AUDIO = TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)


class WanInfiniteTalkToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanInfiniteTalkToVideo",
            category="model/conditioning/wan/infinite talk",
            combos=(
                DynamicComboSpec(
                    "mode",
                    (
                        DynamicComboOption("single_speaker"),
                        DynamicComboOption(
                            "two_speakers",
                            (
                                InputSpec("audio_encoder_output_2", _AUDIO, required=False),
                                InputSpec(
                                    "mask_1",
                                    TypeExpr.concrete(MASK_TYPE),
                                    required=False,
                                    doc=(
                                        "Mask for the first speaker, "
                                        "required if using two audio inputs."
                                    ),
                                ),
                                InputSpec(
                                    "mask_2",
                                    TypeExpr.concrete(MASK_TYPE),
                                    required=False,
                                    doc=(
                                        "Mask for the second speaker, "
                                        "required if using two audio inputs."
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("model_patch", TypeExpr.concrete(MODEL_PATCH_TYPE)),
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec(
                    "width", _INT, default=832, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "height", _INT, default=480, widget=NumberWidget(min=16, max=16384, step=16)
                ),
                InputSpec(
                    "length", _INT, default=81, widget=NumberWidget(min=1, max=16384, step=4)
                ),
                InputSpec(
                    "clip_vision_output", TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE), required=False
                ),
                InputSpec("start_image", _IMAGE, required=False),
                InputSpec("audio_encoder_output_1", _AUDIO),
                InputSpec(
                    "motion_frame_count",
                    _INT,
                    default=9,
                    widget=NumberWidget(min=1, max=33, step=1),
                    doc="Number of previous frames to use as motion context.",
                    advanced=True,
                ),
                InputSpec(
                    "audio_scale",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
                InputSpec("previous_frames", _IMAGE, required=False),
            ),
            input_order=(
                "mode",
                "model",
                "model_patch",
                "positive",
                "negative",
                "vae",
                "width",
                "height",
                "length",
                "clip_vision_output",
                "start_image",
                "audio_encoder_output_1",
                "motion_frame_count",
                "audio_scale",
                "previous_frames",
            ),
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
                OutputSpec("trim_image", _INT),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        mode: str | Mapping[str, Any],
        model: Any,
        model_patch: Any,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        width: int,
        height: int,
        length: int,
        audio_encoder_output_1: object,
        motion_frame_count: int,
        start_image: object = None,
        previous_frames: object = None,
        audio_scale: float | None = None,
        clip_vision_output: object = None,
        audio_encoder_output_2: object = None,
        mask_1: object = None,
        mask_2: object = None,
        **values: Any,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        utils = importlib.import_module("dinkster_inference.utils")
        runtime = importlib.import_module("dinkster_inference.ldm.wan.model_multitalk")
        extension = importlib.import_module("dinkster_inference.patcher_extension")
        previous = materialize_image(previous_frames) if previous_frames is not None else None
        if previous is not None and previous.shape[0] < motion_frame_count:
            raise ValueError("Not enough previous frames provided.")
        selected = (
            mode
            if isinstance(mode, Mapping)
            else {
                "mode": mode,
                **{key.removeprefix("mode."): value for key, value in values.items()},
            }
        )
        if selected["mode"] == "two_speakers":
            audio_encoder_output_2 = selected["audio_encoder_output_2"]
            mask_1, mask_2 = selected["mask_1"], selected["mask_2"]
        if audio_encoder_output_2 is not None and (mask_1 is None or mask_2 is None):
            raise ValueError("Masks must be provided if two audio encoder outputs are used.")
        ref_masks = None
        if mask_1 is not None and mask_2 is not None:
            if audio_encoder_output_2 is None:
                raise ValueError(
                    "Second audio encoder output must be provided if two masks are used."
                )
            ref_masks = torch.cat([materialize_image(mask_1), materialize_image(mask_2)])

        latent = torch.zeros(
            [1, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=management.intermediate_device(),
        )
        concat_latent_image: Any = None
        if start_image is not None:
            image = utils.common_upscale(
                materialize_image(start_image)[:length].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)
            full = (
                torch.ones(
                    (length, height, width, image.shape[-1]),
                    device=image.device,
                    dtype=image.dtype,
                )
                * 0.5
            )
            full[: image.shape[0]] = image
            concat_latent_image = vae.encode(full[:, :, :, :3])
            mask = torch.ones(
                (
                    1,
                    1,
                    latent.shape[2],
                    concat_latent_image.shape[-2],
                    concat_latent_image.shape[-1],
                ),
                device=image.device,
                dtype=image.dtype,
            )
            mask[:, :, : ((image.shape[0] - 1) // 4) + 1] = 0.0
            conditioning = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            positive = conditioning_set_values(positive, conditioning)
            negative = conditioning_set_values(negative, conditioning)
        if clip_vision_output is not None:
            conditioning = {
                "clip_vision_output": materialize_clip_vision_output(clip_vision_output)
            }
            positive = conditioning_set_values(positive, conditioning)
            negative = conditioning_set_values(negative, conditioning)

        patched = model.clone()
        encoded: list[Any] = []
        for audio in (audio_encoder_output_1, audio_encoder_output_2):
            if audio is None:
                continue
            layers = materialize_latent(audio)["encoded_audio_all_layers"]
            features = torch.stack(layers, dim=0).squeeze(1)[1:]
            features = torch.nn.functional.interpolate(
                features.transpose(1, 2),
                size=int(features.shape[1] / 50.0 * 25),
                align_corners=True,
                mode="linear",
            ).transpose(1, 2)
            encoded.append(features.movedim(0, 1))
        if len(encoded) > 1:
            total = sum(features.shape[0] for features in encoded)
            offset = 0
            padded: list[Any] = []
            for features in encoded:
                full_audio = torch.zeros(total, *features.shape[1:], dtype=features.dtype)
                full_audio[offset : offset + features.shape[0]] = features
                padded.append(full_audio)
                offset += features.shape[0]
            encoded = padded
        targets = None
        if ref_masks is not None:
            targets = torch.nn.functional.interpolate(
                ref_masks.unsqueeze(0),
                size=(latent.shape[-2] // 2, latent.shape[-1] // 2),
                mode="nearest",
            )[0]
            targets = (targets > 0).view(targets.shape[0], -1)
        if previous is not None:
            motion = utils.common_upscale(
                previous[-motion_frame_count:].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)
            audio_start = previous.shape[0] - motion_frame_count
            motion_latent = vae.encode(motion[:, :, :, :3])
            trim_image = motion_frame_count
        else:
            audio_start = trim_image = 0
            if concat_latent_image is None:
                raise UnboundLocalError(
                    "cannot access local variable 'concat_latent_image' "
                    "where it is not associated with a value"
                )
            motion_latent = concat_latent_image[:, :, :1]
        embedding = runtime.project_audio_features(
            model_patch.model.audio_proj,
            encoded,
            audio_start,
            audio_start + length,
        ).to(patched.model_dtype())
        patched.model_options["transformer_options"]["audio_embeds"] = embedding
        patched.add_wrapper_with_key(
            extension.WrappersMP.OUTER_SAMPLE,
            "infinite_talk_outer_sample",
            runtime.InfiniteTalkOuterSampleWrapper(
                motion_latent,
                model_patch,
                is_extend=previous is not None,
            ),
        )
        patched.set_model_patch(
            runtime.MultiTalkCrossAttnPatch(model_patch, audio_scale), "attn2_patch"
        )
        if targets is not None:
            patched.set_model_patch(runtime.MultiTalkGetAttnMapPatch(targets), "attn1_patch")
        return cls.outputs(
            model=patched,
            positive=positive,
            negative=negative,
            latent={"samples": latent},
            trim_image=trim_image,
        )
