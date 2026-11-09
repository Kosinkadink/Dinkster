"""Native Wan control conditioning nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MASK_TYPE,
    VAE_TYPE,
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
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_MASK = TypeExpr.concrete(MASK_TYPE)
_VAE = TypeExpr.concrete(VAE_TYPE)


def _integer(
    name: str, default: int, minimum: int, maximum: int, step: int | None = None
) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _common_upscale(image: Any, width: int, height: int) -> Any:
    utils = importlib.import_module("dinkster_inference.utils")
    return utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(
        1, -1
    )


def _append_values(
    conditioning: Sequence[Sequence[Any]], values: Mapping[str, Sequence[object]]
) -> list[list[Any]]:
    result: list[list[Any]] = []
    for text, metadata in conditioning:
        appended = {key: [*(metadata.get(key) or ()), *value] for key, value in values.items()}
        result.append([text, {**metadata, **appended}])
    return result


class WanFunControlToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanFunControlToVideo",
            category="model/conditioning/wan/fun control",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                _integer("width", 832, 16, 16384, 16),
                _integer("height", 480, 16, 16384, 16),
                _integer("length", 81, 1, 16384, 4),
                _integer("batch_size", 1, 1, 4096),
                InputSpec("clip_vision_output", _CLIP_VISION_OUTPUT, required=False),
                InputSpec("start_image", _IMAGE, required=False),
                InputSpec("control_video", _IMAGE, required=False),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec("latent", _LATENT, display_name="latent"),
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
        start_image: object | None = None,
        clip_vision_output: object | None = None,
        control_video: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        latent_formats = importlib.import_module("dinkster_inference.latent_formats")
        management = importlib.import_module("dinkster_inference.model_management")
        latent = torch.zeros(
            (batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8),
            device=management.intermediate_device(),
        )
        concat_latent = (
            latent_formats.Wan21().process_out(torch.zeros_like(latent)).repeat(1, 2, 1, 1, 1)
        )
        if start_image is not None:
            start = _common_upscale(materialize_image(start_image)[:length], width, height)
            encoded = vae.encode(start[:, :, :, :3])
            concat_latent[:, 16:, : encoded.shape[2]] = encoded[:, :, : concat_latent.shape[2]]
        if control_video is not None:
            control = _common_upscale(materialize_image(control_video)[:length], width, height)
            encoded = vae.encode(control[:, :, :, :3])
            concat_latent[:, :16, : encoded.shape[2]] = encoded[:, :, : concat_latent.shape[2]]
        values = {"concat_latent_image": concat_latent}
        positive = conditioning_set_values(positive, values)
        negative = conditioning_set_values(negative, values)
        if clip_vision_output is not None:
            values = {"clip_vision_output": materialize_clip_vision_output(clip_vision_output)}
            positive = conditioning_set_values(positive, values)
            negative = conditioning_set_values(negative, values)
        return cls.outputs(positive=positive, negative=negative, latent={"samples": latent})


class Wan22FunControlToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="Wan22FunControlToVideo",
            category="model/conditioning/wan/fun control",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                _integer("width", 832, 16, 16384, 16),
                _integer("height", 480, 16, 16384, 16),
                _integer("length", 81, 1, 16384, 4),
                _integer("batch_size", 1, 1, 4096),
                InputSpec("ref_image", _IMAGE, required=False),
                InputSpec("control_video", _IMAGE, required=False),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec("latent", _LATENT, display_name="latent"),
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
        ref_image: object | None = None,
        start_image: object | None = None,
        control_video: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        latent_formats = importlib.import_module("dinkster_inference.latent_formats")
        management = importlib.import_module("dinkster_inference.model_management")
        spatial_scale = vae.spacial_compression_encode()
        latent = torch.zeros(
            (
                batch_size,
                vae.latent_channels,
                ((length - 1) // 4) + 1,
                height // spatial_scale,
                width // spatial_scale,
            ),
            device=management.intermediate_device(),
        )
        latent_format = latent_formats.Wan22 if vae.latent_channels == 48 else latent_formats.Wan21
        concat_latent = latent_format().process_out(torch.zeros_like(latent)).repeat(1, 2, 1, 1, 1)
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))
        if start_image is not None:
            start = _common_upscale(materialize_image(start_image)[:length], width, height)
            encoded = vae.encode(start[:, :, :, :3])
            concat_latent[:, vae.latent_channels :, : encoded.shape[2]] = encoded[
                :, :, : concat_latent.shape[2]
            ]
            mask[:, :, : start.shape[0] + 3] = 0.0
        reference: Any = None
        if ref_image is not None:
            reference = vae.encode(
                _common_upscale(materialize_image(ref_image)[:1], width, height)[:, :, :, :3]
            )
        if control_video is not None:
            control = _common_upscale(materialize_image(control_video)[:length], width, height)
            encoded = vae.encode(control[:, :, :, :3])
            concat_latent[:, : vae.latent_channels, : encoded.shape[2]] = encoded[
                :, :, : concat_latent.shape[2]
            ]
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        values = {
            "concat_latent_image": concat_latent,
            "concat_mask": mask,
            "concat_mask_index": vae.latent_channels,
        }
        positive = conditioning_set_values(positive, values)
        negative = conditioning_set_values(negative, values)
        if reference is not None:
            positive = _append_values(positive, {"reference_latents": [reference]})
            negative = _append_values(negative, {"reference_latents": [reference]})
        return cls.outputs(positive=positive, negative=negative, latent={"samples": latent})


class WanVaceToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanVaceToVideo",
            search_terms=("video conditioning", "video control"),
            category="model/conditioning/wan/vace",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                _integer("width", 832, 16, 16384, 16),
                _integer("height", 480, 16, 16384, 16),
                _integer("length", 81, 1, 16384, 4),
                _integer("batch_size", 1, 1, 4096),
                InputSpec(
                    "strength",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1000.0, step=0.01),
                ),
                InputSpec("control_video", _IMAGE, required=False),
                InputSpec("control_masks", _MASK, required=False),
                InputSpec("reference_image", _IMAGE, required=False),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
                OutputSpec("latent", _LATENT, display_name="latent"),
                OutputSpec("trim_latent", _INT, display_name="trim_latent"),
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
        strength: float,
        control_video: object | None = None,
        control_masks: object | None = None,
        reference_image: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
        latent_formats = importlib.import_module("dinkster_inference.latent_formats")
        management = importlib.import_module("dinkster_inference.model_management")
        latent_length = ((length - 1) // 4) + 1
        if control_video is not None:
            control = _common_upscale(materialize_image(control_video)[:length], width, height)
            if control.shape[0] < length:
                control = functional.pad(
                    control, (0, 0, 0, 0, 0, 0, 0, length - control.shape[0]), value=0.5
                )
        else:
            control = torch.ones((length, height, width, 3)) * 0.5
        reference: Any = None
        if reference_image is not None:
            reference = vae.encode(
                _common_upscale(materialize_image(reference_image)[:1], width, height)[:, :, :, :3]
            )
            reference = torch.cat(
                (reference, latent_formats.Wan21().process_out(torch.zeros_like(reference))), dim=1
            )
        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = materialize_image(control_masks)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            utils = importlib.import_module("dinkster_inference.utils")
            mask = utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(
                1, -1
            )
            if mask.shape[0] < length:
                mask = functional.pad(
                    mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0
                )
        control = control - 0.5
        inactive = vae.encode(((control * (1 - mask)) + 0.5)[:, :, :, :3])
        reactive = vae.encode(((control * mask) + 0.5)[:, :, :, :3])
        frames = torch.cat((inactive, reactive), dim=1)
        if reference is not None:
            frames = torch.cat((reference, frames), dim=2)
        vae_stride = 8
        height_mask, width_mask = height // vae_stride, width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3).reshape(
            vae_stride * vae_stride, length, height_mask, width_mask
        )
        mask = functional.interpolate(
            mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode="nearest-exact"
        ).squeeze(0)
        trim_latent = 0
        if reference is not None:
            mask = torch.cat((torch.zeros_like(mask[:, : reference.shape[2], :, :]), mask), dim=1)
            latent_length += reference.shape[2]
            trim_latent = reference.shape[2]
        mask = mask.unsqueeze(0)
        values = {"vace_frames": [frames], "vace_mask": [mask], "vace_strength": [strength]}
        positive = _append_values(positive, values)
        negative = _append_values(negative, values)
        latent = torch.zeros(
            (batch_size, 16, latent_length, height // 8, width // 8),
            device=management.intermediate_device(),
        )
        return cls.outputs(
            positive=positive,
            negative=negative,
            latent={"samples": latent},
            trim_latent=trim_latent,
        )


class WanPhantomSubjectToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="WanPhantomSubjectToVideo",
            category="model/conditioning/wan/phantom subject",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", _VAE),
                _integer("width", 832, 16, 16384, 16),
                _integer("height", 480, 16, 16384, 16),
                _integer("length", 81, 1, 16384, 4),
                _integer("batch_size", 1, 1, 4096),
                InputSpec("images", _IMAGE, required=False),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative_text", _CONDITIONING, display_name="negative_text"),
                OutputSpec("negative_img_text", _CONDITIONING, display_name="negative_img_text"),
                OutputSpec("latent", _LATENT, display_name="latent"),
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
        images: object | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        latent_formats = importlib.import_module("dinkster_inference.latent_formats")
        management = importlib.import_module("dinkster_inference.model_management")
        latent = torch.zeros(
            (batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8),
            device=management.intermediate_device(),
        )
        negative_text = negative
        negative_img_text = negative
        if images is not None:
            image = _common_upscale(materialize_image(images)[:length], width, height)
            encoded = torch.cat(
                [vae.encode(frame.unsqueeze(0)[:, :, :, :3]) for frame in image], dim=2
            )
            positive = conditioning_set_values(positive, {"time_dim_concat": encoded})
            negative_text = conditioning_set_values(negative, {"time_dim_concat": encoded})
            negative_img_text = conditioning_set_values(
                negative,
                {"time_dim_concat": latent_formats.Wan21().process_out(torch.zeros_like(encoded))},
            )
        return cls.outputs(
            positive=positive,
            negative_text=negative_text,
            negative_img_text=negative_img_text,
            latent={"samples": latent},
        )


WAN_CONTROL_CONDITIONING_NODES = (
    WanFunControlToVideo,
    Wan22FunControlToVideo,
    WanVaceToVideo,
    WanPhantomSubjectToVideo,
)
