"""Qwen Image edit conditioning nodes."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    ComboOption,
    ComboWidget,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)

from .conditioning import conditioning_set_values

_CLIP = TypeExpr.concrete(CLIP_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)
_VAE = TypeExpr.concrete(VAE_TYPE)


def _text(name: str) -> InputSpec:
    return InputSpec(name, _STRING, widget=StringWidget(multiline=True, dynamic_prompts=True))


def _upscale(image: Any, width: int, height: int, method: str) -> Any:
    initialize_inference_process()
    return (
        importlib.import_module("dinkster_inference.utils")
        .common_upscale(image.movedim(-1, 1), width, height, method, "disabled")
        .movedim(1, -1)
    )


def _scale(image: Any, pixels: int, multiple: int = 1) -> tuple[int, int]:
    height, width = image.shape[1:3]
    scale = math.sqrt(pixels / (width * height))
    return round(width * scale / multiple) * multiple, round(height * scale / multiple) * multiple


def _append(conditioning: Any, references: list[Any]) -> Any:
    return conditioning_set_values(
        conditioning,
        {"reference_latents": references},
        append=True,
    )


class TextEncodeQwenImageEdit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextEncodeQwenImageEdit",
            display_name="Text Encode Qwen Image Edit",
            category="model/conditioning/qwen image",
            inputs=(
                InputSpec("clip", _CLIP),
                _text("prompt"),
                InputSpec("vae", _VAE, required=False),
                InputSpec("image", _IMAGE, required=False),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls, *, clip: Any, prompt: str, vae: Any = None, image: object = None
    ) -> Mapping[str, object]:
        images: list[Any] = []
        references: list[Any] = []
        if image is not None:
            pixels: Any = materialize_image(image)
            width, height = _scale(pixels, 1_048_576)
            pixels = _upscale(pixels, width, height, "area")
            images = [pixels[..., :3]]
            if vae is not None:
                references.append(vae.encode(pixels[..., :3]))
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=images))
        return cls.outputs(
            CONDITIONING=_append(conditioning, references) if references else conditioning
        )


class TextEncodeQwenImageEditPlus(Node):
    llama_template = (
        "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
        "texture, objects, background), then explain how the user's text instruction should alter "
        "or modify the image. Generate a new image that meets the user's requirements while "
        "maintaining consistency with the original input where appropriate.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextEncodeQwenImageEditPlus",
            display_name="Text Encode Qwen Image Edit Plus",
            category="model/conditioning/qwen image",
            inputs=(
                InputSpec("clip", _CLIP),
                _text("prompt"),
                InputSpec("vae", _VAE, required=False),
                *(InputSpec(f"image{i}", _IMAGE, required=False) for i in range(1, 4)),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        prompt: str,
        vae: Any = None,
        image1: object = None,
        image2: object = None,
        image3: object = None,
    ) -> Mapping[str, object]:
        vision: list[Any] = []
        references: list[Any] = []
        prefix = ""
        for index, raw_image in enumerate((image1, image2, image3), 1):
            if raw_image is None:
                continue
            image = materialize_image(raw_image)
            width, height = _scale(image, 384 * 384)
            vision.append(_upscale(image, width, height, "area"))
            if vae is not None:
                width, height = _scale(image, 1_048_576, 8)
                references.append(vae.encode(_upscale(image, width, height, "area")[..., :3]))
            prefix += f"Picture {index}: <|vision_start|><|image_pad|><|vision_end|>"
        conditioning = clip.encode_from_tokens_scheduled(
            clip.tokenize(prefix + prompt, images=vision, llama_template=cls.llama_template)
        )
        return cls.outputs(
            CONDITIONING=_append(conditioning, references) if references else conditioning
        )


class TextEncodeQwenImage21(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextEncodeQwenImage21",
            display_name="Text Encode Qwen Image 2.1",
            category="model/conditioning/qwen image",
            inputs=(
                InputSpec("clip", _CLIP),
                _text("prompt"),
                _text("negative_prompt"),
                InputSpec("vae", _VAE, required=False),
                InputSpec(
                    "resolution", _INT, default=1024, widget=NumberWidget(min=0, max=4096, step=32)
                ),
            ),
            input_families=(
                InputFamilySpec(
                    "images",
                    (InputSpec("image", _IMAGE),),
                    min_members=0,
                    member_names=tuple(f"image_{index}" for index in range(1, 17)),
                ),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("latent", _LATENT),
            ),
            input_order=("clip", "prompt", "negative_prompt", "vae", "resolution", "images"),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        prompt: str,
        negative_prompt: str,
        vae: Any = None,
        resolution: int = 1024,
        images: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        vision: list[Any] = []
        references: list[Any] = []
        latent_width = latent_height = resolution or 1024
        for _, raw_image in sorted(
            (images or {}).items(), key=lambda item: int(item[0].rsplit("_", 1)[-1])
        ):
            if raw_image is None:
                continue
            image = materialize_image(raw_image)[:1]
            height, width = image.shape[1:3]
            if resolution:
                output_width = round(math.sqrt(resolution * resolution * width / height) / 32) * 32
                output_height = round(math.sqrt(resolution * resolution * height / width) / 32) * 32
            else:
                output_width, output_height = round(width / 32) * 32, round(height / 32) * 32
            output_width, output_height = max(32, output_width), max(32, output_height)
            scaled = (
                image
                if (output_width, output_height) == (width, height)
                else _upscale(image, output_width, output_height, "lanczos")
            )
            if not vision:
                latent_width, latent_height = output_width, output_height
            rgb = scaled[..., :3]
            if scaled.shape[-1] > 3:
                rgb = rgb * scaled[..., 3:] + (1.0 - scaled[..., 3:])
            vision.append(rgb)
            if vae is not None:
                references.append(vae.encode(scaled))
        keep_vision = not references

        def encode(text: str) -> Any:
            return clip.encode_from_tokens_scheduled(
                clip.tokenize(text, images=vision, keep_vision=keep_vision, prevent_empty_text=True)
            )

        torch = importlib.import_module("torch")
        positive, negative = encode(prompt), encode(negative_prompt)
        if references:
            positive, negative = _append(positive, references), _append(negative, references)
        return cls.outputs(
            positive=positive,
            negative=negative,
            latent={"samples": torch.zeros((1, 64, latent_height // 16, latent_width // 16))},
        )


class QwenImage21Cache(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        combo = TypeExpr.concrete(CORE_COMBO)
        return NodeSchema(
            node_type="QwenImage21Cache",
            display_name="Qwen Image 2.1 Cache",
            category="model/conditioning/qwen image",
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec(
                    "device",
                    combo,
                    default="auto",
                    widget=ComboWidget(
                        tuple(ComboOption(x) for x in ("auto", "gpu", "cpu", "off"))
                    ),
                ),
                InputSpec(
                    "dtype",
                    combo,
                    default="default",
                    widget=ComboWidget(tuple(ComboOption(x) for x in ("default", "int8", "int4"))),
                ),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(cls, *, model: Any, device: str, dtype: str) -> Mapping[str, object]:
        patched = model.clone()
        patched.model_options["transformer_options"]["qwen_image21_cache"] = {
            "device": device,
            "dtype": dtype,
        }
        return cls.outputs(MODEL=patched)
