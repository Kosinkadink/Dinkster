"""MOSS image layout prompts and reference-image text encoding."""

from __future__ import annotations

import importlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    ARRAY_TYPE,
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    VAE_TYPE,
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

from .conditioning import conditioning_set_values, conditioning_set_values_with_timestep_range

_STRING = TypeExpr.concrete(CORE_STRING)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_EDIT_FORMATS = {
    "ai-toolkit": ("index_timestep_zero_isolated", True),
    "ai-toolkit (kv_cache off)": ("index_timestep_zero", True),
    "Moss-Image-Edit": ("index_timestep_zero_text_start", False),
}


class MOSSImageLayoutPrompt(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MOSSImageLayoutPrompt",
            display_name="MOSS-Image Layout Prompt",
            category="model/conditioning/moss_image",
            description=(
                "Builds a MOSS-Image layout prompt from the prompt and the boxes from "
                "Create Bounding Boxes."
            ),
            inputs=(
                InputSpec(
                    "prompt", _STRING, widget=StringWidget(multiline=True, dynamic_prompts=True)
                ),
                InputSpec("element", TypeExpr.concrete(ARRAY_TYPE)),
                InputSpec(
                    "width",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=16, max=16384, step=16),
                ),
                InputSpec(
                    "height",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=16, max=16384, step=16),
                ),
                InputSpec(
                    "character_boxes", _STRING, default="", widget=StringWidget(multiline=False)
                ),
            ),
            outputs=(OutputSpec("prompt", _STRING, display_name="prompt"),),
        )

    @classmethod
    def execute(
        cls, *, prompt: str, element: Any, width: int, height: int, character_boxes: str = ""
    ) -> Mapping[str, object]:
        characters = {int(index) for index in re.findall(r"\d+", character_boxes)}
        items: list[dict[str, Any]] = []
        elements = cast("list[dict[str, Any]]", element) if isinstance(element, list) else []
        for index, el in enumerate(elements):
            label = (
                "text"
                if el.get("type") == "text"
                else ("character" if index + 1 in characters else "object")
            )
            item: dict[str, Any] = {"label": label, "caption": el.get("desc", "")}
            if label == "text":
                item["text"] = el.get("text", "")
            if el.get("color_palette"):
                item["palette"] = el["color_palette"]
            y0, x0, y1, x1 = el["bbox"]
            item["bbox"] = [
                math.floor(round(y0 * height / 1000, 9)) - height // 2,
                math.floor(round(x0 * width / 1000, 9)) - width // 2,
                math.ceil(round(y1 * height / 1000, 9)) - height // 2,
                math.ceil(round(x1 * width / 1000, 9)) - width // 2,
            ]
            items.append(item)
        if not items:
            return cls.outputs(prompt=prompt)
        layout = {
            "summary": {"caption": prompt.strip()},
            "items": items,
            "bbox_coord_space": "abs_centralized",
            "bbox_format": "yxyx",
        }
        return cls.outputs(prompt=json.dumps(layout, ensure_ascii=False, separators=(",", ":")))


class MOSSImageControl(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MOSSImageControl",
            display_name="MOSS-Image Color/Depth Control",
            category="model/conditioning/moss_image",
            description=(
                "Adds the color region and depth maps for a MOSS-Image color/depth control "
                "adapter loaded with a LoRA loader."
            ),
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec(
                    "color_map",
                    TypeExpr.concrete(IMAGE_TYPE),
                    required=False,
                    doc=(
                        "One flat color per region, at the output size. "
                        "The colors are only region IDs."
                    ),
                ),
                InputSpec(
                    "depth_map",
                    TypeExpr.concrete(IMAGE_TYPE),
                    required=False,
                    doc="Depth map at the output size, near is white.",
                ),
                InputSpec(
                    "strength",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                    doc=(
                        "Scales the maps' contribution to the image tokens. 0.0 is the same "
                        "as no maps, 1.0 is the trained strength."
                    ),
                ),
                InputSpec(
                    "start_percent",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                    doc="Sampling percent at which the control starts.",
                ),
                InputSpec(
                    "end_percent",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                    doc=(
                        "Sampling percent at which the control ends. Ending early keeps the "
                        "layout and depth while the rest of the steps loosen the shapes."
                    ),
                ),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING, display_name="positive"),
                OutputSpec("negative", _CONDITIONING, display_name="negative"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Any,
        negative: Any,
        vae: Any,
        color_map: object = None,
        depth_map: object = None,
        strength: float = 1.0,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
    ) -> Mapping[str, object]:
        latents = [
            vae.encode(materialize_image(image)[..., :3]) if image is not None else None
            for image in (color_map, depth_map)
        ]
        reference = next((latent for latent in latents if latent is not None), None)
        if reference is None:
            return cls.outputs(positive=positive, negative=negative)
        initialize_inference_process()
        torch = importlib.import_module("torch")
        utils = importlib.import_module("dinkster_inference.utils")
        batch = max(latent.shape[0] for latent in latents if latent is not None)
        aligned = [
            torch.zeros_like(reference)
            if latent is None
            else latent
            if latent.shape[-2:] == reference.shape[-2:]
            else utils.common_upscale(
                latent, reference.shape[-1], reference.shape[-2], "bilinear", "center"
            )
            for latent in latents
        ]
        control = (
            torch.cat([utils.resize_to_batch_size(latent, batch) for latent in aligned], dim=1)
            * strength
        )
        values = {"concat_latent_image": control}
        return cls.outputs(
            positive=conditioning_set_values_with_timestep_range(
                positive, values, start_percent, end_percent
            ),
            negative=conditioning_set_values_with_timestep_range(
                negative, values, start_percent, end_percent
            ),
        )


class TextEncodeMOSSImageEdit(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextEncodeMOSSImageEdit",
            display_name="Text Encode MOSS-Image Edit",
            category="model/conditioning/moss_image",
            description=(
                "Encodes an edit instruction with reference images for a MOSS-Image edit LoRA, "
                "use it for the negative prompt too. The references are used at the size they "
                "come in, scale them first."
            ),
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec(
                    "prompt", _STRING, widget=StringWidget(multiline=True, dynamic_prompts=True)
                ),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec(
                    "edit_format",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=ComboWidget(tuple(_EDIT_FORMATS)),
                ),
            ),
            input_families=(
                InputFamilySpec(
                    "images",
                    (InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
                    member_prefix="image_",
                    min_members=1,
                    max_members=10,
                    optional_after_minimum=True,
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        prompt: str,
        vae: Any,
        edit_format: str,
        images: Mapping[str, object],
    ) -> Mapping[str, object]:
        method, vision = _EDIT_FORMATS[edit_format]
        pixels = [
            materialize_image(images[name]) for name in sorted(images) if images[name] is not None
        ]
        vl: list[Any] = []
        if vision:
            initialize_inference_process()
            utils = importlib.import_module("dinkster_inference.utils")
            for image in pixels:
                scale = min(1.0, math.sqrt(384 * 384 / (image.shape[1] * image.shape[2])))
                vl.append(
                    utils.common_upscale(
                        image.movedim(-1, 1),
                        round(image.shape[2] * scale),
                        round(image.shape[1] * scale),
                        "area",
                        "disabled",
                    ).movedim(1, -1)
                )
            prompt = (
                "".join(
                    f"Picture {index + 1}: <|vision_start|><|image_pad|><|vision_end|>"
                    for index in range(len(vl))
                )
                + prompt
            )
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=vl))
        refs = [vae.encode(image[:, :, :, :3]) for image in pixels]
        return cls.outputs(
            CONDITIONING=conditioning_set_values(
                conditioning, {"reference_latents": refs, "reference_latents_method": method}
            )
        )
