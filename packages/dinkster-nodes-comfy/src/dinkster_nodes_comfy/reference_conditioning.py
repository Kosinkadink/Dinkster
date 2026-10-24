"""Image-edit and reference-latent conditioning nodes."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_FLOAT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    DynamicComboOption,
    DynamicComboSpec,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
    materialize_latent,
)

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)


def _combo_values(
    combo_id: str, selected: str | Mapping[str, Any], values: Mapping[str, Any]
) -> Mapping[str, Any]:
    if isinstance(selected, Mapping):
        return selected
    prefix = f"{combo_id}."
    return {
        combo_id: selected,
        **{input_id.removeprefix(prefix): value for input_id, value in values.items()},
    }


class InstructPixToPixConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="InstructPixToPixConditioning",
            category="model/conditioning/instructpix2pix",
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec("pixels", TypeExpr.concrete(IMAGE_TYPE)),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("latent", _LATENT),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        negative: Sequence[Sequence[Any]],
        vae: Any,
        pixels: object,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        image: Any = materialize_image(pixels)
        height = (image.shape[1] // 8) * 8
        width = (image.shape[2] // 8) * 8
        if image.shape[1] != height or image.shape[2] != width:
            top = (image.shape[1] % 8) // 2
            left = (image.shape[2] % 8) // 2
            image = image[:, top : height + top, left : width + left, :]
        encoded = vae.encode(image)
        return cls.outputs(
            positive=[
                [text, {**metadata, "concat_latent_image": encoded}] for text, metadata in positive
            ],
            negative=[
                [text, {**metadata, "concat_latent_image": encoded}] for text, metadata in negative
            ],
            latent={"samples": torch.zeros_like(encoded)},
        )


class ReferenceLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ReferenceLatent",
            display_name="Set Reference Latent",
            category="model/conditioning",
            description=(
                "This node sets the guiding latent for an edit model. If the model supports it "
                "you can chain multiple to set multiple reference images."
            ),
            inputs=(
                InputSpec("conditioning", _CONDITIONING),
                InputSpec("latent", _LATENT, required=False),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls, *, conditioning: Sequence[Sequence[Any]], latent: object = None
    ) -> Mapping[str, object]:
        if latent is None:
            return cls.outputs(CONDITIONING=conditioning)
        samples = materialize_image(materialize_latent(latent)["samples"])
        result: list[list[Any]] = []
        for text, metadata in conditioning:
            references = metadata.get("reference_latents")
            values = [samples] if references is None else references + [samples]
            result.append([text, {**metadata, "reference_latents": values}])
        return cls.outputs(CONDITIONING=result)


class ReferenceMultiLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ReferenceMultiLatent",
            display_name="Set Reference Multi Latent",
            category="model/conditioning",
            description=(
                "Encodes reference images and appends their latents to positive and optional "
                "negative conditioning for edit models."
            ),
            inputs=(
                InputSpec("positive", _CONDITIONING),
                InputSpec("negative", _CONDITIONING, required=False),
                InputSpec(
                    "vae",
                    TypeExpr.concrete(VAE_TYPE),
                    required=False,
                    doc="Reference images are ignored when no VAE is connected.",
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "resolution",
                    (
                        DynamicComboOption("keep resolution"),
                        DynamicComboOption(
                            "resize",
                            (
                                InputSpec(
                                    "megapixels",
                                    _FLOAT,
                                    default=1.0,
                                    widget=NumberWidget(min=0.01, max=16.0, step=0.01),
                                    doc=(
                                        "Target size of each reference image, preserving aspect "
                                        "ratio. "
                                        "1.0 is about 1024 x 1024 pixels."
                                    ),
                                ),
                            ),
                        ),
                    ),
                    default="keep resolution",
                ),
            ),
            input_families=(
                InputFamilySpec(
                    "images",
                    (InputSpec("image", _IMAGE, required=False),),
                    min_members=1,
                    max_members=100,
                    required=False,
                    member_prefix="image",
                ),
            ),
            outputs=(
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
                OutputSpec("empty_latent", _LATENT),
            ),
            input_order=("positive", "negative", "vae", "resolution", "images"),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Sequence[Sequence[Any]],
        resolution: str | Mapping[str, Any],
        negative: Sequence[Sequence[Any]] | None = None,
        vae: Any = None,
        images: Mapping[str, object] | None = None,
        **values: Any,
    ) -> Mapping[str, object]:
        resolution = _combo_values("resolution", resolution, values)
        references: list[Any] = []
        if vae is not None:
            initialize_inference_process()
            for _, pixels in sorted(
                (images or {}).items(), key=lambda item: int(item[0].removeprefix("image"))
            ):
                if pixels is None:
                    continue
                image = materialize_image(pixels)
                if resolution["resolution"] == "resize":
                    height, width = image.shape[1:3]
                    scale = math.sqrt(resolution["megapixels"] * 1024 * 1024 / (width * height))
                    image = (
                        importlib.import_module("dinkster_inference.utils")
                        .common_upscale(
                            image.movedim(-1, 1),
                            max(1, round(width * scale)),
                            max(1, round(height * scale)),
                            "lanczos",
                            "disabled",
                        )
                        .movedim(1, -1)
                    )
                references.append(materialize_image(vae.encode(image)))

        if references:
            empty_latent: Mapping[str, object] = {"samples": references[0] * 0.0}
        else:
            from .latent import EmptyLatentImage

            size = 1024
            if resolution["resolution"] == "resize":
                size = round(math.sqrt(resolution["megapixels"] * 1024 * 1024))
            empty_latent = materialize_latent(
                EmptyLatentImage.execute(width=size, height=size, batch_size=1)["LATENT"]
            )

        def append_references(conditioning: Sequence[Sequence[Any]]) -> list[list[Any]]:
            result: list[list[Any]] = []
            for text, metadata in conditioning:
                result.append(
                    [
                        text,
                        {
                            **metadata,
                            "reference_latents": [
                                *(metadata.get("reference_latents") or ()),
                                *references,
                            ],
                        },
                    ]
                )
            return result

        result_positive = append_references(positive) if references else positive
        result_negative = (
            append_references(negative)
            if references and negative is not None
            else negative
            if negative is not None
            else result_positive
        )
        return cls.outputs(
            positive=result_positive,
            negative=result_negative,
            empty_latent=empty_latent,
        )


REFERENCE_CONDITIONING_NODES = (InstructPixToPixConditioning, ReferenceLatent, ReferenceMultiLatent)
