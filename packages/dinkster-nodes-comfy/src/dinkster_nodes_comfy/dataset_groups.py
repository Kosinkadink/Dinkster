"""Image-list merging and resolution-based training batches."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    Deprecation,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
    materialize_image,
    materialize_latent,
)

_IMAGES = TypeExpr.list_of(TypeExpr.concrete(IMAGE_TYPE))
_LATENTS = TypeExpr.list_of(TypeExpr.concrete(LATENT_TYPE))
_CONDITIONS = TypeExpr.list_of(TypeExpr.concrete(CONDITIONING_TYPE))


class MergeImageLists(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MergeImageLists",
            display_name="Merge Image Lists (DEPRECATED)",
            category="image/batch",
            description="Concatenate multiple image lists into one.",
            search_terms=("list", "merge list", "make list"),
            search_visibility="deprecated",
            deprecation=Deprecation("Deprecated in ComfyUI."),
            inputs=(InputSpec("images", _IMAGES, doc="List of images to process."),),
            outputs=(OutputSpec("images", _IMAGES, doc="Processed images"),),
        )

    @classmethod
    def execute(cls, *, images: Any) -> Mapping[str, object]:
        items = cast(list[Any], images) if isinstance(images, list) else [images]
        frames: list[Any] = []
        for item in items:
            image = materialize_image(item)
            if image.ndim != 4:
                if isinstance(images, list):
                    raise ValueError(
                        f"Expected 4D image tensor, got {type(image).__name__} shape {image.shape}"
                    )
                raise ValueError(f"Expected 4D image tensor, got shape {tuple(image.shape)}")
            frames.extend(image[i : i + 1] for i in range(image.shape[0]))
        return cls.outputs(images=frames)


class ResolutionBucket(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ResolutionBucket",
            display_name="Resolution Bucket",
            category="model/training",
            description="Group latents and conditionings into buckets",
            search_terms=("bucket by resolution", "group by resolution", "batch by resolution"),
            inputs=(
                InputSpec("latents", _LATENTS, doc="List of latent dicts to bucket by resolution."),
                InputSpec(
                    "conditioning",
                    _CONDITIONS,
                    doc="List of conditioning lists (must match latents length).",
                ),
            ),
            outputs=(
                OutputSpec(
                    "latents",
                    _LATENTS,
                    doc="List of batched latent dicts, one per resolution bucket.",
                ),
                OutputSpec(
                    "conditioning",
                    _CONDITIONS,
                    doc="List of condition lists, one per resolution bucket.",
                ),
            ),
        )

    @classmethod
    def execute(cls, *, latents: list[Any], conditioning: list[Any]) -> Mapping[str, object]:
        if len(latents) != len(conditioning):
            raise ValueError(
                f"Number of latents ({len(latents)}) does not match "
                f"number of conditions ({len(conditioning)})."
            )
        torch = importlib.import_module("torch")
        buckets: dict[tuple[int, int], tuple[list[Any], list[Any]]] = {}
        for latent, conditions in zip(latents, conditioning, strict=False):
            samples = materialize_latent(latent)["samples"]
            for index in range(samples.shape[0]):
                sample = samples[index]
                key = (sample.shape[-2], sample.shape[-1])
                bucket_samples, bucket_conditions = buckets.setdefault(key, ([], []))
                bucket_samples.append(sample)
                bucket_conditions.append(conditions[index])
        return cls.outputs(
            latents=[{"samples": torch.stack(samples, dim=0)} for samples, _ in buckets.values()],
            conditioning=[conditions for _, conditions in buckets.values()],
        )


class MakeTrainingDataset(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="MakeTrainingDataset",
            display_name="Make Training Dataset",
            category="model/training",
            description=(
                "Encode images with VAE and texts with CLIP to create a training dataset "
                "of latents and conditionings."
            ),
            search_terms=("encode dataset",),
            inputs=(
                InputSpec("images", _IMAGES, doc="List of images to encode."),
                InputSpec(
                    "vae",
                    TypeExpr.concrete(VAE_TYPE),
                    doc="VAE model for encoding images to latents.",
                ),
                InputSpec(
                    "clip",
                    TypeExpr.concrete(CLIP_TYPE),
                    doc="CLIP model for encoding text to conditioning.",
                ),
                InputSpec(
                    "texts",
                    TypeExpr.list_of(TypeExpr.concrete(CORE_STRING)),
                    required=False,
                    force_input=True,
                    widget=StringWidget(multiline=False),
                    doc=(
                        "List of text captions. Can be length n (matching images), "
                        "1 (repeated for all), or omitted (uses empty string)."
                    ),
                ),
            ),
            outputs=(
                OutputSpec("latents", _LATENTS, doc="List of latent dicts"),
                OutputSpec("conditioning", _CONDITIONS, doc="List of conditioning lists"),
            ),
        )

    @classmethod
    def execute(
        cls, *, images: list[Any], vae: Any, clip: Any, texts: list[str] | None = None
    ) -> Mapping[str, object]:
        num_images = len(images)
        if texts is None or len(texts) == 0:
            texts = [""]
        if len(texts) == 1 and num_images > 1:
            texts = texts * num_images
        elif len(texts) != num_images:
            raise ValueError(
                f"Number of texts ({len(texts)}) does not match number of images ({num_images}). "
                f"Text list should have length {num_images}, 1, or 0."
            )
        latents = [
            {"samples": vae.encode(materialize_image(image)[:, :, :, :3])} for image in images
        ]
        conditioning = [clip.encode_from_tokens_scheduled(clip.tokenize(text)) for text in texts]
        return cls.outputs(latents=latents, conditioning=conditioning)


DATASET_GROUP_NODES = (MergeImageLists, ResolutionBucket)
