"""Image-list merging and resolution-based training batches."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    IMAGE_TYPE,
    LATENT_TYPE,
    Deprecation,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
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


DATASET_GROUP_NODES = (MergeImageLists, ResolutionBucket)
