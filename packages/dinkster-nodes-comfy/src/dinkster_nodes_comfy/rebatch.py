"""ComfyUI image and latent rebatching nodes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_IMAGES = TypeExpr.list_of(_IMAGE)
_LATENTS = TypeExpr.list_of(_LATENT)
_INT = TypeExpr.concrete(CORE_INT)


def _batch_size() -> InputSpec:
    return InputSpec("batch_size", _INT, default=1, widget=NumberWidget(min=1, max=4096))


class RebatchLatents(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="RebatchLatents",
            display_name="Rebatch Latents",
            category="model/latent/batch",
            inputs=(InputSpec("latents", _LATENTS), _batch_size()),
            outputs=(OutputSpec("LATENT", _LATENTS),),
        )

    @staticmethod
    def _batch(
        latents: list[dict[str, Any]], index: int, offset: int
    ) -> tuple[Any, Any, list[int]]:
        samples = latents[index]["samples"]
        shape = samples.shape
        torch = importlib.import_module("torch")
        mask = (
            latents[index]["noise_mask"]
            if "noise_mask" in latents[index]
            else torch.ones((shape[0], 1, shape[2] * 8, shape[3] * 8), device="cpu")
        )
        if mask.shape[-1] != shape[-1] * 8 or mask.shape[-2] != shape[-2]:
            torch.nn.functional.interpolate(
                mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                size=(shape[-2] * 8, shape[-1] * 8),
                mode="bilinear",
            )
        if mask.shape[0] < samples.shape[0]:
            mask = mask.repeat((shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[: shape[0]]
        batch_indices = latents[index].get(
            "batch_index", [value + offset for value in range(shape[0])]
        )
        return samples, mask, batch_indices

    @staticmethod
    def _slices(indexable: Any, count: int, batch_size: int) -> tuple[list[Any], Any | None]:
        slices = [
            indexable[index * batch_size : (index + 1) * batch_size] for index in range(count)
        ]
        return slices, indexable[count * batch_size :] if count * batch_size < len(
            indexable
        ) else None

    @classmethod
    def execute(cls, *, latents: Any, batch_size: int) -> Mapping[str, object]:
        size = batch_size
        torch = importlib.import_module("torch")
        latents = [materialize_latent(latent) for latent in latents]
        output_list: list[dict[str, Any]] = []
        current: tuple[Any, ...] = (None, None, None)
        processed = 0
        for index in range(len(latents)):
            next_batch = cls._batch(latents, index, processed)
            processed += len(next_batch[2])
            if current[0] is None:
                current = next_batch
            elif (
                next_batch[0].shape[-1] != current[0].shape[-1]
                or next_batch[0].shape[-2] != current[0].shape[-2]
            ):
                samples, _ = cls._slices(current[0], 1, size)
                masks, _ = cls._slices(current[1], 1, size)
                indices, _ = cls._slices(current[2], 1, size)
                output_list.append(
                    {"samples": samples[0], "noise_mask": masks[0], "batch_index": indices[0]}
                )
                current = next_batch
            else:
                current = tuple(
                    torch.cat((left, right)) if torch.is_tensor(left) else left + right
                    for left, right in zip(current, next_batch, strict=True)
                )
            if current[0].shape[0] > size:
                count = current[0].shape[0] // size
                sample_slices, sample_remainder = cls._slices(current[0], count, size)
                mask_slices, mask_remainder = cls._slices(current[1], count, size)
                index_slices, index_remainder = cls._slices(current[2], count, size)
                output_list.extend(
                    {
                        "samples": sample_slices[item],
                        "noise_mask": mask_slices[item],
                        "batch_index": index_slices[item],
                    }
                    for item in range(count)
                )
                current = (sample_remainder, mask_remainder, index_remainder)
        if current[0] is not None:
            samples, _ = cls._slices(current[0], 1, size)
            masks, _ = cls._slices(current[1], 1, size)
            indices, _ = cls._slices(current[2], 1, size)
            output_list.append(
                {"samples": samples[0], "noise_mask": masks[0], "batch_index": indices[0]}
            )
        for latent in output_list:
            if latent["noise_mask"].mean() == 1.0:
                del latent["noise_mask"]
        return cls.outputs(LATENT=output_list)


class RebatchImages(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="RebatchImages",
            display_name="Rebatch Images",
            category="image/batch",
            inputs=(InputSpec("images", _IMAGES), _batch_size()),
            outputs=(OutputSpec("IMAGE", _IMAGES),),
        )

    @classmethod
    def execute(cls, *, images: Any, batch_size: int) -> Mapping[str, object]:
        size = batch_size
        torch = importlib.import_module("torch")
        frames = [
            image[index : index + 1]
            for value in images
            for image in [materialize_image(value)]
            for index in range(image.shape[0])
        ]
        return cls.outputs(
            IMAGE=[
                torch.cat(frames[index : index + size], dim=0)
                for index in range(0, len(frames), size)
            ]
        )


REBATCH_NODES = (RebatchImages, RebatchLatents)
