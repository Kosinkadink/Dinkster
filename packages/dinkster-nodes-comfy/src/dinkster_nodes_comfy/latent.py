"""Latent batch selection, transforms and resizing with core node interfaces."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
    materialize_latent,
)

_LATENT = TypeExpr.concrete("LATENT")
_SAMPLES = InputSpec("samples", _LATENT)
_UPSCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")


def _integer(
    name: str, default: int, minimum: int, maximum: int, step: int | None = None
) -> InputSpec:
    return InputSpec(
        name,
        TypeExpr.concrete("INT"),
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _combo(name: str, options: tuple[str, ...]) -> InputSpec:
    return InputSpec(
        name,
        TypeExpr.concrete("COMBO"),
        widget=ComboWidget(tuple(ComboOption(value) for value in options)),
    )


def _schema(
    node_type: str, display_name: str, inputs: tuple[InputSpec, ...], category: str = "model/latent"
) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category=category,
        inputs=inputs,
        outputs=(OutputSpec("LATENT", _LATENT),),
    )


class EmptyLatentImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "EmptyLatentImage",
            "Empty Latent Image",
            (
                _integer("width", 1024, 16, 16384, 8),
                _integer("height", 1024, 16, 16384, 8),
                _integer("batch_size", 1, 1, 4096),
            ),
        )

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int = 1) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        samples = torch.zeros(
            [batch_size, 4, height // 8, width // 8],
            device=management.intermediate_device(),
            dtype=management.intermediate_dtype(),
        )
        return cls.outputs(LATENT={"samples": samples, "downscale_ratio_spacial": 8})


class LatentFromBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentFromBatch",
            "Get Latent From Batch",
            (_SAMPLES, _integer("batch_index", 0, -16384, 16384), _integer("length", 1, 1, 64)),
            "model/latent/batch",
        )

    @classmethod
    def execute(
        cls, *, samples: dict[str, Any], batch_index: int, length: int
    ) -> Mapping[str, object]:
        samples = materialize_latent(samples)
        result = samples.copy()
        source = samples["samples"]
        if batch_index < 0:
            batch_index += source.shape[0]
        batch_index = max(0, min(source.shape[0] - 1, batch_index))
        length = min(source.shape[0] - batch_index, length)
        result["samples"] = source[batch_index : batch_index + length].clone()
        if "noise_mask" in samples:
            masks = samples["noise_mask"]
            if masks.shape[0] == 1:
                result["noise_mask"] = masks.clone()
            else:
                if masks.shape[0] < source.shape[0]:
                    masks = masks.repeat(math.ceil(source.shape[0] / masks.shape[0]), 1, 1, 1)[
                        : source.shape[0]
                    ]
                result["noise_mask"] = masks[batch_index : batch_index + length].clone()
        if "batch_index" in samples:
            result["batch_index"] = samples["batch_index"][batch_index : batch_index + length]
        else:
            result["batch_index"] = list(range(batch_index, batch_index + length))
        return cls.outputs(LATENT=result)


class RepeatLatentBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "RepeatLatentBatch",
            "Repeat Latent Batch",
            (_SAMPLES, _integer("amount", 1, 1, 64)),
            "model/latent/batch",
        )

    @classmethod
    def execute(cls, *, samples: dict[str, Any], amount: int) -> Mapping[str, object]:
        samples = materialize_latent(samples)
        result = samples.copy()
        source = samples["samples"]
        result["samples"] = source.repeat((amount,) + (1,) * (source.ndim - 1))
        if "noise_mask" in samples and samples["noise_mask"].shape[0] > 1:
            mask = samples["noise_mask"]
            result["noise_mask"] = mask.repeat((amount,) + (1,) * (mask.ndim - 1))
        if "batch_index" in samples:
            indices = samples["batch_index"]
            offset = max(indices) - min(indices) + 1
            result["batch_index"] = indices + [
                value + repetition * offset for repetition in range(1, amount) for value in indices
            ]
        return cls.outputs(LATENT=result)


class LatentUpscale(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentUpscale",
            "Upscale Latent",
            (
                _SAMPLES,
                _combo("upscale_method", _UPSCALE_METHODS),
                _integer("width", 512, 0, 16384, 8),
                _integer("height", 512, 0, 16384, 8),
                _combo("crop", ("disabled", "center")),
            ),
        )

    @classmethod
    def execute(
        cls, *, samples: dict[str, Any], upscale_method: str, width: int, height: int, crop: str
    ) -> Mapping[str, object]:
        if width == 0 and height == 0:
            return cls.outputs(LATENT=samples)
        samples = materialize_latent(samples)
        source = samples["samples"]
        if width == 0:
            height = max(64, height)
            width = max(64, round(source.shape[-1] * height / source.shape[-2]))
        elif height == 0:
            width = max(64, width)
            height = max(64, round(source.shape[-2] * width / source.shape[-1]))
        else:
            width, height = max(64, width), max(64, height)
        utils = importlib.import_module("dinkster_inference.utils")
        result = samples.copy()
        result["samples"] = utils.common_upscale(
            source, width // 8, height // 8, upscale_method, crop
        )
        return cls.outputs(LATENT=result)


class LatentUpscaleBy(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentUpscaleBy",
            "Upscale Latent By",
            (
                _SAMPLES,
                _combo("upscale_method", _UPSCALE_METHODS),
                InputSpec(
                    "scale_by",
                    TypeExpr.concrete("FLOAT"),
                    default=1.5,
                    widget=NumberWidget(min=0.01, max=8.0, step=0.01),
                ),
            ),
        )

    @classmethod
    def execute(
        cls, *, samples: dict[str, Any], upscale_method: str, scale_by: float
    ) -> Mapping[str, object]:
        samples = materialize_latent(samples)
        source = samples["samples"]
        utils = importlib.import_module("dinkster_inference.utils")
        result = samples.copy()
        result["samples"] = utils.common_upscale(
            source,
            round(source.shape[-1] * scale_by),
            round(source.shape[-2] * scale_by),
            upscale_method,
            "disabled",
        )
        return cls.outputs(LATENT=result)


class LatentRotate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentRotate",
            "Rotate Latent",
            (_SAMPLES, _combo("rotation", ("none", "90 degrees", "180 degrees", "270 degrees"))),
            "model/latent/transform",
        )

    @classmethod
    def execute(cls, *, samples: dict[str, Any], rotation: str) -> Mapping[str, object]:
        samples = materialize_latent(samples)
        torch = importlib.import_module("torch")
        turns = 0
        for prefix, count in (("90", 1), ("180", 2), ("270", 3)):
            if rotation.startswith(prefix):
                turns = count
                break
        result = samples.copy()
        result["samples"] = torch.rot90(samples["samples"], k=turns, dims=[3, 2])
        return cls.outputs(LATENT=result)


class LatentFlip(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentFlip",
            "Flip Latent",
            (_SAMPLES, _combo("flip_method", ("x-axis: vertically", "y-axis: horizontally"))),
            "model/latent/transform",
        )

    @classmethod
    def execute(cls, *, samples: dict[str, Any], flip_method: str) -> Mapping[str, object]:
        samples = materialize_latent(samples)
        torch = importlib.import_module("torch")
        result = samples.copy()
        if flip_method.startswith("x"):
            result["samples"] = torch.flip(samples["samples"], dims=[2])
        elif flip_method.startswith("y"):
            result["samples"] = torch.flip(samples["samples"], dims=[3])
        return cls.outputs(LATENT=result)


class SetLatentNoiseMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "SetLatentNoiseMask",
            "Set Latent Noise Mask",
            (_SAMPLES, InputSpec("mask", TypeExpr.concrete("MASK"))),
        )

    @classmethod
    def execute(cls, *, samples: dict[str, Any], mask: Any) -> Mapping[str, object]:
        mask = materialize_image(mask)
        result = samples.copy()
        result["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        return cls.outputs(LATENT=result)


LATENT_NODES = (
    EmptyLatentImage,
    LatentFromBatch,
    RepeatLatentBatch,
    LatentUpscale,
    LatentUpscaleBy,
    LatentRotate,
    LatentFlip,
    SetLatentNoiseMask,
)
