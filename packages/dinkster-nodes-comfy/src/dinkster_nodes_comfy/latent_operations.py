"""Latent transforms and model denoising hooks with core node interfaces."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)

_LATENT = TypeExpr.concrete("LATENT")
_OPERATION = TypeExpr.concrete("LATENT_OPERATION")
_MODEL = TypeExpr.concrete("MODEL")
_CATEGORY = "model/latent/advanced/operations"


def _float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
    step: float,
    *,
    optional: bool = False,
    advanced: bool = False,
) -> InputSpec:
    return InputSpec(
        name,
        TypeExpr.concrete("FLOAT"),
        default=default,
        required=not optional,
        advanced=advanced,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


class LatentApplyOperation(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LatentApplyOperation",
            category=_CATEGORY,
            inputs=(InputSpec("samples", _LATENT), InputSpec("operation", _OPERATION)),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(cls, *, samples: dict[str, Any], operation: Any) -> Mapping[str, object]:
        samples = materialize_latent(samples)
        result = samples.copy()
        result["samples"] = operation(latent=samples["samples"])
        return cls.outputs(LATENT=result)


class LatentApplyOperationCFG(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LatentApplyOperationCFG",
            category=_CATEGORY,
            inputs=(
                InputSpec("model", _MODEL),
                InputSpec("operation", _OPERATION),
                _float("start_percent", 0.0, 0.0, 1.0, 0.001, optional=True),
                _float("end_percent", 1.0, 0.0, 1.0, 0.001, optional=True),
            ),
            outputs=(OutputSpec("MODEL", _MODEL),),
        )

    @classmethod
    def execute(
        cls, *, model: Any, operation: Any, start_percent: float = 0.0, end_percent: float = 1.0
    ) -> Mapping[str, object]:
        result = model.clone()
        sampling = result.get_model_object("model_sampling")
        sigma_start = sampling.percent_to_sigma(start_percent)
        sigma_end = sampling.percent_to_sigma(end_percent)

        def pre_cfg(args: dict[str, Any]) -> Any:
            outputs = args["conds_out"]
            sigma = args["sigma"][0].item()
            if sigma > sigma_start or sigma < sigma_end:
                return outputs
            if len(outputs) == 2:
                outputs[0] = operation(latent=outputs[0] - outputs[1]) + outputs[1]
            else:
                outputs[0] = operation(latent=outputs[0])
            return outputs

        result.set_model_sampler_pre_cfg_function(pre_cfg)
        return cls.outputs(MODEL=result)


class LatentOperationBlend(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LatentOperationBlend",
            display_name="Latent Operation Blend",
            category=_CATEGORY,
            inputs=(InputSpec("reference", _LATENT), _float("strength", 1.0, 0.0, 1.0, 0.0001)),
            outputs=(OutputSpec("LATENT_OPERATION", _OPERATION),),
        )

    @classmethod
    def execute(cls, *, reference: dict[str, Any], strength: float) -> Mapping[str, object]:
        samples = materialize_latent(reference)["samples"]

        def match_reference(latent: Any, **kwargs: Any) -> Any:
            if strength == 0:
                return latent
            torch = importlib.import_module("torch")
            utils = importlib.import_module("dinkster_inference.utils")
            target = samples.to(latent)
            if target.shape[2:] != latent.shape[2:]:
                target = torch.nn.functional.interpolate(
                    target, size=latent.shape[2:], mode="nearest"
                )
            target = utils.repeat_to_batch_size(target, latent.shape[0])
            return torch.lerp(latent, target, strength)

        return cls.outputs(LATENT_OPERATION=match_reference)


class LatentOperationTonemapReinhard(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LatentOperationTonemapReinhard",
            category=_CATEGORY,
            inputs=(_float("multiplier", 1.0, 0.0, 100.0, 0.01),),
            outputs=(OutputSpec("LATENT_OPERATION", _OPERATION),),
        )

    @classmethod
    def execute(cls, *, multiplier: float) -> Mapping[str, object]:
        def tonemap(latent: Any, **kwargs: Any) -> Any:
            torch = importlib.import_module("torch")
            magnitude = (torch.linalg.vector_norm(latent, dim=1) + 0.0000000001)[:, None]
            normalized = latent / magnitude
            dims = list(range(1, magnitude.ndim))
            mean = torch.mean(magnitude, dim=dims, keepdim=True)
            std = torch.std(magnitude, dim=dims, keepdim=True)
            top = (std * 5 + mean) * multiplier
            magnitude *= 1.0 / top
            result = magnitude / (magnitude + 1.0)
            result *= top
            return normalized * result

        return cls.outputs(LATENT_OPERATION=tonemap)


class LatentOperationSharpen(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LatentOperationSharpen",
            category=_CATEGORY,
            inputs=(
                InputSpec(
                    "sharpen_radius",
                    TypeExpr.concrete("INT"),
                    default=9,
                    advanced=True,
                    widget=NumberWidget(min=1, max=31, step=1),
                ),
                _float("sigma", 1.0, 0.1, 10.0, 0.1, advanced=True),
                _float("alpha", 0.1, 0.0, 5.0, 0.01, advanced=True),
            ),
            outputs=(OutputSpec("LATENT_OPERATION", _OPERATION),),
        )

    @classmethod
    def execute(cls, *, sharpen_radius: int, sigma: float, alpha: float) -> Mapping[str, object]:
        def sharpen(latent: Any, **kwargs: Any) -> Any:
            torch = importlib.import_module("torch")
            luminance = (torch.linalg.vector_norm(latent, dim=1) + 1e-6)[:, None]
            normalized = latent / luminance
            channels = latent.shape[1]
            size = sharpen_radius * 2 + 1
            x, y = torch.meshgrid(
                torch.linspace(-1, 1, size, device=luminance.device),
                torch.linspace(-1, 1, size, device=luminance.device),
                indexing="ij",
            )
            distance = torch.sqrt(x * x + y * y)
            gaussian = torch.exp(-(distance * distance) / (2.0 * sigma * sigma))
            kernel = (gaussian / gaussian.sum()).to(torch.float32)
            center = size // 2
            kernel *= alpha * -10
            kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0
            padded = torch.nn.functional.pad(normalized, (sharpen_radius,) * 4, "reflect")
            result = torch.nn.functional.conv2d(
                padded,
                kernel.repeat(channels, 1, 1).unsqueeze(1),
                padding=size // 2,
                groups=channels,
            )[:, :, sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]
            return luminance * result

        return cls.outputs(LATENT_OPERATION=sharpen)


class ReplaceVideoLatentFrames(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ReplaceVideoLatentFrames",
            display_name="Replace Video Latent Frames",
            category="model/latent/batch",
            inputs=(
                InputSpec("destination", _LATENT),
                InputSpec(
                    "index",
                    TypeExpr.concrete("INT"),
                    default=0,
                    widget=NumberWidget(min=-16384, max=16384, step=1),
                ),
                InputSpec("source", _LATENT, required=False),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(
        cls, *, destination: dict[str, Any], index: int = 0, source: dict[str, Any] | None = None
    ) -> Mapping[str, object]:
        if source is None:
            return cls.outputs(LATENT=destination)
        destination = materialize_latent(destination)
        materialized_source = materialize_latent(source)
        frames = destination["samples"].shape[2]
        source_frames = materialized_source["samples"].shape[2]
        if index < 0:
            index += frames
        if index > frames or index + source_frames > frames:
            logging.warning("ReplaceVideoLatentFrames: Source frames do not fit at index %s", index)
            return cls.outputs(LATENT=destination)
        result = materialized_source.copy()
        samples = destination["samples"].clone()
        samples[:, :, index : index + source_frames] = materialized_source["samples"]
        result["samples"] = samples
        return cls.outputs(LATENT=result)


LATENT_OPERATION_NODES = (
    LatentApplyOperation,
    LatentApplyOperationCFG,
    LatentOperationBlend,
    LatentOperationTonemapReinhard,
    LatentOperationSharpen,
    ReplaceVideoLatentFrames,
)
