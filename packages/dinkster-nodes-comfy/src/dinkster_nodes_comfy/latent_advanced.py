"""Advanced latent arithmetic, layout and batch operations."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ComboOption,
    ComboWidget,
    Deprecation,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)

_LATENT = TypeExpr.concrete("LATENT")
_INT = TypeExpr.concrete("INT")
_FLOAT = TypeExpr.concrete("FLOAT")
_COMBO = TypeExpr.concrete("COMBO")
_MAX_RESOLUTION = 16384


def _integer(name: str, default: int, minimum: int, maximum: int) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=1),
    )


def _combo(name: str, options: tuple[str, ...], default: str | None = None) -> InputSpec:
    return InputSpec(
        name,
        _COMBO,
        default=default,
        widget=ComboWidget(tuple(ComboOption(option) for option in options)),
    )


def _schema(
    node_type: str,
    display_name: str,
    inputs: tuple[InputSpec, ...],
    category: str,
    *,
    deprecated: bool = False,
) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category=category,
        inputs=inputs,
        outputs=(OutputSpec("LATENT", _LATENT),),
        deprecation=Deprecation("Deprecated in ComfyUI.") if deprecated else None,
        search_visibility="deprecated" if deprecated else "normal",
    )


def _reshape_latent_to(
    target_shape: tuple[int, ...], latent: Any, repeat_batch: bool = True
) -> Any:
    utils = importlib.import_module("dinkster_inference.utils")
    if latent.shape[1:] != target_shape[1:]:
        latent = utils.common_upscale(
            latent, target_shape[-1], target_shape[-2], "bilinear", "center"
        )
    return utils.repeat_to_batch_size(latent, target_shape[0]) if repeat_batch else latent


class LatentAdd(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentAdd",
            "LatentAdd",
            (InputSpec("samples1", _LATENT), InputSpec("samples2", _LATENT)),
            "model/latent/advanced",
        )

    @classmethod
    def execute(cls, *, samples1: dict[str, Any], samples2: dict[str, Any]) -> Mapping[str, object]:
        samples1, samples2 = materialize_latent(samples1), materialize_latent(samples2)
        result = samples1.copy()
        result["samples"] = samples1["samples"] + _reshape_latent_to(
            samples1["samples"].shape, samples2["samples"]
        )
        return cls.outputs(LATENT=result)


class LatentSubtract(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentSubtract",
            "LatentSubtract",
            (InputSpec("samples1", _LATENT), InputSpec("samples2", _LATENT)),
            "model/latent/advanced",
        )

    @classmethod
    def execute(cls, *, samples1: dict[str, Any], samples2: dict[str, Any]) -> Mapping[str, object]:
        samples1, samples2 = materialize_latent(samples1), materialize_latent(samples2)
        result = samples1.copy()
        result["samples"] = samples1["samples"] - _reshape_latent_to(
            samples1["samples"].shape, samples2["samples"]
        )
        return cls.outputs(LATENT=result)


class LatentMultiply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentMultiply",
            "LatentMultiply",
            (
                InputSpec("samples", _LATENT),
                InputSpec(
                    "multiplier",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=-10.0, max=10.0, step=0.01),
                ),
            ),
            "model/latent/advanced",
        )

    @classmethod
    def execute(cls, *, samples: dict[str, Any], multiplier: float) -> Mapping[str, object]:
        samples = materialize_latent(samples)
        result = samples.copy()
        result["samples"] = samples["samples"] * multiplier
        return cls.outputs(LATENT=result)


class LatentInterpolate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentInterpolate",
            "LatentInterpolate",
            (
                InputSpec("samples1", _LATENT),
                InputSpec("samples2", _LATENT),
                InputSpec(
                    "ratio", _FLOAT, default=1.0, widget=NumberWidget(min=0.0, max=1.0, step=0.01)
                ),
            ),
            "model/latent/advanced",
        )

    @classmethod
    def execute(
        cls, *, samples1: dict[str, Any], samples2: dict[str, Any], ratio: float
    ) -> Mapping[str, object]:
        samples1, samples2 = materialize_latent(samples1), materialize_latent(samples2)
        torch = importlib.import_module("torch")
        result = samples1.copy()
        first = samples1["samples"]
        second = _reshape_latent_to(first.shape, samples2["samples"])
        first_magnitude = torch.linalg.vector_norm(first, dim=1)
        second_magnitude = torch.linalg.vector_norm(second, dim=1)
        first = torch.nan_to_num(first / first_magnitude)
        second = torch.nan_to_num(second / second_magnitude)
        blended = first * ratio + second * (1.0 - ratio)
        normalized = torch.nan_to_num(blended / torch.linalg.vector_norm(blended, dim=1))
        result["samples"] = normalized * (
            first_magnitude * ratio + second_magnitude * (1.0 - ratio)
        )
        return cls.outputs(LATENT=result)


class LatentConcat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentConcat",
            "LatentConcat",
            (
                InputSpec("samples1", _LATENT),
                InputSpec("samples2", _LATENT),
                _combo("dim", ("x", "-x", "y", "-y", "t", "-t")),
            ),
            "model/latent/advanced",
        )

    @classmethod
    def execute(
        cls, *, samples1: dict[str, Any], samples2: dict[str, Any], dim: str
    ) -> Mapping[str, object]:
        samples1, samples2 = materialize_latent(samples1), materialize_latent(samples2)
        torch = importlib.import_module("torch")
        result = samples1.copy()
        first, second = samples1["samples"], samples2["samples"]
        second = importlib.import_module("dinkster_inference.utils").repeat_to_batch_size(
            second, first.shape[0]
        )
        if "-" in dim:
            first, second = second, first
        axis = -1 if "x" in dim else -2 if "y" in dim else -3
        result["samples"] = torch.cat((first, second), dim=axis)
        return cls.outputs(LATENT=result)


class LatentCut(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentCut",
            "LatentCut",
            (
                InputSpec("samples", _LATENT),
                _combo("dim", ("x", "y", "t")),
                _integer("index", 0, -_MAX_RESOLUTION, _MAX_RESOLUTION),
                _integer("amount", 1, 1, _MAX_RESOLUTION),
            ),
            "model/latent/advanced",
        )

    @classmethod
    def execute(
        cls, *, samples: dict[str, Any], dim: str, index: int, amount: int
    ) -> Mapping[str, object]:
        samples = materialize_latent(samples)
        torch = importlib.import_module("torch")
        result = samples.copy()
        source = samples["samples"]
        axis = source.ndim - (1 if dim == "x" else 2 if dim == "y" else 3)
        if index >= 0:
            index = min(index, source.shape[axis] - 1)
            amount = min(source.shape[axis] - index, amount)
        else:
            index = max(index, -source.shape[axis])
            amount = min(-index, amount)
        result["samples"] = torch.narrow(source, axis, index, amount)
        return cls.outputs(LATENT=result)


class LatentCutToBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentCutToBatch",
            "LatentCutToBatch",
            (
                InputSpec("samples", _LATENT),
                _combo("dim", ("t", "x", "y")),
                _integer("slice_size", 1, 1, _MAX_RESOLUTION),
            ),
            "model/latent/advanced",
        )

    @classmethod
    def execute(cls, *, samples: dict[str, Any], dim: str, slice_size: int) -> Mapping[str, object]:
        samples = materialize_latent(samples)
        result = samples.copy()
        source = samples["samples"]
        axis = source.ndim - (1 if dim == "x" else 2 if dim == "y" else 3)
        if axis < 2:
            return cls.outputs(LATENT=samples)
        split = source.movedim(axis, 1)
        if split.shape[1] < slice_size:
            slice_size = split.shape[1]
        elif split.shape[1] % slice_size != 0:
            split = split[:, : math.floor(split.shape[1] / slice_size) * slice_size]
        result["samples"] = split.reshape([-1, slice_size, *split.shape[2:]]).movedim(1, axis)
        return cls.outputs(LATENT=result)


class LatentBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentBatch",
            "Batch Latents (DEPRECATED)",
            (InputSpec("samples1", _LATENT), InputSpec("samples2", _LATENT)),
            "model/latent/batch",
            deprecated=True,
        )

    @classmethod
    def execute(cls, *, samples1: dict[str, Any], samples2: dict[str, Any]) -> Mapping[str, object]:
        samples1, samples2 = materialize_latent(samples1), materialize_latent(samples2)
        torch = importlib.import_module("torch")
        result = samples1.copy()
        first, second = samples1["samples"], samples2["samples"]
        second = _reshape_latent_to(first.shape, second, repeat_batch=False)
        result["samples"] = torch.cat((first, second), dim=0)
        result["batch_index"] = samples1.get(
            "batch_index", list(range(first.shape[0]))
        ) + samples2.get("batch_index", list(range(second.shape[0])))
        return cls.outputs(LATENT=result)


class LatentBatchSeedBehavior(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "LatentBatchSeedBehavior",
            "LatentBatchSeedBehavior",
            (InputSpec("samples", _LATENT), _combo("seed_behavior", ("random", "fixed"), "fixed")),
            "model/latent/advanced",
        )

    @classmethod
    def execute(cls, *, samples: dict[str, Any], seed_behavior: str) -> Mapping[str, object]:
        samples = materialize_latent(samples)
        result = samples.copy()
        if seed_behavior == "random":
            result.pop("batch_index", None)
        elif seed_behavior == "fixed":
            result["batch_index"] = [result.get("batch_index", [0])[0]] * samples["samples"].shape[
                0
            ]
        return cls.outputs(LATENT=result)


LATENT_ADVANCED_NODES = (
    LatentAdd,
    LatentSubtract,
    LatentMultiply,
    LatentInterpolate,
    LatentConcat,
    LatentCut,
    LatentCutToBatch,
    LatentBatch,
    LatentBatchSeedBehavior,
)
