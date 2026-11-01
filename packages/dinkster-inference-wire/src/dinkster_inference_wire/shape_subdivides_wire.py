"""Ordered sparse subdivision stages on the structural tensor wire."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, cast

from dinkster_values import (
    SHAPE_SUBDIVIDES_TYPE,
    EncodedLatentTensor,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    stable_hash,
)

from .media_wire import materialize_latent
from .process import initialize_inference_process


def _stages(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("SHAPE_SUBDIVIDES requires an ordered stage list")
    stages: list[dict[str, object]] = []
    for stage in cast("list[Any]", value):
        if isinstance(stage, Mapping):
            attributes = cast("Mapping[str, Any]", stage)
            if set(attributes) != {"feats", "coords", "shape", "scale"}:
                raise ValueError("SHAPE_SUBDIVIDES stage fields do not match")
            feats, coords, shape, scale = (
                attributes["feats"],
                attributes["coords"],
                attributes["shape"],
                attributes["scale"],
            )
        else:
            feats, coords, shape, scale = stage.feats, stage.coords, stage.shape, stage._scale
        if not isinstance(shape, tuple):
            raise ValueError("SHAPE_SUBDIVIDES shape must contain nonnegative integers")
        dimensions = cast("tuple[Any, ...]", shape)
        if any(type(n) is not int or n < 0 for n in dimensions):
            raise ValueError("SHAPE_SUBDIVIDES shape must contain nonnegative integers")
        if not isinstance(scale, tuple) or len(cast("tuple[object, ...]", scale)) != 3:
            raise ValueError("SHAPE_SUBDIVIDES scale must contain three rational components")
        components: list[tuple[int, int]] = []
        for component in cast("tuple[Any, ...]", scale):
            if isinstance(component, Fraction) or type(component) is int:
                component = (component.numerator, component.denominator)
            if not isinstance(component, tuple):
                raise ValueError("SHAPE_SUBDIVIDES scale requires integer numerator/denominator")
            pair = cast("tuple[Any, ...]", component)
            if len(pair) != 2 or any(type(n) is not int for n in pair) or pair[1] <= 0:
                raise ValueError("SHAPE_SUBDIVIDES scale requires integer numerator/denominator")
            components.append(cast("tuple[int, int]", pair))
        stages.append(
            {
                "feats": feats,
                "coords": coords,
                "shape": tuple(dimensions),
                "scale": tuple(components),
            }
        )
    return stages


def _decode(data: bytes) -> list[dict[str, object]]:
    frame = cast("dict[str, object]", decode_latent(data))
    if set(frame) != {"samples"}:
        raise ValueError("SHAPE_SUBDIVIDES frame must contain only samples")
    samples = frame["samples"]
    if not isinstance(samples, list) or any(
        not isinstance(stage, Mapping) for stage in cast("list[object]", samples)
    ):
        raise ValueError("SHAPE_SUBDIVIDES frame requires a list of stage mappings")
    stages = _stages(samples)
    for stage in stages:
        feats, coords = stage["feats"], stage["coords"]
        shape = cast("tuple[int, ...]", stage["shape"])
        if not isinstance(feats, EncodedLatentTensor) or len(feats.shape) < 2:
            raise ValueError("SHAPE_SUBDIVIDES feats must be a rank-two or higher tensor")
        if (
            not isinstance(coords, EncodedLatentTensor)
            or coords.dtype not in {"int32", "int64"}
            or coords.shape != (feats.shape[0], 4)
        ):
            raise ValueError("SHAPE_SUBDIVIDES coords must be matching integer batch/xyz rows")
        if len(shape) != len(feats.shape) or shape[1:] != feats.shape[1:]:
            raise ValueError("SHAPE_SUBDIVIDES shape must retain feature dimensions")
    return stages


def _encode(value: object) -> bytes:
    data = encode_latent({"samples": _stages(value)})
    _decode(data)
    return data


def _validate(data: bytes, _metadata: Mapping[str, object]) -> None:
    _decode(data)


def register_shape_subdivides_type(registry: TypeRegistry) -> TypeSpec:
    if SHAPE_SUBDIVIDES_TYPE in registry:
        return registry.spec(SHAPE_SUBDIVIDES_TYPE)
    return registry.register(
        SHAPE_SUBDIVIDES_TYPE,
        encode=_encode,
        decode=_decode,
        fingerprint=lambda value: stable_hash(
            [SHAPE_SUBDIVIDES_TYPE.encode("ascii"), _encode(value)]
        ),
        validate_encoded=_validate,
    )


def materialize_shape_subdivides(value: object) -> list[Any]:
    """Rebuild sparse stages without spatial caches; leave device placement to the node."""
    stages = _stages(value)
    initialize_inference_process()
    torch = importlib.import_module("torch")
    runtime = importlib.import_module("dinkster_inference.ldm.trellis2.vae")
    return [
        runtime.SparseTensor(
            feats=materialize_latent(stage["feats"]),
            coords=materialize_latent(stage["coords"]),
            shape=torch.Size(stage["shape"]),
            scale=tuple(
                Fraction(n, d) for n, d in cast("tuple[tuple[int, int], ...]", stage["scale"])
            ),
        )
        for stage in stages
    ]
