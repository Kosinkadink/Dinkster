"""Portable sampling settings; model-bearing guiders remain resident handles."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from dinkster_values import TypeRegistry, TypeSpec, stable_hash

from .sampling import BuiltinSamplerSelection, OptionValue


@dataclass(frozen=True, slots=True)
class NoiseSelection:
    seed: int | None


@dataclass(frozen=True, slots=True)
class SigmaSchedule:
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SamplerSelection(BuiltinSamplerSelection):
    extension_snapshot_digest: str | None = None
    extension_ids: tuple[str, ...] = ()


def _encode(type_id: str, obj: object) -> bytes:
    record: dict[str, object]
    if type_id == "dinkster.noise" and type(obj) is NoiseSelection:
        seed = obj.seed
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("noise seed must be None or an unsigned 64-bit integer")
        record = {"seed": seed}
    elif type_id == "dinkster.sigmas" and type(obj) is SigmaSchedule:
        if type(obj.values) is not tuple or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in obj.values
        ):
            raise ValueError("sigmas must be a tuple of finite numbers")
        record = {"values": obj.values}
    elif type_id == "dinkster.sampler" and type(obj) in (BuiltinSamplerSelection, SamplerSelection):
        selection = cast("BuiltinSamplerSelection", obj)
        selection.__post_init__()
        options: list[tuple[str, object]] = []
        for name, value in selection.options:
            if type(value) not in (int, float, bool, str, type(None)) or (
                type(value) is float and math.isnan(value)
            ):
                raise ValueError("sampler options must be scalar values other than NaN")
            encoded_value = (
                {"float": "+inf" if value > 0 else "-inf"}
                if type(value) is float and math.isinf(value)
                else value
            )
            options.append((name, encoded_value))
        record = {"sampler_id": selection.sampler_id, "options": options}
        if type(obj) is SamplerSelection:
            if obj.extension_snapshot_digest is not None and (
                type(obj.extension_snapshot_digest) is not str or not obj.extension_snapshot_digest
            ):
                raise ValueError("sampler snapshot must be None or a nonempty string")
            if type(obj.extension_ids) is not tuple or any(
                type(value) is not str or not value for value in obj.extension_ids
            ):
                raise ValueError("sampler extension ids must be nonempty strings")
            record.update(
                extension_snapshot_digest=obj.extension_snapshot_digest,
                extension_ids=obj.extension_ids,
            )
    else:
        raise TypeError(f"invalid {type_id} sampling settings: {type(obj).__name__}")
    return json.dumps(
        {"format": type_id + ".v1", **record},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _decode(type_id: str, data: bytes) -> object:
    try:
        loaded: object = json.loads(data)
        if type(loaded) is not dict:
            raise ValueError("sampling settings must be an object")
        record = cast("dict[str, Any]", loaded)
        if record.pop("format") != type_id + ".v1":
            raise ValueError("invalid sampling settings format")
        if type_id == "dinkster.noise":
            result: object = NoiseSelection(**record)
        elif type_id == "dinkster.sigmas":
            if type(record["values"]) is not list:
                raise ValueError("sigmas values must be an array")
            result = SigmaSchedule(tuple(record.pop("values")), **record)
        else:
            if type(record["options"]) is not list or any(
                type(item) is not list or len(item) != 2
                for item in cast("list[object]", record["options"])
            ):
                raise ValueError("sampler options must be an array of pairs")
            decoded_options: list[tuple[str, OptionValue]] = []
            for name, value in record.pop("options"):
                if value == {"float": "+inf"}:
                    value = math.inf
                elif value == {"float": "-inf"}:
                    value = -math.inf
                decoded_options.append((name, value))
            options = tuple(decoded_options)
            if "extension_ids" in record:
                if type(record["extension_ids"]) is not list:
                    raise ValueError("sampler extension ids must be an array")
                result = SamplerSelection(
                    options=options, extension_ids=tuple(record.pop("extension_ids")), **record
                )
            else:
                result = BuiltinSamplerSelection(options=options, **record)
        if _encode(type_id, result) != data:
            raise ValueError("sampling settings must use canonical encoding")
        return result
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid sampling settings payload") from exc


def register_sampling_type(
    registry: TypeRegistry,
    type_id: str,
    *,
    sampler_coerce: Callable[[object], object] | None = None,
) -> TypeSpec:
    if type_id not in ("dinkster.sampler", "dinkster.sigmas", "dinkster.noise"):
        raise ValueError(f"unknown sampling type: {type_id}")
    if type_id in registry:
        return registry.spec(type_id)

    def encode(obj: object) -> bytes:
        if type_id == "dinkster.sampler" and sampler_coerce is not None:
            obj = sampler_coerce(obj)
        return _encode(type_id, obj)

    def validate(data: bytes, meta: Mapping[str, object]) -> None:
        del meta
        _decode(type_id, data)

    return registry.register(
        type_id,
        encode=encode,
        decode=lambda data: _decode(type_id, data),
        fingerprint=lambda obj: stable_hash([encode(obj)]),
        validate_encoded=validate,
    )
