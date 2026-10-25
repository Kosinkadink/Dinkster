"""Portable sampling settings and governed process-resident runtime objects."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from dinkster_memory.resident import default_pool
from dinkster_values import (
    LATENT_CODEC_MAGIC,
    NOISE_TYPE,
    SAMPLER_TYPE,
    SIGMAS_TYPE,
    EncodedLatentTensor,
    ResidencyTable,
    ResidentCodec,
    TypeRegistry,
    TypeSpec,
    decode_latent,
    encode_latent,
    stable_hash,
)

OptionValue: TypeAlias = float | int | bool | str | None


@dataclass(frozen=True, slots=True)
class BuiltinSamplerSelection:
    sampler_id: str
    options: tuple[tuple[str, OptionValue], ...]

    def __post_init__(self) -> None:
        if type(self.sampler_id) is not str or not self.sampler_id:
            raise ValueError("built-in sampler selection requires a sampler id")
        if type(self.options) is not tuple:
            raise TypeError("built-in sampler options must be a tuple")
        names: list[str] = []
        for item in self.options:
            if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str:
                raise TypeError("built-in sampler options must be (name, value) tuples")
            names.append(item[0])
        if len(names) != len(set(names)):
            raise ValueError("built-in sampler option names must be unique")


@dataclass(frozen=True, slots=True)
class NoiseSelection:
    seed: int | None


@dataclass(frozen=True, slots=True)
class SigmaSchedule:
    values: tuple[float, ...]
    source_scheduler_id: str | None = None


@dataclass(frozen=True, slots=True)
class SamplerSelection(BuiltinSamplerSelection):
    extension_snapshot_digest: str | None = None
    extension_ids: tuple[str, ...] = ()


def _encode_sigma_tensor(value: object) -> bytes:
    try:
        encoded = encode_latent({"samples": value})
        decoded = decode_latent(encoded)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"invalid SIGMAS tensor: {type(value).__name__}") from exc
    if type(decoded) is not dict:
        raise TypeError("SIGMAS tensor frame must contain only samples")
    record = cast("dict[str, object]", decoded)
    if set(record) != {"samples"}:
        raise TypeError("SIGMAS tensor frame must contain only samples")
    samples = record["samples"]
    if type(samples) is not EncodedLatentTensor:
        raise TypeError("SIGMAS tensor frame samples must be a tensor")
    return encoded


def _encode(type_id: str, obj: object) -> bytes:
    record: dict[str, object]
    if type_id == NOISE_TYPE and type(obj) is NoiseSelection:
        seed = obj.seed
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("noise seed must be None or an unsigned 64-bit integer")
        record = {"seed": seed}
    elif type_id == SIGMAS_TYPE and type(obj) is not SigmaSchedule:
        return _encode_sigma_tensor(obj)
    elif type_id == SIGMAS_TYPE and type(obj) is SigmaSchedule:
        if type(obj.values) is not tuple or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in obj.values
        ):
            raise ValueError("sigmas must be a tuple of finite numbers")
        if obj.source_scheduler_id is not None and (
            type(obj.source_scheduler_id) is not str or not obj.source_scheduler_id
        ):
            raise ValueError("sigmas source scheduler id must be None or a nonempty string")
        record = {"values": obj.values}
        if obj.source_scheduler_id is not None:
            record["source_scheduler_id"] = obj.source_scheduler_id
    elif type_id == SAMPLER_TYPE and type(obj) in (BuiltinSamplerSelection, SamplerSelection):
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
    if type_id == SIGMAS_TYPE and data.startswith(LATENT_CODEC_MAGIC):
        decoded = decode_latent(data)
        if type(decoded) is not dict:
            raise ValueError("SIGMAS tensor frame must contain only samples")
        record = cast("dict[str, object]", decoded)
        if set(record) != {"samples"}:
            raise ValueError("SIGMAS tensor frame must contain only samples")
        tensor = record["samples"]
        if type(tensor) is not EncodedLatentTensor:
            raise ValueError("SIGMAS tensor frame samples must be a tensor")
        if _encode_sigma_tensor(tensor) != data:
            raise ValueError("SIGMAS tensor frame must use canonical encoding")
        return tensor
    try:
        loaded: object = json.loads(data)
        if type(loaded) is not dict:
            raise ValueError("sampling settings must be an object")
        record = cast("dict[str, Any]", loaded)
        if record.pop("format") != type_id + ".v1":
            raise ValueError("invalid sampling settings format")
        if type_id == NOISE_TYPE:
            result: object = NoiseSelection(**record)
        elif type_id == SIGMAS_TYPE:
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
    resident_table: ResidencyTable | None = None,
) -> TypeSpec:
    if type_id not in (SAMPLER_TYPE, SIGMAS_TYPE, NOISE_TYPE):
        raise ValueError(f"unknown sampling type: {type_id}")
    if type_id in registry:
        return registry.spec(type_id)

    table = resident_table if resident_table is not None else default_pool()
    resident = ResidentCodec(table)
    resident_format = type_id + ".resident.v1"
    resident_magic = resident_format.encode("ascii") + b"\x00"

    def normalize(obj: object) -> object:
        if type_id == SAMPLER_TYPE and sampler_coerce is not None:
            return sampler_coerce(obj)
        return obj

    def is_runtime(obj: object) -> bool:
        if type_id == SIGMAS_TYPE:
            return False
        if type(obj) in (BuiltinSamplerSelection, SamplerSelection, NoiseSelection):
            return False
        method = "sample" if type_id == SAMPLER_TYPE else "generate_noise"
        if not callable(getattr(obj, method, None)):
            raise TypeError(f"invalid {type_id} runtime object: expected callable {method}")
        return True

    def encode(obj: object) -> bytes:
        obj = normalize(obj)
        if is_runtime(obj):
            return resident_magic + resident.encode(obj)
        return _encode(type_id, obj)

    def decode(data: bytes) -> object:
        if type_id != SIGMAS_TYPE and data.startswith(resident_magic):
            payload = data[len(resident_magic) :]
            resident.validate_encoded(payload)
            obj = resident.decode(payload)
            if not is_runtime(obj):
                raise TypeError(f"invalid {type_id} resident object")
            return obj
        return _decode(type_id, data)

    def fingerprint(obj: object) -> str:
        obj = normalize(obj)
        if is_runtime(obj):
            return resident.fingerprint(obj)
        return stable_hash([_encode(type_id, obj)])

    def metadata(obj: object) -> Mapping[str, object]:
        obj = normalize(obj)
        if is_runtime(obj):
            return {**resident.metadata(obj), "format": resident_format}
        return {}

    def validate(data: bytes, meta: Mapping[str, object]) -> None:
        if type_id != SIGMAS_TYPE and data.startswith(resident_magic):
            if meta.get("format") != resident_format:
                raise ValueError(f"invalid {type_id} resident metadata format")
            resident.validate_encoded(data[len(resident_magic) :])
            return
        if meta.get("format") == resident_format:
            raise ValueError(f"invalid {type_id} resident envelope")
        _decode(type_id, data)

    spec = registry.register(
        type_id,
        encode=encode,
        decode=decode,
        fingerprint=fingerprint,
        meta=metadata if type_id != SIGMAS_TYPE else None,
        validate_encoded=validate,
    )
    if type_id != SIGMAS_TYPE:
        registry.register_resident_table(table)
    return spec
