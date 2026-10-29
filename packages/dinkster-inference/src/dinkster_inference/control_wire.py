"""Registration of the ``dinkster.control`` boundary type.

The carrier is the torch-free frozen :class:`ControlApplication` chain.
Encoding is canonical compact JSON (oldest application first), so equal
chains produce equal bytes and the fingerprint is content identity. Decode
rebuilds the chain through the constructors so every declared invariant
fires on arrival.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal, cast

from dinkster_values import TypeRegistry, TypeSpec, stable_hash

from .conditioning import PayloadReference, PercentRange
from .controlnet import ControlApplication, SDControlMode, SDControlModeToken

CONTROL_TYPE_ID = "dinkster.control"
CONTROL_WIRE_FORMAT = "dinkster.control-wire.v1"

_RECORD_KEYS = {"childId", "hintId", "strength", "window", "mode"}


def encode_control_application(application: ControlApplication) -> bytes:
    if type(application) is not ControlApplication:
        raise TypeError("dinkster.control carries exact ControlApplication values")
    chain: list[ControlApplication] = []
    current: ControlApplication | None = application
    while current is not None:
        chain.append(current)
        current = current.previous
    chain.reverse()
    records: list[dict[str, object]] = []
    for entry in chain:
        mode: dict[str, object] | None = None
        if entry.mode is not None:
            mode = {"provider": entry.mode.provider, "token": entry.mode.token}
        records.append(
            {
                "childId": entry.child_id,
                "hintId": entry.hint.id,
                "strength": entry.strength,
                "window": [entry.window.start_percent, entry.window.end_percent],
                "mode": mode,
            }
        )
    payload = {"format": CONTROL_WIRE_FORMAT, "applications": records}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _decode_mode(record: object) -> SDControlMode | None:
    if record is None:
        return None
    if not isinstance(record, Mapping) or set(cast("Mapping[str, object]", record)) != {
        "provider",
        "token",
    }:
        raise ValueError("dinkster.control mode record must contain exactly provider and token")
    mapping = cast("Mapping[str, object]", record)
    return SDControlMode(
        provider=cast(Literal["sdxl-controlnet-union"], mapping["provider"]),
        token=cast(SDControlModeToken, mapping["token"]),
    )


def decode_control_application(data: bytes) -> ControlApplication:
    try:
        wire = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid dinkster.control payload") from exc
    if not isinstance(wire, Mapping) or set(cast("Mapping[str, object]", wire)) != {
        "format",
        "applications",
    }:
        raise ValueError("dinkster.control payload must contain exactly format and applications")
    mapping = cast("Mapping[str, object]", wire)
    if mapping["format"] != CONTROL_WIRE_FORMAT:
        raise ValueError(f"unsupported dinkster.control wire format: {mapping['format']!r}")
    records = mapping["applications"]
    if not isinstance(records, list) or not records:
        raise ValueError("dinkster.control payload must contain a nonempty application list")
    application: ControlApplication | None = None
    for item in cast("list[object]", records):
        if not isinstance(item, Mapping) or set(cast("Mapping[str, object]", item)) != _RECORD_KEYS:
            raise ValueError(
                "dinkster.control application record must contain exactly "
                "childId, hintId, strength, window, and mode"
            )
        record = cast("Mapping[str, object]", item)
        window = record["window"]
        if not isinstance(window, list) or len(cast("list[object]", window)) != 2:
            raise ValueError("dinkster.control window must be a two-element list")
        start, end = cast("list[object]", window)
        application = ControlApplication(
            child_id=cast("str", record["childId"]),
            hint=PayloadReference(cast("str", record["hintId"])),
            strength=cast("float", record["strength"]),
            window=PercentRange(cast("float", start), cast("float", end)),
            previous=application,
            mode=_decode_mode(record["mode"]),
        )
    if application is None:
        raise ValueError("dinkster.control payload must contain a nonempty application list")
    # Canonical-bytes enforcement: whitespace, key reordering, exponent-form
    # floats, and duplicate keys (json.loads keeps the last value, which would
    # bypass the exact-key-set checks above) all fail here.
    if encode_control_application(application) != data:
        raise ValueError("dinkster.control payload must be canonical")
    return application


def register_control_type(registry: TypeRegistry) -> TypeSpec:
    def coerce(obj: object) -> object:
        if type(obj) is not ControlApplication:
            raise TypeError("dinkster.control carries exact ControlApplication values")
        return obj

    def encode(obj: object) -> bytes:
        return encode_control_application(cast("ControlApplication", obj))

    def decode(data: bytes) -> object:
        return decode_control_application(data)

    def fingerprint(obj: object) -> str:
        return stable_hash([encode_control_application(cast("ControlApplication", obj))])

    def validate_encoded(data: bytes, meta: Mapping[str, object]) -> None:
        del meta
        decode_control_application(data)

    return registry.register(
        CONTROL_TYPE_ID,
        encode=encode,
        decode=decode,
        fingerprint=fingerprint,
        meta=lambda _obj: {"format": CONTROL_WIRE_FORMAT},
        coerce=coerce,
        validate_encoded=validate_encoded,
    )


__all__ = [
    "CONTROL_TYPE_ID",
    "CONTROL_WIRE_FORMAT",
    "decode_control_application",
    "encode_control_application",
    "register_control_type",
]
