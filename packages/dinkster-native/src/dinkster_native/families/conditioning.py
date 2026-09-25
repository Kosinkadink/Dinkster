"""Shared prepared-conditioning parsing for native family adapters."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import Any, cast


def _resident_payload(value: object, inference: Any, name: str) -> Any:
    del inference
    try:
        return cast("Any", value)._dinkster_resident_payload
    except (AttributeError, TypeError) as error:
        raise TypeError(f"{name} must contain one resident payload") from error


def _prepared_multistream_carrier(value: object, inference: Any, name: str) -> Any | None:
    if hasattr(value, "_dinkster_resident_payload"):
        resident = _resident_payload(value, inference, name)
        value = getattr(resident, "conditioning", value)
    if value == []:
        return None
    entries = cast("list[object]", value) if type(value) is list else []
    entry = (
        cast("list[object]", entries[0]) if len(entries) == 1 and type(entries[0]) is list else []
    )
    if (
        len(entry) != 2
        or type(entry[0]) is not inference.PreparedMultiStreamConditioning
        or entry[1] != {}
    ):
        raise TypeError(f"{name} must contain exact prepared multi-stream conditioning")
    return cast("Any", entry[0])


def _prepared_multistream_conditioning(
    value: object, inference: Any, name: str, runtime_identity: str
) -> object | None:
    prepared = _prepared_multistream_carrier(value, inference, name)
    if prepared is None:
        return None
    if prepared.runtime_identity != runtime_identity:
        raise ValueError(f"{name} conditioning was prepared by a different runtime")
    return cast("object", prepared.payload)
