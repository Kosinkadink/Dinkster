"""Named containers retain each field's envelope and stable identity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from .model import Value, ValueMeta, stable_hash


def record_fields(raw: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError("record fields must be a sequence of identity/name pairs")
    fields: list[tuple[str, str]] = []
    for item in cast(Sequence[object], raw):
        if not isinstance(item, (list, tuple)):
            raise ValueError("record fields require non-empty identity/name pairs")
        pair = cast(Sequence[object], item)
        if len(pair) != 2 or any(not isinstance(part, str) or not part for part in pair):
            raise ValueError("record fields require non-empty identity/name pairs")
        fields.append(cast(tuple[str, str], tuple(pair)))
    if len({item[0] for item in fields}) != len(fields):
        raise ValueError("record field identities must be unique")
    if len({item[1] for item in fields}) != len(fields):
        raise ValueError("record field names must be unique")
    return tuple(fields)


@dataclass(frozen=True)
class RecordPayload:
    fields: tuple[tuple[str, str], ...]
    children: tuple[Value, ...]
    transport: str = "record"

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.fields), tuple)
            or record_fields(self.fields) != self.fields
        ):
            raise ValueError("record fields must be immutable pairs")
        if not isinstance(cast(object, self.children), tuple) or len(self.fields) != len(
            self.children
        ):
            raise ValueError("record fields and children must have equal lengths")

    def load(self) -> object:
        return {
            name: child.resolve()
            for (_, name), child in zip(self.fields, self.children, strict=True)
        }


def make_record_value(type_id: str, fields: Mapping[str, tuple[str, Value]]) -> Value:
    """Fields map explicit identity to current name and child envelope.

    Names and declaration order do not identify computation. Child type and
    content do, including recursively nested records.
    """
    ordered = sorted(fields.items())
    payload = RecordPayload(
        tuple((identity, name) for identity, (name, _) in ordered),
        tuple(child for _, (_, child) in ordered),
    )
    parts = [b"record", type_id.encode("utf-8")]
    for identity, (_, child) in ordered:
        parts.extend(
            (
                identity.encode("utf-8"),
                child.type_id.encode("utf-8"),
                child.fingerprint.encode("utf-8"),
            )
        )
    return Value(
        type_id=type_id, fingerprint=stable_hash(parts), meta=ValueMeta({}), payload=payload
    )
