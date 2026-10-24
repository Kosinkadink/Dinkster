"""Canonical positional product type ids."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .model import Value, ValueMeta, stable_hash


def tuple_type_id(*elements: str) -> str:
    return f"tuple<{','.join(elements)}>"


def parse_tuple_type_id(type_id: str) -> tuple[str, ...] | None:
    """Split positions without splitting nested type constructors."""
    if not type_id.startswith("tuple<") or not type_id.endswith(">"):
        return None
    body = type_id[6:-1]
    if not body:
        return ()
    positions: list[str] = []
    start = depth = 0
    for index, char in enumerate(body):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth < 0:
                return None
        elif char == "," and depth == 0:
            positions.append(body[start:index])
            start = index + 1
    if depth:
        return None
    positions.append(body[start:])
    return tuple(positions) if all(positions) else None


@dataclass(frozen=True)
class TuplePayload:
    """Keep each position's full envelope until a node resolves the tuple."""

    children: tuple[Value, ...]
    transport: str = "tuple"

    def load(self) -> object:
        return tuple(child.resolve() for child in self.children)


def make_tuple_value(children: Sequence[Value]) -> Value:
    ordered = tuple(children)
    parts = [b"tuple"]
    for child in ordered:
        parts.extend((child.type_id.encode("utf-8"), child.fingerprint.encode("utf-8")))
    return Value(
        type_id=tuple_type_id(*(child.type_id for child in ordered)),
        fingerprint=stable_hash(parts),
        meta=ValueMeta({"length": len(ordered)}),
        payload=TuplePayload(ordered),
    )
