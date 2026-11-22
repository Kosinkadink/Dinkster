"""What ``execute`` receives and returns for a node body: the body's tree without its layout.

A run carries one flat value per flattened id (``loras.2.strength``). Before
``execute`` the values are put back into the tree the body declares; after it,
the returned tree is flattened to output ids again.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from .body import (
    BodyRow,
    Choice,
    Described,
    Group,
    State,
    TypeChoice,
    option_rows,
    scope_rows,
)
from .model import Input, Output, SlotValue
from .output_descriptors import output_descriptor_entries


class Record(Mapping[str, object]):
    """A group member of records or a chosen option: its ``key`` and its rows by id."""

    __slots__ = ("_rows", "key")

    def __init__(self, key: str, rows: Mapping[str, object]) -> None:
        self.key = key
        self._rows = dict(rows)

    def __getitem__(self, row_id: str) -> object:
        return self._rows[row_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __repr__(self) -> str:
        return f"Record(key={self.key!r}, rows={self._rows!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Record):
            return self.key == other.key and self._rows == other._rows
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Item:
    """A member of a group of values: its ``key`` and its value."""

    key: str
    value: object


class Members[M: (Record, Item)](tuple[M, ...]):
    """A group's members in the run's order: ``Members[Record]`` for a group of
    records, ``Members[Item]`` for a group of values."""

    __slots__ = ()

    def as_dict(self) -> dict[str, object]:
        """Key to value, for a group of values; key to record for a group of records."""
        return {member.key: member.value if isinstance(member, Item) else member for member in self}


def _row_id(row: BodyRow) -> str:
    return getattr(row, "id", "")


def nest_inputs(
    body: Sequence[BodyRow],
    values: Mapping[str, object],
    members: Mapping[str, Sequence[str]],
    choices: Mapping[str, str],
    outputs: Collection[str] = (),
) -> dict[str, object]:
    """Put a run's flat input values back into the body's tree.

    ``members`` is each group's member keys in the run's order and ``choices``
    each choice's and type choice's chosen key, both by flattened path. A row
    with no value in the run is left out, as an absent input is; an optional
    choice with nothing chosen and a type choice with nothing plugged in are
    present as ``None``, because "none" is a stored answer. ``outputs`` is the
    run's flattened output ids: a described output entry is present as
    ``None``, as a member of a group of output values is, so the node knows
    which entries to return."""

    def scope(rows: Sequence[BodyRow], prefix: str) -> dict[str, object]:
        out: dict[str, object] = {}
        for row in scope_rows(tuple(rows)):
            path = prefix + _row_id(row)
            if isinstance(row, (Input, State)):
                if path in values:
                    out[row.id] = values[path]
            elif isinstance(row, Group):
                keys = members.get(path, ())
                if row.value is not None:
                    out[row.id] = Members(Item(key, values.get(f"{path}.{key}")) for key in keys)
                else:
                    assert row.member is not None
                    out[row.id] = Members(
                        Record(key, scope(row.member, f"{path}.{key}.")) for key in keys
                    )
            elif isinstance(row, Choice):
                key = choices.get(path)
                out[row.id] = (
                    None if key is None else Record(key, scope(option_rows(row, key), f"{path}."))
                )
            elif isinstance(row, TypeChoice):
                key = choices.get(path)
                variant = None if key is None else row.variant(key)
                plugged = variant is not None if row.variants is not None else path in values
                if not plugged:
                    out[row.id] = None
                    continue
                inner = (*row.rows, *(variant.rows if variant is not None else ()))
                out[row.id] = SlotValue(
                    variant=key or "", value=values.get(path), options=scope(inner, f"{path}.")
                )
            elif isinstance(row, Described):
                # Entries arrive in the description's order, as flattening
                # lists them; an output entry is present as ``None``.
                description = values.get(path)
                entries: dict[str, object] = {}
                for entry in (
                    output_descriptor_entries(description) if isinstance(description, str) else ()
                ):
                    entry_id = cast("str", entry["id"])
                    flat_id = f"{path}.{entry_id}"
                    if flat_id in values:
                        entries[entry_id] = values[flat_id]
                    elif flat_id in outputs:
                        entries[entry_id] = None
                out[row.id] = entries
        return out

    return scope(body, "")


class OutputShapeError(ValueError):
    """A returned tree that does not match the body's outputs."""


def flatten_outputs(body: Sequence[BodyRow], returned: Mapping[str, object]) -> dict[str, object]:
    """Flatten what ``execute`` returned into output ids.

    A group returns ``{key: {row: value}}`` for records and ``{key: value}``
    for values; a choice, a type choice and a described row return
    ``{row: value}``. Which keys must be present is checked afterwards
    against the run's flattened outputs."""
    flat: dict[str, object] = {}

    def holds_output(rows: Sequence[BodyRow]) -> bool:
        for row in scope_rows(tuple(rows)):
            if isinstance(row, Output):
                return True
            if isinstance(row, Group):
                if isinstance(row.value, Output) or holds_output(row.member or ()):
                    return True
            elif isinstance(row, Choice):
                if holds_output(
                    (*row.shared, *(r for o in row.options for r in scope_rows(o.rows)))
                ):
                    return True
            elif isinstance(row, TypeChoice):
                variants = row.variants or ()
                if holds_output((*row.rows, *(r for v in variants for r in v.rows))):
                    return True
            elif isinstance(row, Described) and any(isinstance(kind, Output) for kind in row.kinds):
                return True
        return False

    def mapping(value: object, path: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise OutputShapeError(f"output '{path}' must be a mapping, got {type(value).__name__}")
        return cast("Mapping[str, object]", value)

    def scope(rows: Sequence[BodyRow], prefix: str, given: Mapping[str, object]) -> None:
        known: set[str] = set()
        for row in scope_rows(tuple(rows)):
            row_id = _row_id(row)
            path = prefix + row_id
            if isinstance(row, Output):
                known.add(row_id)
                if row_id in given:
                    flat[path] = given[row_id]
                continue
            if isinstance(row, (Group, Choice, TypeChoice, Described)):
                if not holds_output((row,)):
                    continue
                known.add(row_id)
                if row_id not in given:
                    continue
                nested = mapping(given[row_id], path)
                if isinstance(row, Group):
                    for key, member in nested.items():
                        if row.value is not None:
                            flat[f"{path}.{key}"] = member
                        else:
                            assert row.member is not None
                            scope(row.member, f"{path}.{key}.", mapping(member, f"{path}.{key}"))
                elif isinstance(row, Choice):
                    every = (*row.shared, *(r for o in row.options for r in scope_rows(o.rows)))
                    scope(every, f"{path}.", nested)
                elif isinstance(row, TypeChoice):
                    variants = row.variants or ()
                    scope((*row.rows, *(r for v in variants for r in v.rows)), f"{path}.", nested)
                else:
                    for entry, value in nested.items():
                        flat[f"{path}.{entry}"] = value
        unknown = sorted(set(given) - known)
        if unknown:
            where = f" in '{prefix[:-1]}'" if prefix else ""
            raise OutputShapeError(f"execute() returned unknown outputs{where}: {unknown}")

    scope(body, "", returned)
    return flat
