"""Flattening: a node body tree becomes flat inputs and outputs with dotted ids.

Flattening reads only the declaration and the node's document state: its
stored input values (and which are wired), its stored output members, and its
stored choices. A member's rows are ``<group>.<key>.<row>`` in a group of
records and ``<group>.<key>`` in a group of values; a choice's rows are
``<choice>.<row>``, with the option key left out of the path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from .body import (
    UNSET,
    BodyRow,
    Button,
    Choice,
    Described,
    Group,
    Named,
    Note,
    Numbered,
    Option,
    Row,
    Section,
    State,
    TypeChoice,
    Use,
    scope_rows,
)
from .model import InputSpec, OutputSpec, TypeExpr

MAX_MEMBERS = 512


class FlattenError(ValueError):
    """The node's document state does not fit its declared body."""


@dataclass
class Flattened:
    inputs: list[InputSpec]
    outputs: list[OutputSpec]
    choices: list[tuple[str, str]]


def member_id(group_path: str, key: str, row: str | None = None) -> str:
    """The flattened id of a member's row (``None`` for a group of values)."""
    return f"{group_path}.{key}" if row is None else f"{group_path}.{key}.{row}"


def flatten_body(
    body: Sequence[BodyRow],
    stored_ids: Sequence[str],
    output_members: Mapping[str, Sequence[str]],
    choices: Mapping[str, str],
) -> Flattened:
    flat = Flattened([], [], [])
    _Flattener(stored_ids, output_members, choices, flat).rows(body, "")
    return flat


def _scope_member_type(expr: TypeExpr, member: str) -> TypeExpr:
    if expr.kind == "variable":
        return replace(expr, template_id=f"{member}:{expr.template_id}")
    return replace(
        expr,
        element=None if expr.element is None else _scope_member_type(expr.element, member),
        elements=tuple(_scope_member_type(item, member) for item in expr.elements),
    )


def _key_position(keys: Numbered | Named, key: str) -> int | None:
    """The position the naming rule gives a key, or ``None`` if the rule cannot make it."""
    if isinstance(keys, Named):
        return keys.names.index(key) if key in keys.names else None
    number = key.removeprefix(keys.prefix)
    if number == key and keys.prefix:
        return None
    if not number.isascii() or not number.isdecimal():
        return None
    position = int(number) - keys.start
    if position < 0 or keys.key(position) != key:
        return None
    return position


def _apply_use(row: BodyRow, use: Use) -> BodyRow:
    """A shared row as one option places it, with that place's overrides."""
    if isinstance(row, InputSpec):
        changes: dict[str, object] = {}
        if use.label is not UNSET:
            changes["display_name"] = use.label
        if use.tooltip is not UNSET:
            changes["doc"] = use.tooltip
        if use.widget is not UNSET:
            changes["widget"] = use.widget
        if use.required is not UNSET:
            changes["required"] = use.required
            if use.required:
                changes["default"] = None
        if use.default is not UNSET and use.required is not True:
            changes["default"] = use.default
        return replace(row, **changes)
    if isinstance(row, Choice):
        changes = {}
        if use.label is not UNSET:
            changes["label"] = use.label
        if use.tooltip is not UNSET:
            changes["tooltip"] = use.tooltip
        if use.required is not UNSET:
            changes["required"] = use.required
        if use.only is not None:
            changes["options"] = tuple(o for o in row.options if o.key in use.only)
        if use.default is not UNSET:
            changes["default"] = use.default
        choice = replace(row, **changes)
        if use.inside:
            inner = {item.id: item for item in use.inside}
            choice = replace(
                choice,
                shared=tuple(_use_inside(item, inner) for item in choice.shared),
                options=tuple(
                    replace(option, rows=tuple(_use_inside(item, inner) for item in option.rows))
                    for option in choice.options
                ),
            )
        return choice
    return row


def _use_inside(row: BodyRow | Use, inner: Mapping[str, Use]) -> BodyRow | Use:
    if isinstance(row, (Section, Row)):
        return replace(row, rows=cast("Any", tuple(_use_inside(item, inner) for item in row.rows)))
    if isinstance(row, Use):
        override = inner.get(row.id)
        return row if override is None else _merge_use(row, override)
    override = inner.get(row.id) if not isinstance(row, Row) else None
    return row if override is None else _apply_use(row, override)


def _merge_use(place: Use, override: Use) -> Use:
    """An inner ``Use`` over a place that is itself a ``Use``: the inner one wins."""
    return Use(
        place.id,
        label=override.label if override.label is not UNSET else place.label,
        tooltip=override.tooltip if override.tooltip is not UNSET else place.tooltip,
        widget=override.widget if override.widget is not UNSET else place.widget,
        default=override.default if override.default is not UNSET else place.default,
        required=override.required if override.required is not UNSET else place.required,
        only=override.only if override.only is not None else place.only,
        inside=(*place.inside, *override.inside),
    )


def _has_required_row(group: Group) -> bool:
    """Whether a member has a required input of its own, so the minimum binds."""
    rows = (group.value,) if group.value is not None else group.member or ()
    return any(isinstance(row, InputSpec) and row.required for row in scope_rows(cast("Any", rows)))


class _Flattener:
    def __init__(
        self,
        stored_ids: Sequence[str],
        output_members: Mapping[str, Sequence[str]],
        choices: Mapping[str, str],
        flat: Flattened,
    ) -> None:
        self.stored_ids = stored_ids
        self.output_members = output_members
        self.choices = choices
        self.flat = flat

    def rows(
        self,
        rows: Sequence[BodyRow | Use],
        prefix: str,
        shared: Mapping[str, BodyRow] | None = None,
        member: str | None = None,
        required: bool = True,
    ) -> None:
        for row in rows:
            if isinstance(row, Use):
                assert shared is not None
                row = _apply_use(shared[row.id], row)
            self.row(row, prefix, member, required)

    def row(self, row: BodyRow, prefix: str, member: str | None, required: bool) -> None:
        path = f"{prefix}{row.id}" if not isinstance(row, Row) else prefix
        if isinstance(row, InputSpec):
            spec = replace(row, id=path, required=row.required and required)
            if member is not None:
                spec = replace(spec, type=_scope_member_type(spec.type, member))
            self.flat.inputs.append(spec)
        elif isinstance(row, OutputSpec):
            spec = replace(row, id=path)
            if member is not None:
                spec = replace(spec, type=_scope_member_type(spec.type, member))
            self.flat.outputs.append(spec)
        elif isinstance(row, State):
            self.flat.inputs.append(row.as_input(path))
        elif isinstance(row, (Section, Row)):
            self.rows(row.rows, prefix, member=member, required=required)
        elif isinstance(row, Group):
            self.group(row, path, member, required)
        elif isinstance(row, Choice):
            self.choice(row, path, member, required)
        elif isinstance(row, TypeChoice):
            self.type_choice(row, path, member, required)
        elif isinstance(row, Described):
            raise FlattenError(f"described rows '{path}' are not flattened yet")
        else:
            assert isinstance(row, (Note, Button))

    def member_keys(self, group: Group, path: str) -> list[str]:
        if not isinstance(group.grows, str):
            raise FlattenError(f"group '{path}': {type(group.grows).__name__} is not built yet")
        found: list[str] = []
        head = path + "."
        for stored in (*self.stored_ids, *self.choices):
            if stored.startswith(head):
                key = stored[len(head) :].split(".", 1)[0]
                if key and key not in found:
                    found.append(key)
        for key in self.output_members.get(path, ()):
            if key not in found:
                found.append(key)
        assert group.keys is not None
        for key in found:
            if _key_position(group.keys, key) is None:
                raise FlattenError(
                    f"group '{path}' has a member key its naming rule cannot make: {key!r}"
                )
        if group.max_members is not None and len(found) > group.max_members:
            raise FlattenError(f"group '{path}' has more than {group.max_members} members")
        if len(found) < group.min_members and _has_required_row(group):
            raise FlattenError(
                f"{path} needs at least {group.min_members} members; the run has {len(found)}"
            )
        if len(found) > MAX_MEMBERS:
            raise FlattenError(f"group '{path}' has {len(found)} members, budget is {MAX_MEMBERS}")
        return found

    def group(self, group: Group, path: str, member: str | None, required: bool) -> None:
        for position, key in enumerate(self.member_keys(group, path)):
            member_required = required and (
                not group.optional_after_minimum or position < group.min_members
            )
            scope = member_id(path, key) if group.independent_types else member
            if group.value is not None:
                self.row(replace(group.value, id=key), f"{path}.", scope, member_required)
            else:
                assert group.member is not None
                self.rows(
                    group.member, f"{member_id(path, key)}.", member=scope, required=member_required
                )

    def choice(self, choice: Choice, path: str, member: str | None, required: bool) -> None:
        if choice.leader is not None:
            raise FlattenError(f"choice '{path}': a leader formula is not built yet")
        key = self.choices.get(path)
        if key is None:
            if choice.required:
                raise FlattenError(f"required choice '{path}' has no stored option")
            return
        option: Option | None = choice.option(key)
        if option is None:
            raise FlattenError(f"choice '{path}' has unknown option {key!r}")
        self.flat.choices.append((path, key))
        shared = {row.id: row for row in choice.shared if not isinstance(row, Row)}
        self.rows(option.rows, f"{path}.", shared, member, required)

    def type_choice(
        self, choice: TypeChoice, path: str, member: str | None, required: bool
    ) -> None:
        if choice.variants is None:
            assert choice.type is not None
            self.flat.inputs.append(
                InputSpec(
                    path, choice.type, required=False, doc=choice.tooltip, display_name=choice.label
                )
            )
            if path in self.stored_ids:
                self.rows(choice.rows, f"{path}.", member=member, required=required)
            return
        key = self.choices.get(path)
        if key is None:
            if choice.required:
                raise FlattenError(f"required type choice '{path}' has no stored variant")
            return
        variant = choice.variant(key)
        if variant is None:
            raise FlattenError(f"type choice '{path}' has unknown variant {key!r}")
        self.flat.choices.append((path, key))
        self.flat.inputs.append(
            InputSpec(
                path, variant.type, required=True, doc=choice.tooltip, display_name=choice.label
            )
        )
        self.rows((*choice.rows, *variant.rows), f"{path}.", member=member, required=required)
