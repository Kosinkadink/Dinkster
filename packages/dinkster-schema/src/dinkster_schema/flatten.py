"""Flattening: a node body tree becomes flat inputs and outputs with dotted ids.

Flattening reads only the declaration and the node's document state: its
stored input values (and which are wired), its stored output members, and its
stored choices. A member's rows are ``<group>.<key>.<row>`` in a group of
records and ``<group>.<key>`` in a group of values; a choice's rows are
``<choice>.<row>``, with the option key left out of the path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from keyword import iskeyword
from typing import Any, cast

from dinkster_values.builtin_types import CORE_STRING

from .body import (
    ID_PATTERN,
    UNSET,
    BodyRow,
    Button,
    Choice,
    Count,
    Described,
    Editor,
    Formula,
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
    choice_scope_rows,
    formula_names,
    scope_rows,
)
from .formula import UNKNOWN, FormulaScope, FormulaValues, Outcome, parse_formula
from .model import Input, Output, TypeExpr
from .output_descriptors import output_descriptor_entries

MAX_MEMBERS = 512


class FlattenError(ValueError):
    """The node's document state does not fit its declared body."""


@dataclass
class Flattened:
    inputs: list[Input]
    outputs: list[Output]
    choices: list[tuple[str, str]]
    members: list[tuple[str, tuple[str, ...]]] = field(
        default_factory=list[tuple[str, tuple[str, ...]]]
    )


def member_id(group_path: str, key: str, row: str | None = None) -> str:
    """The flattened id of a member's row (``None`` for a group of values)."""
    return f"{group_path}.{key}" if row is None else f"{group_path}.{key}.{row}"


def flatten_body(
    body: Sequence[BodyRow],
    stored: Mapping[str, object] | Sequence[str],
    output_members: Mapping[str, Sequence[str]],
    choices: Mapping[str, str],
    *,
    tables: Mapping[str, object] | None = None,
    derived: Mapping[str, Formula] | None = None,
    linked: Callable[[object], bool] | None = None,
    literal: Callable[[object], object] | None = None,
    formulas: bool = True,
) -> Flattened:
    """Flatten ``body`` against one node's document state.

    ``stored`` is the node's stored input mapping, or only its input ids when
    the values are not at hand; a count or leader formula whose values are
    missing then falls back to the stored members and choices. ``linked``
    tells a wire from a stored value and ``literal`` unwraps a stored value
    that carries its own type. ``formulas=False`` says the mapping holds only
    some values, such as the descriptions of described rows when a worker
    re-checks a run: formulas are then unknown as well."""
    flat = Flattened([], [], [])
    values = stored if isinstance(stored, Mapping) else None
    flattener = _Flattener(
        tuple(stored),
        values,
        output_members,
        choices,
        flat,
        tables=dict(tables or {}),
        derived={name: formula.text for name, formula in (derived or {}).items()},
        linked=linked or (lambda _value: False),
        literal=literal or (lambda value: value),
        formulas=formulas,
    )
    flattener.frames.append(_Frame("", tuple(body)))
    flattener.rows(body, "")
    return flat


@dataclass(frozen=True)
class _Frame:
    """One scope a formula sees: its flattened prefix, its rows, and its member."""

    prefix: str
    rows: tuple[BodyRow | Use, ...]
    index: int | None = None
    key: str | None = None


_MISSING = object()


def _scope_member_type(expr: TypeExpr, member: str) -> TypeExpr:
    if expr.kind == "variable":
        return replace(expr, template_id=f"{member}:{expr.template_id}")
    return replace(
        expr,
        element=None if expr.element is None else _scope_member_type(expr.element, member),
        elements=tuple(_scope_member_type(item, member) for item in expr.elements),
    )


def described_ids(body: Sequence[BodyRow], ids: Iterable[str]) -> set[str]:
    """The ids among ``ids`` that store the description of a ``Described`` row."""

    def describes(rows: tuple[BodyRow | Use, ...], parts: list[str]) -> bool:
        for row in scope_rows(rows):
            if not parts or _row_id(row) != parts[0]:
                continue
            if isinstance(row, Described) and len(parts) == 1:
                return True
            if isinstance(row, Group) and row.member is not None and len(parts) >= 3:
                if describes(row.member, parts[2:]):
                    return True
            elif isinstance(row, Choice) and describes(choice_scope_rows(row), parts[1:]):
                return True
            elif isinstance(row, TypeChoice):
                variants = row.variants or ()
                inner = (*row.rows, *(r for variant in variants for r in variant.rows))
                if describes(inner, parts[1:]):
                    return True
        return False

    return {flat_id for flat_id in ids if describes(tuple(body), flat_id.split("."))}


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


def apply_use(row: BodyRow, use: Use) -> BodyRow:
    """A shared row as one option places it, with that place's overrides."""
    if isinstance(row, Input):
        changes: dict[str, object] = {}
        if use.label is not UNSET:
            changes["label"] = use.label
        if use.tooltip is not UNSET:
            changes["tooltip"] = use.tooltip
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
    return row if override is None else apply_use(row, override)


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
    return any(isinstance(row, Input) and row.required for row in scope_rows(cast("Any", rows)))


def _through_layout(rows: Sequence[BodyRow | Use]) -> list[BodyRow | Use]:
    """A scope's rows with the rows of its sections and Rows in their place."""
    found: list[BodyRow | Use] = []
    for row in rows:
        if isinstance(row, (Section, Row)):
            found.extend(_through_layout(row.rows))
        else:
            found.append(row)
    return found


def _row_id(row: BodyRow | Use) -> str:
    return "" if isinstance(row, Row) else row.id


def _frozen(value: object) -> object:
    """JSON lists as the tuples a declaration holds."""
    if isinstance(value, list):
        return tuple(_frozen(item) for item in cast("list[object]", value))
    return value


class _Flattener:
    def __init__(
        self,
        stored_ids: Sequence[str],
        values: Mapping[str, object] | None,
        output_members: Mapping[str, Sequence[str]],
        choices: Mapping[str, str],
        flat: Flattened,
        *,
        tables: Mapping[str, object],
        derived: Mapping[str, str],
        linked: Callable[[object], bool],
        literal: Callable[[object], object],
        formulas: bool,
    ) -> None:
        self.stored_ids = stored_ids
        self.formulas = formulas
        self.values = values
        self.output_members = output_members
        self.choices = choices
        self.flat = flat
        self.tables = tables
        self.derived = derived
        self.linked = linked
        self.literal = literal
        self.frames: list[_Frame] = []
        self.chosen: dict[str, str] = {}

    def discovered(self, path: str) -> list[str]:
        """Member keys the document names under a group, in its order.

        The run's member list for the group comes first and gives the order,
        because a member whose only stored state is a choice or an output has
        no stored input to place it; keys found only in stored ids follow."""
        found: list[str] = list(dict.fromkeys(self.output_members.get(path, ())))
        head = path + "."
        for stored in (*self.stored_ids, *self.choices):
            if stored.startswith(head):
                key = stored[len(head) :].split(".", 1)[0]
                if key and key not in found:
                    found.append(key)
        return found

    def value(self, flat_id: str, defaults: Mapping[str, object]) -> tuple[object, bool]:
        """A stored value as a formula reads it, and whether a wire feeds it."""
        if flat_id in self.chosen:
            return self.chosen[flat_id], False
        if flat_id in self.choices:
            return self.choices[flat_id], False
        if self.values is None:
            return UNKNOWN, False
        raw = self.values.get(flat_id, _MISSING)
        if raw is _MISSING:
            return defaults.get(flat_id), False
        if self.linked(raw):
            return UNKNOWN, True
        return self.literal(raw), False

    def evaluate(self, formula: Formula) -> Outcome:
        """Work out a count or leader formula over stored values and link state."""
        if self.values is None or not self.formulas:
            return UNKNOWN
        names: dict[str, str] = {}
        where: dict[str, str] = {}
        inputs: set[str] = set()
        defaults: dict[str, object] = {}
        for frame in self.frames:
            found = formula_names(frame.rows)
            for name, kind in found.names.items():
                names[name] = kind
                where[name] = frame.prefix + name
                if name in found.inputs:
                    inputs.add(name)
                else:
                    inputs.discard(name)
            for row in scope_rows(frame.rows):
                if isinstance(row, (Input, State)):
                    defaults[frame.prefix + row.id] = row.default
        member = next((frame for frame in reversed(self.frames) if frame.index is not None), None)
        values: dict[str, object] = {}
        wired: dict[str, bool] = {}
        sizes: dict[str, int] = {}
        for name, kind in names.items():
            flat_id = where[name]
            if kind == "group":
                sizes[name] = len(self.discovered(flat_id))
            elif kind.endswith("[]"):
                group_path, row_id = flat_id.rsplit(".", 1)
                values[name] = tuple(
                    self.value(f"{group_path}.{key}.{row_id}", defaults)[0]
                    for key in self.discovered(group_path)
                )
            else:
                value, is_wired = self.value(flat_id, defaults)
                if kind != "opaque":
                    values[name] = value
                if name in inputs:
                    wired[name] = is_wired
        scope = FormulaScope(
            names=names,
            inputs=frozenset(inputs),
            tables=self.tables,
            derived=self.derived,
            in_member=member is not None,
        )
        return parse_formula(formula.text, scope).evaluate(
            FormulaValues(
                values=values,
                linked=wired,
                group_sizes=sizes,
                member_index=None if member is None else member.index,
                member_key=None if member is None else member.key,
            )
        )

    def rows(
        self,
        rows: Sequence[BodyRow | Use],
        prefix: str,
        shared: Mapping[str, BodyRow] | None = None,
        member: str | None = None,
        required: bool = True,
    ) -> None:
        # A scope's rows flatten in one canonical order, by id, so reordering
        # rows or moving one between sections leaves the flat interface and
        # its signature unchanged. Member order is the run's and is kept.
        for row in sorted(_through_layout(rows), key=_row_id):
            if isinstance(row, Use):
                assert shared is not None
                row = apply_use(shared[row.id], row)
            self.row(row, prefix, member, required)

    def row(self, row: BodyRow, prefix: str, member: str | None, required: bool) -> None:
        path = f"{prefix}{row.id}" if not isinstance(row, Row) else prefix
        if isinstance(row, Input):
            spec = replace(row, id=path, required=row.required and required)
            if member is not None:
                spec = replace(spec, type=_scope_member_type(spec.type, member))
            self.flat.inputs.append(spec)
        elif isinstance(row, Output):
            spec = replace(row, id=path)
            if member is not None:
                spec = replace(spec, type=_scope_member_type(spec.type, member))
            self.flat.outputs.append(spec)
        elif isinstance(row, State):
            if row.part_of_run:
                self.flat.inputs.append(row.as_input(path))
        elif isinstance(row, Group):
            self.group(row, path, member, required)
        elif isinstance(row, Choice):
            self.choice(row, path, member, required)
        elif isinstance(row, TypeChoice):
            self.type_choice(row, path, member, required)
        elif isinstance(row, Described):
            self.described(row, path)
        else:
            assert isinstance(row, (Note, Button, Editor))

    def member_keys(self, group: Group, path: str) -> list[str]:
        assert group.keys is not None
        if isinstance(group.grows, Count):
            return self.counted_keys(group, path)
        if not isinstance(group.grows, str):
            raise FlattenError(
                f"group '{path}' follows a list, and a field that holds a list of assets "
                "is not built yet"
            )
        found = self.discovered(path)
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

    def counted_keys(self, group: Group, path: str) -> list[str]:
        """The first N keys of the naming rule, N from the count formula.

        When the formula's values are not at hand (a worker re-checking a
        run), the members the run names stand, and must be those keys."""
        assert isinstance(group.grows, Count) and group.keys is not None
        outcome = self.evaluate(group.grows.formula)
        if outcome is UNKNOWN:
            found = self.discovered(path)
            count = len(found)
        else:
            found = None
            count = outcome if type(outcome) is int and outcome >= 0 else group.min_members
        limit = MAX_MEMBERS if group.max_members is None else group.max_members
        if isinstance(group.keys, Named):
            limit = min(limit, len(group.keys.names))
        count = min(count, limit)
        keys = [
            group.keys.key(index) if isinstance(group.keys, Numbered) else group.keys.names[index]
            for index in range(count)
        ]
        if found is not None and found != keys:
            raise FlattenError(f"group '{path}' members must be its first {count} keys")
        return keys

    def group(self, group: Group, path: str, member: str | None, required: bool) -> None:
        keys = self.member_keys(group, path)
        self.flat.members.append((path, tuple(keys)))
        for position, key in enumerate(keys):
            member_required = required and (
                not group.optional_after_minimum or position < group.min_members
            )
            scope = member_id(path, key) if group.independent_types else member
            if group.value is not None:
                self.frames.append(_Frame(f"{path}.", (), position, key))
                self.row(replace(group.value, id=key), f"{path}.", scope, member_required)
            else:
                assert group.member is not None
                prefix = f"{member_id(path, key)}."
                self.frames.append(_Frame(prefix, group.member, position, key))
                self.rows(group.member, prefix, member=scope, required=member_required)
            self.frames.pop()

    def choice(self, choice: Choice, path: str, member: str | None, required: bool) -> None:
        stored = self.choices.get(path)
        key: str | None = stored
        if choice.leader is not None:
            outcome = self.evaluate(choice.leader)
            if outcome is UNKNOWN:
                # Values not at hand: the run's recorded choice stands.
                key = stored
            elif isinstance(outcome, str) and choice.option(outcome) is not None:
                key = outcome
            elif outcome is None and not choice.required:
                key = None
            else:
                key = choice.options[0].key
            if stored is not None and outcome is not UNKNOWN and stored != key:
                raise FlattenError(f"choice '{path}' is led by a formula; {stored!r} was stored")
        if key is None:
            if choice.required:
                raise FlattenError(f"required choice '{path}' has no stored option")
            return
        option: Option | None = choice.option(key)
        if option is None:
            raise FlattenError(f"choice '{path}' has unknown option {key!r}")
        self.chosen[path] = key
        self.flat.choices.append((path, key))
        shared = {row.id: row for row in choice.shared if not isinstance(row, Row)}
        self.frames.append(_Frame(f"{path}.", choice_scope_rows(choice)))
        self.rows(option.rows, f"{path}.", shared, member, required)
        self.frames.pop()

    def described(self, described: Described, path: str) -> None:
        """The description stored under the row's own id, and one row per entry."""
        self.flat.inputs.append(
            Input(path, TypeExpr.concrete(CORE_STRING), required=False, hidden=True, pin=False)
        )
        if self.values is None:
            if any(stored.startswith(path + ".") for stored in self.stored_ids):
                raise FlattenError(f"described rows '{path}' need the stored description")
            return
        raw = self.values.get(path)
        if raw is None:
            entries: tuple[dict[str, object], ...] = ()
        elif self.linked(raw):
            raise FlattenError(f"described '{path}': the description is a stored value, not a wire")
        else:
            try:
                entries = output_descriptor_entries(self.literal(raw))
            except ValueError as error:
                raise FlattenError(f"described '{path}': {error}") from error
        if not described.min_entries <= len(entries) <= described.max_entries:
            raise FlattenError(f"described '{path}' has {len(entries)} entries, outside its bounds")
        ids: set[str] = set()
        labels: set[str] = set()
        for entry in entries:
            entry_id, label, kind_id = entry.get("id"), entry.get("label"), entry.get("kind")
            if (
                not isinstance(entry_id, str)
                or not ID_PATTERN.match(entry_id)
                or iskeyword(entry_id)
            ):
                raise FlattenError(f"described '{path}': entry id {entry_id!r} is not a row id")
            if not isinstance(label, str) or not label.strip() or len(label) > 256 or "\0" in label:
                raise FlattenError(f"described '{path}': entry {entry_id} needs a label")
            if entry_id in ids or label in labels:
                raise FlattenError(f"described '{path}': entry ids and labels must be unique")
            ids.add(entry_id)
            labels.add(label)
            kind = described.kind(kind_id) if isinstance(kind_id, str) else None
            if kind is None:
                raise FlattenError(
                    f"described '{path}': entry {entry_id} has unknown kind {kind_id!r}"
                )
            if described.fixed_ids and entry_id != kind.id:
                raise FlattenError(f"described '{path}': fixed entry ids equal their kind ids")
            flat_id = f"{path}.{entry_id}"
            extra = {
                name: value for name, value in entry.items() if name not in ("id", "label", "kind")
            }
            if isinstance(kind, Output):
                if extra:
                    raise FlattenError(
                        f"described '{path}': output entry {entry_id} takes no fields"
                    )
                self.flat.outputs.append(replace(kind, id=flat_id, label=label))
                continue
            changes: dict[str, object] = {"id": flat_id, "label": label}
            if "default" in extra:
                changes["default"] = _frozen(extra.pop("default"))
            if extra:
                widget = kind.widget
                names: set[str] = (
                    {item.name for item in fields(widget)} if is_dataclass(widget) else set()
                )
                unknown = sorted(set(extra) - names)
                if unknown:
                    raise FlattenError(
                        f"described '{path}': entry {entry_id} has fields its widget lacks: "
                        f"{', '.join(unknown)}"
                    )
                try:
                    changes["widget"] = replace(
                        cast("Any", widget),
                        **{name: _frozen(value) for name, value in extra.items()},
                    )
                except (TypeError, ValueError) as error:
                    raise FlattenError(f"described '{path}': entry {entry_id}: {error}") from error
            try:
                self.flat.inputs.append(replace(kind, **changes))
            except (TypeError, ValueError) as error:
                raise FlattenError(f"described '{path}': entry {entry_id}: {error}") from error

    def type_choice(
        self, choice: TypeChoice, path: str, member: str | None, required: bool
    ) -> None:
        if choice.variants is None:
            assert choice.type is not None
            self.flat.inputs.append(
                Input(path, choice.type, required=False, tooltip=choice.tooltip, label=choice.label)
            )
            if path in self.stored_ids:
                self.frames.append(_Frame(f"{path}.", choice.rows))
                self.rows(choice.rows, f"{path}.", member=member, required=required)
                self.frames.pop()
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
            Input(path, variant.type, required=True, tooltip=choice.tooltip, label=choice.label)
        )
        rows = (*choice.rows, *variant.rows)
        self.frames.append(_Frame(f"{path}.", rows))
        self.rows(rows, f"{path}.", member=member, required=required)
        self.frames.pop()
