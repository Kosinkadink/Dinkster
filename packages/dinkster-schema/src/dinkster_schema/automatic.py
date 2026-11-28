"""Working out the automatic values of one node.

A field that declares ``automatic`` is automatic while nothing is stored for
it and no wire feeds it. Its value is worked out over the node's values in
use: a stored value, the worked-out value of another automatic field, or a
default. When it cannot be worked out, the field's default is used.

A lookup form is not called here. Its arguments are worked out, and the
caller calls the lookup and works the node out again with the answers, so a
formula that reads a looked-up field reads the answer.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, cast

from .body import (
    Formula,
    Lookup,
    automatic_kind,
    automatic_value_fits,
    function_name,
)
from .flatten import (
    NOT_A_FILE,
    Flattened,
    InputSite,
    SiteEvaluator,
    flatten_body,
    site_evaluator,
)
from .formula import UNKNOWN, Failed, FormulaRefused
from .model import ComboOption, ComboWidget, Input, NamedList, WidgetRepresentations

type AutomaticReason = Literal[
    "unknown", "failed", "wrong-type", "argument-unknown", "lookup-failed", "lookup-timeout"
]
"""Why an automatic value could not be worked out, so the default was used."""

MAX_PASSES: Final = 8
"""Flattening passes: a count or a leader that reads an automatic value may
change which automatic fields exist."""


@dataclass(frozen=True)
class LookupCall:
    """One call of a declared lookup with arguments worked out from the node's values.

    Calls are equal when their function and JSON arguments are, so an argument
    that is a list or a picked file's wire object still names one call."""

    function: str
    args: tuple[object, ...] = field(compare=False)
    key: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "key", json.dumps(list(self.args), sort_keys=True, separators=(",", ":"))
        )


@dataclass(frozen=True)
class AutomaticOutcome:
    """The value an automatic field has in use.

    ``worked_out`` is False when ``value`` is the field's default, with the
    ``reason``; ``call`` is the lookup whose answer is still to come, while
    ``value`` is the default."""

    id: str
    value: object
    worked_out: bool
    reason: AutomaticReason | None = None
    call: LookupCall | None = None

    @property
    def pending(self) -> bool:
        return self.call is not None and not self.worked_out and self.reason is None


LookupAnswer = object | Failed
"""A lookup's answer, or why it gave none (``Failed("lookup-timeout")`` when it ran out of time)."""


def _choice_options(input_spec: Input) -> tuple[object, ...] | None:
    """The static options of a choice field, or None when they are not known here."""
    widgets = (
        tuple(item.widget for item in input_spec.widget.representations)
        if isinstance(input_spec.widget, WidgetRepresentations)
        else (input_spec.widget,)
    )
    for widget in widgets:
        if (
            isinstance(widget, ComboWidget)
            and not widget.remote_route
            and not widget.editable_options
            and not isinstance(widget.options, NamedList)
        ):
            return tuple(
                option.value if isinstance(option, ComboOption) else option
                for option in widget.static_options
            )
    return None


_REFUSED: Final = object()
"""A formula the checker refuses at the site."""


class _InUse(Mapping[str, object]):
    """The value in use of each automatic site, worked out when a formula reads it."""

    def __init__(self, read: Callable[[str], object], ids: Sequence[str]) -> None:
        self._read = read
        self._ids = ids

    def __contains__(self, key: object) -> bool:
        return key in self._ids

    def __getitem__(self, key: str) -> object:
        if key not in self._ids:
            raise KeyError(key)
        return self._read(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._ids)

    def __len__(self) -> int:
        return len(self._ids)


class _WorkingOut:
    def __init__(
        self,
        sites: Mapping[str, InputSite],
        answers: Mapping[LookupCall, LookupAnswer],
    ) -> None:
        self.evaluator = SiteEvaluator(
            lambda _formula, _site: UNKNOWN, lambda _name, _site: UNKNOWN
        )
        self.sites = sites
        self.answers = answers
        self.outcomes: dict[str, AutomaticOutcome] = {}
        self.working: set[str] = set()

    def in_use(self, flat_id: str) -> object:
        """What a formula reads for an automatic field: its worked-out value,
        its default when it cannot be worked out, unknown while its lookup is
        to come (or when reading it would go round in a circle)."""
        outcome = self.outcome(flat_id)
        if outcome is None or outcome.pending:
            return UNKNOWN
        return outcome.value

    def outcome(self, flat_id: str) -> AutomaticOutcome | None:
        if flat_id in self.outcomes:
            return self.outcomes[flat_id]
        if flat_id in self.working:
            return None
        self.working.add(flat_id)
        try:
            outcome = self.work_out(self.sites[flat_id])
        finally:
            self.working.discard(flat_id)
        self.outcomes[flat_id] = outcome
        return outcome

    def formula(self, formula: Formula, site: InputSite) -> object:
        """A formula's outcome at ``site``. One the checker refuses here (the
        load check passed it, so it reads a row it cannot) is ``_REFUSED``:
        the field takes its default and the run goes on."""
        try:
            return self.evaluator.formula(formula, site)
        except FormulaRefused:
            return _REFUSED

    def argument(self, argument: str | Formula, site: InputSite) -> object:
        """A lookup argument's value. A file field gives its picked file's
        wire form, which the lookup's worker binds to an ``AssetRef`` as for
        any lookup call; with no file picked it is unknown."""
        if isinstance(argument, str):
            found = self.evaluator.file(argument, site)
            if found is not NOT_A_FILE:
                return UNKNOWN if found is None else found
            argument = Formula(argument)
        return self.formula(argument, site)

    def work_out(self, site: InputSite) -> AutomaticOutcome:
        spec = site.input
        default = spec.default
        automatic = spec.automatic
        call: LookupCall | None = None
        if isinstance(automatic, Lookup):
            args: list[object] = []
            for argument in automatic.args:
                found = self.argument(argument, site)
                if found is _REFUSED:
                    return AutomaticOutcome(spec.id, default, False, "failed")
                # A lookup takes no None argument, so none is sent.
                if found is UNKNOWN or found is None or isinstance(found, Failed):
                    return AutomaticOutcome(spec.id, default, False, "argument-unknown")
                args.append(
                    list(cast("tuple[object, ...]", found)) if isinstance(found, tuple) else found
                )
            call = LookupCall(function_name(automatic.function), tuple(args))
            if call not in self.answers:
                return AutomaticOutcome(spec.id, default, False, call=call)
            answer = self.answers[call]
            if isinstance(answer, Failed):
                reason: AutomaticReason = (
                    "lookup-timeout" if answer.reason == "lookup-timeout" else "lookup-failed"
                )
                return AutomaticOutcome(spec.id, default, False, reason, call)
            value: object = answer
        else:
            assert isinstance(automatic, Formula)
            value = self.formula(automatic, site)
            if value is UNKNOWN:
                return AutomaticOutcome(spec.id, default, False, "unknown")
            if value is _REFUSED or isinstance(value, Failed):
                return AutomaticOutcome(spec.id, default, False, "failed")
        kind = automatic_kind(spec.type)
        assert kind is not None
        options = _choice_options(spec) if kind == "choice" else None
        if not automatic_value_fits(kind, value) or (options is not None and value not in options):
            return AutomaticOutcome(spec.id, default, False, "wrong-type", call)
        if kind == "float":
            value = float(value)  # type: ignore[arg-type]
        elif kind == "int":
            value = int(value)  # type: ignore[arg-type]
        return AutomaticOutcome(spec.id, value, True, call=call)


def work_out_automatic(
    body: Sequence[object],
    stored: Mapping[str, object],
    output_members: Mapping[str, Sequence[str]],
    choices: Mapping[str, str],
    *,
    tables: Mapping[str, object] | None = None,
    derived: Mapping[str, Formula] | None = None,
    linked: Callable[[object], bool] | None = None,
    literal: Callable[[object], object] | None = None,
    answers: Mapping[LookupCall, LookupAnswer] | None = None,
) -> dict[str, AutomaticOutcome]:
    """The value in use of every automatic field of one node with nothing
    stored and no wire, by flattened id.

    ``answers`` holds the answers of lookup calls made so far; an outcome
    whose ``pending`` is true names a call still to make."""
    answers = answers or {}
    previous: dict[str, object] = {}
    outcomes: dict[str, AutomaticOutcome] = {}
    for _ in range(MAX_PASSES):
        flat: Flattened = flatten_body(
            body,  # type: ignore[arg-type]
            stored,
            output_members,
            choices,
            tables=tables,
            derived=derived,
            linked=linked,
            literal=literal,
            in_use=previous,
        )
        sites = {site.input.id: site for site in flat.automatic}
        working = _WorkingOut(sites, answers)
        working.evaluator = site_evaluator(
            stored,
            output_members,
            choices,
            dict(flat.choices),
            tables=tables,
            derived=derived,
            linked=linked,
            literal=literal,
            in_use=_InUse(working.in_use, tuple(sites)),
        )
        outcomes = {}
        for flat_id in sites:
            outcome = working.outcome(flat_id)
            assert outcome is not None
            outcomes[flat_id] = outcome
        in_use = {
            flat_id: outcome.value for flat_id, outcome in outcomes.items() if not outcome.pending
        }
        if in_use == previous:
            break
        previous = in_use
    return outcomes
