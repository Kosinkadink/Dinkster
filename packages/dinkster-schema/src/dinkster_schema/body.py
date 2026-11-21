"""The rows a node declares in ``NodeSchema.body``, and the declared data they carry.

A body is an ordered tuple of rows. ``Input``, ``Output`` and ``State`` hold
values; ``Section`` and ``Row`` are layout boxes that open no scope; ``Group``,
``Choice``, ``TypeChoice`` and ``Described`` are structure that decide which
flattened inputs and outputs exist; ``Note`` and ``Button`` are layout.

Each class checks only what it can see by itself here. Checks that need the
whole declaration (scopes and ids, formulas against their scope, size limits,
lookup and step arguments) run when the schema is checked as a whole.

Lookups, pack functions and button steps are carried as declared data only:
nothing in the engine calls a lookup or runs a step yet.
"""

from __future__ import annotations

import inspect
import keyword
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Literal, cast

from .model import InputSpec, OutputProbeSpec, OutputSpec, TypeExpr, Widget

ID_PATTERN: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
"""An id every formula can name: an ASCII identifier that is not a Python keyword."""

KEY_PATTERN: Final = re.compile(r"[A-Za-z0-9_]+\Z")
"""A group member key or a choice option key, the middle part of a flattened id."""

BUTTON_ICONS: Final = frozenset(
    {
        "arrow-down",
        "arrow-left-right",
        "arrow-up",
        "check",
        "clipboard-paste",
        "copy",
        "dice-5",
        "download",
        "eraser",
        "external-link",
        "eye",
        "eye-off",
        "folder-open",
        "image",
        "info",
        "languages",
        "link",
        "list",
        "lock",
        "lock-open",
        "minus",
        "pencil",
        "play",
        "plus",
        "refresh-cw",
        "rotate-ccw",
        "save",
        "search",
        "shuffle",
        "sliders-horizontal",
        "square",
        "trash-2",
        "upload",
        "wand-sparkles",
        "x",
    }
)
"""Lucide icon names a ``Button`` may show (the names exist in lucide 1.25.0)."""

AFTER_RUN_MODES: Final = ("fixed", "increment", "decrement", "randomize")

MAX_PACK_FUNCTION_TIME_LIMIT: Final = 600.0

MAX_DEPTH: Final = 16
MAX_BOXES: Final = 512
MAX_MEMBERS: Final = 512


def check_id(value: object, what: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.match(value) or keyword.iskeyword(value):
        raise ValueError(f"{what} must be an identifier that is not a Python keyword: {value!r}")
    return value


def check_key(value: object, what: str) -> str:
    if not isinstance(value, str) or not KEY_PATTERN.match(value):
        raise ValueError(f"{what} must match [A-Za-z0-9_]+: {value!r}")
    return value


class _Unset:
    _instance: _Unset | None = None

    def __new__(cls) -> _Unset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET: Final = _Unset()
"""A ``Use`` field that does not override the shared row."""


# --- formulas and the data around them ----------------------------------------


@dataclass(frozen=True)
class Formula:
    """A formula in the language of ``dinkster_schema.formula``.

    The text is checked against the scope of the row it sits on when the
    schema is checked as a whole."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.text), str) or not self.text.strip():
            raise ValueError("a formula is non-empty text")


type TextOrFormula = str | Formula
type FlagOrFormula = bool | Formula


def _check_text_or_formula(value: object, what: str) -> None:
    if not isinstance(value, (str, Formula)):
        raise ValueError(f"{what} must be text or a Formula, got {type(value).__name__}")


def _check_flag_or_formula(value: object, what: str) -> None:
    if type(value) is not bool and not isinstance(value, Formula):
        raise ValueError(f"{what} must be a bool or a Formula, got {type(value).__name__}")


@dataclass(frozen=True)
class Numbered:
    """Member keys from a prefix and a number: ``Numbered("any_", start=1, pad=2)``
    gives ``any_01``, ``any_02``. An empty prefix gives bare numbers."""

    prefix: str
    start: int = 1
    pad: int = 0

    def __post_init__(self) -> None:
        if self.prefix:
            check_key(self.prefix, "a Numbered prefix")
        if type(self.start) is not int or self.start < 0:
            raise ValueError("a Numbered start is a whole number of at least 0")
        if type(self.pad) is not int or not 0 <= self.pad <= 8:
            raise ValueError("a Numbered pad is a whole number from 0 to 8")

    def key(self, index: int) -> str:
        """The key of the member at ``index`` counted from 0."""
        return f"{self.prefix}{self.start + index:0{self.pad}d}"


@dataclass(frozen=True)
class Named:
    """Member keys from a declared list of names."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("Named needs at least one name")
        for name in self.names:
            check_key(name, "a Named member key")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Named member keys must be unique")


@dataclass(frozen=True)
class Count:
    """A group with as many members as the formula gives, over stored values and link state."""

    formula: Formula


@dataclass(frozen=True)
class FollowList:
    """A group with one member per item of a list-valued field of the same scope."""

    field: str

    def __post_init__(self) -> None:
        check_id(self.field, "FollowList field")


type Grows = Literal["link", "filled", "button"] | Count | FollowList
GROWS_WORDS: Final = ("link", "filled", "button")


@dataclass(frozen=True)
class NamedList:
    """Options from a list several packs add to, such as ``comfy.samplers``."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.name), str) or not self.name:
            raise ValueError("NamedList needs a list name")


# --- lookups and pack functions -----------------------------------------------


@dataclass(frozen=True)
class LookupMark:
    """What ``@lookup`` records on a function."""

    watches: tuple[str, ...] = ()
    after_run: bool = False


@dataclass(frozen=True)
class PackFunctionMark:
    """What ``@pack_function`` records on a function."""

    time_limit: float = 30.0


_LOOKUP_ATTRIBUTE: Final = "__dinkster_lookup__"
_PACK_FUNCTION_ATTRIBUTE: Final = "__dinkster_pack_function__"


def _top_level_function(function: object, what: str) -> Callable[..., object]:
    """Refuse anything the pack's worker could not find again by importing it."""
    if not inspect.isfunction(function):
        raise ValueError(f"{what} must be a function defined at the top of a module")
    if function.__name__ == "<lambda>":
        raise ValueError(f"{what} cannot be a lambda")
    if function.__qualname__ != function.__name__:
        raise ValueError(
            f"{what} {function.__qualname__} must be defined at the top of a module, "
            "not inside a function or a class"
        )
    return cast("Callable[..., object]", function)


def function_name(function: Callable[..., object]) -> str:
    """The name a declared function carries on the wire: its module path and name."""
    return f"{function.__module__}.{function.__name__}"


def lookup[F: Callable[..., object]](
    function: F | None = None,
    *,
    watches: tuple[str, ...] = (),
    after_run: bool = False,
) -> F | Callable[[F], F]:
    """Mark a function the host may call while the user edits.

    ``watches`` names asset kinds whose change asks for a new answer;
    ``after_run`` asks again after each successful run of a node that uses it."""

    if type(after_run) is not bool:
        raise ValueError("lookup after_run must be a bool")
    if any(not isinstance(cast("object", kind), str) or not kind for kind in watches):
        raise ValueError("lookup watches are asset kind names")

    def mark(target: F) -> F:
        _top_level_function(target, "a lookup")
        setattr(target, _LOOKUP_ATTRIBUTE, LookupMark(tuple(watches), after_run))
        return target

    return mark if function is None else mark(function)


def pack_function[F: Callable[..., object]](
    function: F | None = None, *, time_limit: float = 30.0
) -> F | Callable[[F], F]:
    """Mark a function a ``Call`` step may run when the user presses a button."""

    if type(time_limit) not in (int, float) or not 0 < time_limit <= MAX_PACK_FUNCTION_TIME_LIMIT:
        raise ValueError(
            f"pack_function time_limit is seconds, above 0 and at most "
            f"{MAX_PACK_FUNCTION_TIME_LIMIT:g}"
        )

    def mark(target: F) -> F:
        _top_level_function(target, "a pack function")
        setattr(target, _PACK_FUNCTION_ATTRIBUTE, PackFunctionMark(float(time_limit)))
        return target

    return mark if function is None else mark(function)


def lookup_mark(function: Callable[..., object]) -> LookupMark | None:
    return cast("LookupMark | None", getattr(function, _LOOKUP_ATTRIBUTE, None))


def pack_function_mark(function: Callable[..., object]) -> PackFunctionMark | None:
    return cast("PackFunctionMark | None", getattr(function, _PACK_FUNCTION_ATTRIBUTE, None))


type Argument = str | Formula
"""A field id of the node (found as a formula's names are) or a formula."""


def _check_arguments(arguments: tuple[Argument, ...], what: str) -> None:
    for argument in cast("tuple[object, ...]", arguments):
        if isinstance(argument, str):
            for part in argument.split("."):
                check_id(part, f"{what} argument")
        elif not isinstance(argument, Formula):
            raise ValueError(f"{what} arguments are field ids or Formulas")


@dataclass(frozen=True)
class Lookup:
    """A call to a ``@lookup`` function, with the node's values as its arguments."""

    function: Callable[..., object]
    args: tuple[Argument, ...] = ()

    def __post_init__(self) -> None:
        _top_level_function(self.function, "a lookup")
        if lookup_mark(self.function) is None:
            raise ValueError(f"{function_name(self.function)} is used in Lookup without @lookup")
        _check_arguments(self.args, "Lookup")

    @property
    def name(self) -> str:
        return function_name(self.function)


type StepValue = object
"""A literal, a ``Formula`` or a ``Lookup``."""


# --- button steps -------------------------------------------------------------


@dataclass(frozen=True)
class Set:
    field: str
    value: StepValue


@dataclass(frozen=True)
class Toggle:
    field: str


@dataclass(frozen=True)
class Reset:
    field: str


@dataclass(frozen=True)
class Swap:
    a: str
    b: str


@dataclass(frozen=True)
class Append:
    field: str
    value: StepValue
    separator: str = ", "


@dataclass(frozen=True)
class Randomize:
    field: str
    low: StepValue = None
    high: StepValue = None


@dataclass(frozen=True)
class SetMode:
    field: str
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in AFTER_RUN_MODES:
            raise ValueError(f"SetMode mode is one of {', '.join(AFTER_RUN_MODES)}")


@dataclass(frozen=True)
class AddMember:
    group: str
    values: dict[str, StepValue] = field(default_factory=dict[str, StepValue])


@dataclass(frozen=True)
class RemoveMember:
    group: str
    which: Literal["first", "last"] | Formula

    def __post_init__(self) -> None:
        if self.which not in ("first", "last") and not isinstance(
            cast("object", self.which), Formula
        ):
            raise ValueError('RemoveMember which is "first", "last" or a Formula')


@dataclass(frozen=True)
class ClearMembers:
    group: str


@dataclass(frozen=True)
class SetEach:
    group: str
    field: str
    value: StepValue


@dataclass(frozen=True)
class Disconnect:
    input: str


@dataclass(frozen=True)
class AddNode:
    node_type: str
    to: str
    values: dict[str, StepValue] = field(default_factory=dict[str, StepValue])


@dataclass(frozen=True)
class Pick:
    name: str
    options: object

    def __post_init__(self) -> None:
        check_id(self.name, "Pick name")


@dataclass(frozen=True)
class Ask:
    title: str
    rows: tuple[InputSpec, ...]

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("Ask needs at least one row")
        for row in self.rows:
            if not isinstance(cast("object", row), InputSpec):
                raise ValueError("Ask rows are Input rows")


@dataclass(frozen=True)
class OpenPicker:
    field: str


@dataclass(frozen=True)
class OpenEditor:
    field: str


@dataclass(frozen=True)
class PasteImage:
    field: str


@dataclass(frozen=True)
class Call:
    """Run a ``@pack_function`` and store its answer in ``writes``, or hand a file to the user."""

    function: Callable[..., object]
    args: tuple[Argument, ...] = ()
    writes: tuple[str, ...] = ()
    download: str | None = None

    def __post_init__(self) -> None:
        _top_level_function(self.function, "a pack function")
        if pack_function_mark(self.function) is None:
            raise ValueError(
                f"{function_name(self.function)} is used in Call without @pack_function"
            )
        _check_arguments(self.args, "Call")
        if (self.download is None) == (not self.writes):
            raise ValueError("a Call either writes fields or downloads a file, not both or neither")
        if self.download is not None and not self.download:
            raise ValueError("a Call download needs a file name")

    @property
    def name(self) -> str:
        return function_name(self.function)


@dataclass(frozen=True)
class Queue:
    to_here: bool = False


@dataclass(frozen=True)
class Interrupt:
    pass


@dataclass(frozen=True)
class Copy:
    value: StepValue


@dataclass(frozen=True)
class Download:
    value: StepValue
    filename: str


@dataclass(frozen=True)
class OpenLink:
    url: StepValue


type Step = (
    Set
    | Toggle
    | Reset
    | Swap
    | Append
    | Randomize
    | SetMode
    | AddMember
    | RemoveMember
    | ClearMembers
    | SetEach
    | Disconnect
    | AddNode
    | Pick
    | Ask
    | OpenPicker
    | OpenEditor
    | PasteImage
    | Call
    | Queue
    | Interrupt
    | Copy
    | Download
    | OpenLink
)
STEP_CLASSES: Final = (
    Set,
    Toggle,
    Reset,
    Swap,
    Append,
    Randomize,
    SetMode,
    AddMember,
    RemoveMember,
    ClearMembers,
    SetEach,
    Disconnect,
    AddNode,
    Pick,
    Ask,
    OpenPicker,
    OpenEditor,
    PasteImage,
    Call,
    Queue,
    Interrupt,
    Copy,
    Download,
    OpenLink,
)
OUTSIDE_STEPS: Final = (Queue, Interrupt, Copy, Download, OpenLink)
"""Steps that act outside the workflow; they come after every step that changes it."""


def is_outside_step(step: Step) -> bool:
    return isinstance(step, OUTSIDE_STEPS) or (isinstance(step, Call) and step.download is not None)


def _check_steps(steps: tuple[Step, ...], what: str) -> None:
    seen_outside = False
    for step in steps:
        if not isinstance(cast("object", step), STEP_CLASSES):
            raise ValueError(f"{what}: {type(step).__name__} is not a step")
        if is_outside_step(step):
            seen_outside = True
        elif seen_outside:
            raise ValueError(
                f"{what}: {type(step).__name__} changes the workflow after a step that acts "
                "outside it; steps that act outside the workflow come last"
            )


@dataclass(frozen=True)
class Rule:
    """A check that must hold, its message, and an optional fix of button steps."""

    check: Formula
    message: str
    fix: tuple[Step, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.check), Formula):
            raise ValueError("a Rule's check is a Formula")
        if not isinstance(cast("object", self.message), str) or not self.message:
            raise ValueError("a Rule needs a message")
        _check_steps(self.fix, "rule fix")


# --- rows ---------------------------------------------------------------------


@dataclass(frozen=True)
class State:
    """A stored value with no pin and no field, written by an editor or a step."""

    id: str
    type: TypeExpr
    default: object = None

    def __post_init__(self) -> None:
        check_id(self.id, "state id")


@dataclass(frozen=True)
class Note:
    """Read-only text, usually a formula. ``unknown`` is shown while the formula is unknown."""

    id: str
    text: TextOrFormula | Lookup
    unknown: str = ""
    visible: FlagOrFormula = True

    def __post_init__(self) -> None:
        check_id(self.id, "note id")
        if not isinstance(cast("object", self.text), (str, Formula, Lookup)):
            raise ValueError(f"note {self.id}: text is text, a Formula or a Lookup")
        _check_flag_or_formula(self.visible, f"note {self.id} visible")


@dataclass(frozen=True)
class Button:
    """Runs its steps when pressed. In ``NodeSchema.menu`` it is an item of the node's menu."""

    id: str
    label: str
    steps: tuple[Step, ...]
    enabled: FlagOrFormula = True
    disabled_reason: TextOrFormula = ""
    icon: str | None = None
    tooltip: str = ""
    visible: FlagOrFormula = True

    def __post_init__(self) -> None:
        check_id(self.id, "button id")
        if not isinstance(cast("object", self.label), str) or not self.label:
            raise ValueError(f"button {self.id}: a button needs a label, even with an icon")
        if not self.steps:
            raise ValueError(f"button {self.id}: a button needs at least one step")
        _check_steps(self.steps, f"button {self.id}")
        _check_flag_or_formula(self.enabled, f"button {self.id} enabled")
        _check_text_or_formula(self.disabled_reason, f"button {self.id} disabled_reason")
        _check_flag_or_formula(self.visible, f"button {self.id} visible")
        if self.icon is not None and self.icon not in BUTTON_ICONS:
            raise ValueError(
                f"button {self.id}: icon {self.icon!r} is not one of "
                f"{', '.join(sorted(BUTTON_ICONS))}"
            )


@dataclass(frozen=True)
class Section:
    """A titled box that opens and closes. Layout only; it opens no scope."""

    id: str
    rows: tuple[BodyRow, ...]
    label: str = ""
    tooltip: str = ""
    visible: FlagOrFormula = True
    collapsed: bool = False

    def __post_init__(self) -> None:
        check_id(self.id, "section id")
        _check_flag_or_formula(self.visible, f"section {self.id} visible")
        if type(self.collapsed) is not bool:
            raise ValueError(f"section {self.id}: collapsed must be a bool")


@dataclass(frozen=True)
class Row:
    """Two to four of Input, Button and Note side by side."""

    rows: tuple[InputSpec | Button | Note, ...]

    def __post_init__(self) -> None:
        if not 2 <= len(self.rows) <= 4:
            raise ValueError("a Row holds two to four rows")
        for row in self.rows:
            if not isinstance(cast("object", row), (InputSpec, Button, Note)):
                raise ValueError("a Row holds Input, Button and Note rows only")


@dataclass(frozen=True)
class Use:
    """A row declared once in a choice's ``shared`` rows, placed under one option.

    It may override the row's words, widget, default and whether it is
    required in this place; for a shared choice also which of its options are
    offered (``only``) and the rows inside it (``inside``)."""

    id: str
    label: str | _Unset = UNSET
    tooltip: str | _Unset = UNSET
    widget: Widget | None | _Unset = UNSET
    default: object = UNSET
    required: bool | _Unset = UNSET
    only: tuple[str, ...] | None = None
    inside: tuple[Use, ...] = ()

    def __post_init__(self) -> None:
        check_id(self.id, "Use id")
        if self.required is not UNSET and type(self.required) is not bool:
            raise ValueError(f"Use {self.id}: required must be a bool")
        if self.only is not None:
            if not self.only:
                raise ValueError(f"Use {self.id}: only names at least one option")
            for key in self.only:
                check_key(key, f"Use {self.id} only")
        ids = [inner.id for inner in self.inside]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Use {self.id}: inside names a row twice")


@dataclass(frozen=True)
class Option:
    """One option of a choice: its key and the rows it holds."""

    key: str
    rows: tuple[BodyRow | Use, ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        check_key(self.key, "option key")


@dataclass(frozen=True)
class Choice:
    """A dropdown or a boolean leading the rows each option holds.

    With ``leader`` nothing is stored or drawn for the choice itself: the
    formula, over stored values and link state, gives the option's key."""

    id: str
    options: tuple[Option, ...]
    shared: tuple[BodyRow, ...] = ()
    leader: Formula | None = None
    default: str | None = None
    required: bool = True
    hidden: bool = False
    label: str = ""
    tooltip: str = ""
    visible: FlagOrFormula = True

    def __post_init__(self) -> None:
        check_id(self.id, "choice id")
        if not self.options:
            raise ValueError(f"choice {self.id}: needs at least one option")
        keys = [option.key for option in self.options]
        if len(set(keys)) != len(keys):
            raise ValueError(f"choice {self.id}: duplicate option keys")
        if self.default is not None and self.default not in keys:
            raise ValueError(f"choice {self.id}: default must name one of its options")
        if self.leader is not None and not isinstance(cast("object", self.leader), Formula):
            raise ValueError(f"choice {self.id}: leader is a Formula")
        for flag in ("required", "hidden"):
            if type(getattr(self, flag)) is not bool:
                raise ValueError(f"choice {self.id}: {flag} must be a bool")
        _check_flag_or_formula(self.visible, f"choice {self.id} visible")

    def option(self, key: str) -> Option | None:
        for option in self.options:
            if option.key == key:
                return option
        return None


@dataclass(frozen=True)
class Group:
    """A box that repeats its member rows.

    Exactly one of ``member`` (a group of records: each member has rows,
    flattened ``<group>.<key>.<row>``) and ``value`` (a group of values: each
    member is one ``Input`` or ``Output``, flattened ``<group>.<key>``)."""

    id: str
    member: tuple[BodyRow, ...] | None = None
    value: InputSpec | OutputSpec | None = None
    keys: Numbered | Named | None = None
    grows: Grows = "link"
    min_members: int = 0
    max_members: int | None = None
    independent_types: bool = False
    optional_after_minimum: bool = False
    reorderable: bool = False
    add_label: str = ""
    member_label: TextOrFormula = ""
    label: str = ""
    tooltip: str = ""
    visible: FlagOrFormula = True

    def __post_init__(self) -> None:
        check_id(self.id, "group id")
        if (self.member is None) == (self.value is None):
            raise ValueError(f"group {self.id}: declares exactly one of member and value")
        if self.value is not None and not isinstance(
            cast("object", self.value), (InputSpec, OutputSpec)
        ):
            raise ValueError(f"group {self.id}: value is one Input or one Output")
        if self.member is not None and not self.member:
            raise ValueError(f"group {self.id}: member needs at least one row")
        if not isinstance(cast("object", self.keys), (Numbered, Named)):
            raise ValueError(f"group {self.id}: keys is Numbered(...) or Named(...)")
        if not (
            self.grows in GROWS_WORDS or isinstance(cast("object", self.grows), (Count, FollowList))
        ):
            raise ValueError(
                f'group {self.id}: grows is "link", "filled", "button", Count or FollowList'
            )
        if type(self.min_members) is not int or not 0 <= self.min_members <= MAX_MEMBERS:
            raise ValueError(f"group {self.id}: min_members is a whole number from 0 to 512")
        if self.max_members is not None and (
            type(self.max_members) is not int
            or not max(1, self.min_members) <= self.max_members <= MAX_MEMBERS
        ):
            raise ValueError(f"group {self.id}: max_members is from min_members (and 1) to 512")
        if isinstance(self.keys, Named) and self.max_members is not None:
            if self.max_members > len(self.keys.names):
                raise ValueError(f"group {self.id}: max_members exceeds the declared names")
        for flag in ("independent_types", "optional_after_minimum", "reorderable"):
            if type(getattr(self, flag)) is not bool:
                raise ValueError(f"group {self.id}: {flag} must be a bool")
        _check_text_or_formula(self.member_label, f"group {self.id} member_label")
        _check_flag_or_formula(self.visible, f"group {self.id} visible")

    @property
    def is_values(self) -> bool:
        return self.value is not None


def _recursively_concrete(expr: TypeExpr) -> bool:
    pending = [expr]
    while pending:
        part = pending.pop()
        if part.kind == "wildcard" or part.runtime_type_id() is None:
            return False
        pending.extend(part.elements)
        if part.element is not None:
            pending.append(part.element)
    return True


@dataclass(frozen=True)
class Variant:
    """One type a ``TypeChoice`` may take, and the rows it holds while chosen."""

    key: str
    type: TypeExpr
    rows: tuple[BodyRow, ...] = ()
    tooltip: str = ""

    def __post_init__(self) -> None:
        check_key(self.key, "variant key")
        if not _recursively_concrete(self.type):
            raise ValueError(
                f"variant {self.key}: type must be recursively concrete "
                "(a variant is a type choice; use several variants for several types)"
            )


@dataclass(frozen=True)
class TypeChoice:
    """A pin whose plugged-in type leads the rows that follow it.

    Exactly one of ``variants`` (a closed list of types, the document stores
    the chosen key) and ``type`` (an open pin of that type). ``rows`` are
    present under every variant."""

    id: str
    variants: tuple[Variant, ...] | None = None
    type: TypeExpr | None = None
    rows: tuple[BodyRow, ...] = ()
    required: bool | None = None
    type_variable: str = ""
    label: str = ""
    tooltip: str = ""
    visible: FlagOrFormula = True

    def __post_init__(self) -> None:
        check_id(self.id, "type choice id")
        if (self.variants is None) == (self.type is None):
            raise ValueError(f"type choice {self.id}: declares exactly one of variants and type")
        if self.required is None:
            object.__setattr__(self, "required", self.variants is not None)
        elif type(self.required) is not bool:
            raise ValueError(f"type choice {self.id}: required must be a bool")
        if type(self.type_variable) is not str:
            raise ValueError(f"type choice {self.id}: type_variable must be text")
        _check_flag_or_formula(self.visible, f"type choice {self.id} visible")
        if self.variants is not None:
            if not self.variants:
                raise ValueError(f"type choice {self.id}: needs at least one variant")
            keys = [variant.key for variant in self.variants]
            if len(set(keys)) != len(keys):
                raise ValueError(f"type choice {self.id}: duplicate variant keys")
            if self.type_variable and not self.required:
                raise ValueError(f"type choice {self.id}: type_variable needs a required choice")
        else:
            assert self.type is not None
            if self.type_variable:
                raise ValueError(f"type choice {self.id}: type_variable needs variants")
            if self.required:
                raise ValueError(f"type choice {self.id}: an open pin is optional")
            if self.type.kind == "variable":
                raise ValueError(f"type choice {self.id}: type must not be a variable")

    def variant(self, key: str) -> Variant | None:
        for candidate in self.variants or ():
            if candidate.key == key:
                return candidate
        return None


@dataclass(frozen=True)
class Described:
    """Rows listed by a stored description, each entry one of the declared ``kinds``.

    An entry's flattened id is ``<described>.<entry id>``."""

    id: str
    kinds: tuple[InputSpec | OutputSpec, ...]
    max_entries: int
    min_entries: int = 0
    fixed_ids: bool = False
    probe: OutputProbeSpec | None = None
    label: str = ""
    tooltip: str = ""
    visible: FlagOrFormula = True

    def __post_init__(self) -> None:
        check_id(self.id, "described id")
        if type(self.min_entries) is not int or type(self.max_entries) is not int:
            raise ValueError(f"described {self.id}: entry bounds must be whole numbers")
        if not 0 <= self.min_entries <= self.max_entries <= 512:
            raise ValueError(f"described {self.id}: needs 0 <= min_entries <= max_entries <= 512")
        if type(self.fixed_ids) is not bool:
            raise ValueError(f"described {self.id}: fixed_ids must be a bool")
        if self.probe is not None and not self.fixed_ids:
            raise ValueError(f"described {self.id}: a probe needs fixed_ids")
        if not isinstance(cast("object", self.kinds), tuple) or not 1 <= len(self.kinds) <= 512:
            raise ValueError(f"described {self.id}: kinds is a tuple of 1 to 512 rows")
        input_ids: set[str] = set()
        output_ids: set[str] = set()
        for kind in self.kinds:
            if not isinstance(cast("object", kind), (InputSpec, OutputSpec)):
                raise ValueError(f"described {self.id}: a kind is an Input or an Output")
            check_id(kind.id, f"described {self.id} kind id")
            seen = input_ids if isinstance(kind, InputSpec) else output_ids
            if kind.id in seen:
                raise ValueError(f"described {self.id}: kind {kind.id} is declared twice")
            seen.add(kind.id)
            if isinstance(kind, OutputSpec) and (
                kind.represents is not None or kind.known_value is not None
            ):
                raise ValueError(f"described {self.id}: kind {kind.id} cannot carry estimates")
        _check_flag_or_formula(self.visible, f"described {self.id} visible")


type BodyRow = (
    InputSpec
    | OutputSpec
    | State
    | Section
    | Row
    | Group
    | Choice
    | TypeChoice
    | Described
    | Note
    | Button
)
"""A row of a body."""


# --- the whole body -------------------------------------------------------------

type _Box = Section | Row | Group | Choice | TypeChoice | Described


class _BodyCheck:
    """Checks that need more than one row: scopes and ids, nesting, ``Use``."""

    def __init__(self) -> None:
        self.boxes = 0

    def box(self, depth: int, what: str) -> None:
        self.boxes += 1
        if self.boxes > MAX_BOXES:
            raise ValueError(f"a body holds at most {MAX_BOXES} boxes")
        if depth > MAX_DEPTH:
            raise ValueError(f"{what} nests deeper than {MAX_DEPTH}")

    def scope(self, rows: tuple[BodyRow | Use, ...], depth: int, where: str) -> None:
        """Check one scope: the body, one group member, one choice, or one variant."""
        inputs: set[str] = set()
        outputs: set[str] = set()
        self.rows(rows, depth, where, inputs, outputs, shared={})

    def rows(
        self,
        rows: tuple[BodyRow | Use, ...],
        depth: int,
        where: str,
        inputs: set[str],
        outputs: set[str],
        shared: dict[str, BodyRow],
    ) -> None:
        for row in rows:
            if isinstance(row, Use):
                self.use(row, shared, where)
                continue
            if isinstance(row, OutputSpec):
                _claim(outputs, row.id, where, "output")
            elif isinstance(row, (InputSpec, State, Group, Choice, TypeChoice, Described)):
                _claim(inputs, row.id, where, "input")
            if isinstance(row, Section):
                self.box(depth + 1, f"section {row.id}")
                self.rows(row.rows, depth + 1, where, inputs, outputs, shared)
            elif isinstance(row, Row):
                self.box(depth + 1, "a Row")
                self.rows(row.rows, depth + 1, where, inputs, outputs, shared)
            elif isinstance(row, Group):
                self.box(depth + 1, f"group {row.id}")
                if row.member is not None:
                    self.scope(row.member, depth + 1, f"{where}{row.id}.<key>.")
            elif isinstance(row, Choice):
                self.box(depth + 1, f"choice {row.id}")
                self.choice(row, depth + 1, f"{where}{row.id}.")
            elif isinstance(row, TypeChoice):
                self.box(depth + 1, f"type choice {row.id}")
                inner = f"{where}{row.id}."
                rows_inputs: set[str] = set()
                rows_outputs: set[str] = set()
                self.rows(row.rows, depth + 1, inner, rows_inputs, rows_outputs, {})
                for variant in row.variants or ():
                    self.rows(
                        variant.rows, depth + 1, inner, set(rows_inputs), set(rows_outputs), {}
                    )
            elif isinstance(row, Described):
                self.box(depth + 1, f"described {row.id}")

    def choice(self, choice: Choice, depth: int, where: str) -> None:
        inputs: set[str] = set()
        outputs: set[str] = set()
        shared: dict[str, BodyRow] = {}
        for row in choice.shared:
            if not isinstance(row, Row):
                shared[row.id] = row
        self.rows(choice.shared, depth, where, inputs, outputs, {})
        for option in choice.options:
            used = [row.id for row in option.rows if isinstance(row, Use)]
            if len(set(used)) != len(used):
                raise ValueError(f"choice {choice.id}: option {option.key} uses a row twice")
            self.rows(option.rows, depth, where, inputs, outputs, shared)

    def use(self, use: Use, shared: dict[str, BodyRow], where: str) -> None:
        row = shared.get(use.id)
        if row is None:
            raise ValueError(f"Use {where}{use.id}: names no shared row of its choice")
        _check_use(use, row, f"{where}{use.id}")


def _claim(seen: set[str], row_id: str, where: str, kind: str) -> None:
    if row_id in seen:
        raise ValueError(f"{kind} {where}{row_id} is declared twice in one scope")
    seen.add(row_id)


def _choice_rows(choice: Choice) -> dict[str, BodyRow]:
    """The rows of a choice by id, through the layout boxes, for ``Use(inside=...)``."""
    found: dict[str, BodyRow] = {}
    pending: list[BodyRow | Use] = [*choice.shared]
    for option in choice.options:
        pending.extend(option.rows)
    while pending:
        row = pending.pop()
        if isinstance(row, (Section, Row)):
            pending.extend(row.rows)
        elif not isinstance(row, Use):
            found[row.id] = row
    return found


def _check_use(use: Use, row: BodyRow, where: str) -> None:
    if not isinstance(row, Choice):
        if use.only is not None:
            raise ValueError(f"Use {where}: only applies to a shared choice")
        if use.inside:
            raise ValueError(f"Use {where}: inside applies to a shared choice")
        return
    keys = [option.key for option in row.options]
    for key in use.only or ():
        if key not in keys:
            raise ValueError(f"Use {where}: only names {key!r}, which is not an option")
    if use.default is not UNSET and use.default is not None:
        offered = use.only if use.only is not None else tuple(keys)
        if use.default not in offered:
            raise ValueError(f"Use {where}: default {use.default!r} is not an offered option")
    rows = _choice_rows(row)
    for inner in use.inside:
        inner_row = rows.get(inner.id)
        if inner_row is None:
            raise ValueError(f"Use {where}: inside names {inner.id!r}, not a row of the choice")
        _check_use(inner, inner_row, f"{where}.{inner.id}")


def check_body(body: tuple[BodyRow, ...]) -> None:
    """Refuse a body whose ids, nesting, box count or ``Use`` rows break the contract."""
    _BodyCheck().scope(body, 0, "")
