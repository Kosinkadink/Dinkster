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

import ast
import dataclasses
import inspect
import keyword
import re
import typing
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, cast

from dinkster_values import ASSET_TYPE, CustomWidgetDescriptor
from dinkster_values.builtin_types import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
)

from .formula import Formula as CompiledFormula
from .formula import FormulaScope, check_scope, parse_formula
from .model import (
    ComboOption,
    CompositorWidget,
    CurveWidget,
    Input,
    MultiComboWidget,
    Output,
    OutputProbeSpec,
    SaveTargetWidget,
    TypeExpr,
    Widget,
    WidgetRepresentations,
)

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


def check_layout_id(value: object, what: str) -> str:
    """A ``Section``, ``Button`` or ``Note`` id: the same characters, and it may be a keyword."""
    if not isinstance(value, str) or not ID_PATTERN.match(value):
        raise ValueError(f"{what} must be an identifier: {value!r}")
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


type DeclaredFunction = Callable[..., object] | str
"""A ``@lookup`` or ``@pack_function``: the function itself in the pack, or its
wire name (module path and function name) once read back from the schema wire."""

_FUNCTION_NAME: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+\Z")


def function_name(function: DeclaredFunction) -> str:
    """The name a declared function carries on the wire: its module path and name."""
    if isinstance(function, str):
        return function
    return f"{function.__module__}.{function.__name__}"


def _check_declared(function: object, what: str, marked: Callable[[Any], object]) -> None:
    if isinstance(function, str):
        if not _FUNCTION_NAME.match(function):
            raise ValueError(f"{what} name {function!r} is not a module path and function name")
        return
    _top_level_function(function, what)
    if marked(function) is None:
        decorator = "@lookup" if what == "a lookup" else "@pack_function"
        raise ValueError(
            f"{function_name(cast('Callable[..., object]', function))} is used without {decorator}"
        )


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

    function: DeclaredFunction
    args: tuple[Argument, ...] = ()

    def __post_init__(self) -> None:
        _check_declared(self.function, "a lookup", lookup_mark)
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
    rows: tuple[Input, ...]

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("Ask needs at least one row")
        for row in self.rows:
            if not isinstance(cast("object", row), Input):
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

    function: DeclaredFunction
    args: tuple[Argument, ...] = ()
    writes: tuple[str, ...] = ()
    download: str | None = None

    def __post_init__(self) -> None:
        _check_declared(self.function, "a pack function", pack_function_mark)
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

    def as_input(self, path: str) -> Input:
        """The flat input a run receives: stored and sent, never drawn."""
        return Input(
            path, self.type, required=False, default=self.default, hidden=True, socketless=True
        )


@dataclass(frozen=True)
class Note:
    """Read-only text, usually a formula. ``unknown`` is shown while the formula is unknown."""

    id: str
    text: TextOrFormula | Lookup
    unknown: str = ""
    visible: FlagOrFormula = True

    def __post_init__(self) -> None:
        check_layout_id(self.id, "note id")
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
        check_layout_id(self.id, "button id")
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
    open_name: bool = True
    """False: no header row while open; the closed row still shows the label."""

    def __post_init__(self) -> None:
        check_layout_id(self.id, "section id")
        _check_flag_or_formula(self.visible, f"section {self.id} visible")
        for flag in ("collapsed", "open_name"):
            if type(getattr(self, flag)) is not bool:
                raise ValueError(f"section {self.id}: {flag} must be a bool")


@dataclass(frozen=True)
class Row:
    """Two to four of Input, Button and Note side by side."""

    rows: tuple[Input | Button | Note, ...]

    def __post_init__(self) -> None:
        if not 2 <= len(self.rows) <= 4:
            raise ValueError("a Row holds two to four rows")
        for row in self.rows:
            if not isinstance(cast("object", row), (Input, Button, Note)):
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
    value: Input | Output | None = None
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
    open_name: bool = True
    """False: no header row while open; the closed row still shows the label."""

    def __post_init__(self) -> None:
        check_id(self.id, "group id")
        if (self.member is None) == (self.value is None):
            raise ValueError(f"group {self.id}: declares exactly one of member and value")
        if self.value is not None and not isinstance(cast("object", self.value), (Input, Output)):
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
        for flag in ("independent_types", "optional_after_minimum", "reorderable", "open_name"):
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
    kinds: tuple[Input | Output, ...]
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
        seen: set[str] = set()
        for kind in self.kinds:
            if not isinstance(cast("object", kind), (Input, Output)):
                raise ValueError(f"described {self.id}: a kind is an Input or an Output")
            check_id(kind.id, f"described {self.id} kind id")
            # An entry names its kind by id, so an input and an output kind
            # cannot share one.
            if kind.id in seen:
                raise ValueError(f"described {self.id}: kind {kind.id} is declared twice")
            seen.add(kind.id)
            if isinstance(kind, Output) and (
                kind.represents is not None or kind.known_value is not None
            ):
                raise ValueError(f"described {self.id}: kind {kind.id} cannot carry estimates")
        _check_flag_or_formula(self.visible, f"described {self.id} visible")

    def kind(self, kind_id: str) -> Input | Output | None:
        return next((kind for kind in self.kinds if kind.id == kind_id), None)


type BodyRow = (
    Input | Output | State | Section | Row | Group | Choice | TypeChoice | Described | Note | Button
)
"""A row of a body."""


# --- the whole body -------------------------------------------------------------

type _Box = Section | Row | Group | Choice | TypeChoice | Described


type _Ids = dict[str, set[str]]
"""The ids taken in one scope, by kind: input, output, section, button, note."""


def _new_ids() -> _Ids:
    return {kind: set[str]() for kind in ("input", "output", "section", "button", "note")}


def _holds_output(row: BodyRow | Use) -> bool:
    """Whether an Output sits anywhere inside a box, so its id is also an output id."""
    if isinstance(row, Output):
        return True
    if isinstance(row, (Section, Row)):
        return any(_holds_output(inner) for inner in row.rows)
    if isinstance(row, Group):
        inner_rows = row.member if row.member is not None else (cast("BodyRow", row.value),)
        return any(_holds_output(inner) for inner in inner_rows)
    if isinstance(row, Choice):
        return any(_holds_output(inner) for inner in choice_scope_rows(row))
    if isinstance(row, TypeChoice):
        variant_rows = (inner for variant in row.variants or () for inner in variant.rows)
        return any(_holds_output(inner) for inner in (*row.rows, *variant_rows))
    if isinstance(row, Described):
        return any(isinstance(kind, Output) for kind in row.kinds)
    return False


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
        self.rows(rows, depth, where, _new_ids(), shared={})

    def rows(
        self,
        rows: tuple[BodyRow | Use, ...],
        depth: int,
        where: str,
        ids: _Ids,
        shared: dict[str, BodyRow],
    ) -> None:
        for row in rows:
            if isinstance(row, Use):
                self.use(row, shared, where)
                continue
            if isinstance(row, Output):
                _claim(ids["output"], row.id, where, "output")
            elif isinstance(row, (Input, State, Group, Choice, TypeChoice, Described)):
                _claim(ids["input"], row.id, where, "input")
                if _holds_output(row):
                    _claim(ids["output"], row.id, where, "output")
            elif isinstance(row, Section):
                _claim(ids["section"], row.id, where, "section")
            elif isinstance(row, Button):
                _claim(ids["button"], row.id, where, "button")
            elif isinstance(row, Note):
                _claim(ids["note"], row.id, where, "note")
            if isinstance(row, Section):
                self.box(depth + 1, f"section {row.id}")
                self.rows(row.rows, depth + 1, where, ids, shared)
            elif isinstance(row, Row):
                self.box(depth + 1, "a Row")
                self.rows(row.rows, depth + 1, where, ids, shared)
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
                rows_ids = _new_ids()
                self.rows(row.rows, depth + 1, inner, rows_ids, {})
                for variant in row.variants or ():
                    variant_ids = {kind: set(seen) for kind, seen in rows_ids.items()}
                    self.rows(variant.rows, depth + 1, inner, variant_ids, {})
            elif isinstance(row, Described):
                self.box(depth + 1, f"described {row.id}")

    def choice(self, choice: Choice, depth: int, where: str) -> None:
        ids = _new_ids()
        shared: dict[str, BodyRow] = {}
        for row in choice.shared:
            if not isinstance(row, Row):
                shared[row.id] = row
        self.rows(choice.shared, depth, where, ids, {})
        for option in choice.options:
            used = [row.id for row in option.rows if isinstance(row, Use)]
            if len(set(used)) != len(used):
                raise ValueError(f"choice {choice.id}: option {option.key} uses a row twice")
            self.rows(option.rows, depth, where, ids, shared)

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


_KINDS: Final = {
    CORE_INT: "int",
    CORE_FLOAT: "float",
    CORE_BOOLEAN: "bool",
    CORE_STRING: "text",
    CORE_COMBO: "choice",
}

_CHILD_FIELDS: Final = frozenset(
    {"rows", "member", "value", "options", "shared", "variants", "kinds"}
)
"""Fields that hold rows of their own; their formulas are checked with those rows."""


_OPAQUE_WIDGETS: Final = (SaveTargetWidget, CurveWidget, CompositorWidget, CustomWidgetDescriptor)


def _row_kind(row: Input | State) -> str:
    """A row's kind in a formula: its widget decides for a list of choices and opaque editors."""
    if isinstance(row, Input) and row.widget is not None:
        widgets = (
            tuple(item.widget for item in row.widget.representations)
            if isinstance(row.widget, WidgetRepresentations)
            else (row.widget,)
        )
        if any(isinstance(widget, _OPAQUE_WIDGETS) for widget in widgets):
            return "opaque"
        if any(isinstance(widget, MultiComboWidget) for widget in widgets):
            return "text[]"
    return _value_kind(row.type)


def _value_kind(expr: TypeExpr) -> str:
    if expr.kind == "concrete":
        return _KINDS.get(expr.types[0], "opaque")
    return "opaque"


@dataclass
class FormulaNames:
    names: dict[str, str] = field(default_factory=dict[str, str])
    inputs: set[str] = field(default_factory=set[str])
    assets: set[str] = field(default_factory=set[str])

    def add(self, name: str, kind: str, *, is_input: bool, is_asset: bool = False) -> None:
        self.names.setdefault(name, kind)
        if is_input:
            self.inputs.add(name)
        if is_asset:
            self.assets.add(name)

    def nested(self, prefix: str, inner: FormulaNames, *, across_members: bool) -> None:
        for name, kind in inner.names.items():
            if across_members:
                if kind not in ("group", "opaque") and not kind.endswith("[]"):
                    self.add(f"{prefix}.{name}", f"{kind}[]", is_input=False)
                continue
            self.add(
                f"{prefix}.{name}",
                kind,
                is_input=name in inner.inputs,
                is_asset=name in inner.assets,
            )


def scope_rows(rows: tuple[BodyRow | Use, ...]) -> Iterator[BodyRow]:
    """The rows of one scope, through the layout boxes."""
    for row in rows:
        if isinstance(row, (Section, Row)):
            yield from scope_rows(row.rows)
        elif not isinstance(row, Use):
            yield row


def all_rows(rows: tuple[BodyRow | Use, ...]) -> Iterator[BodyRow]:
    """Every row of a body at every depth: boxes, then the rows inside them."""
    for row in scope_rows(rows):
        yield row
        if isinstance(row, Group):
            yield from all_rows((row.value,) if row.value is not None else row.member or ())
        elif isinstance(row, Choice):
            yield from all_rows(choice_scope_rows(row))
        elif isinstance(row, TypeChoice):
            yield from all_rows(row.rows)
            for variant in row.variants or ():
                yield from all_rows(variant.rows)
        elif isinstance(row, Described):
            yield from all_rows(row.kinds)


def replace_rows(rows: tuple[BodyRow, ...], by_id: Mapping[str, Input]) -> tuple[BodyRow, ...]:
    """The rows of one scope with the inputs named in ``by_id`` replaced, wherever
    a section or a Row holds them."""

    def one(row: BodyRow) -> BodyRow:
        if isinstance(row, Section):
            return replace(row, rows=replace_rows(row.rows, by_id))
        if isinstance(row, Row):
            return replace(
                row,
                rows=tuple(
                    by_id.get(item.id, item) if isinstance(item, Input) else item
                    for item in row.rows
                ),
            )
        if isinstance(row, Input):
            return by_id.get(row.id, row)
        return row

    return tuple(one(row) for row in rows)


@dataclass(frozen=True)
class InputGroupView:
    """A group of the body's own scope as the replacement language's input family.

    ``rows`` is ``None`` when a member holds anything but plain inputs.
    ``records`` says whether a member's row id is part of its flattened id."""

    id: str
    rows: tuple[Input, ...] | None
    records: bool
    position_key: Callable[[int], str] | None


def input_group_view(body: tuple[BodyRow, ...], group_id: str) -> InputGroupView | None:
    for row in scope_rows(body):
        if not isinstance(row, Group) or row.id != group_id:
            continue
        member = (row.value,) if row.value is not None else tuple(row.member or ())
        flat = tuple(scope_rows(member))
        rows = (
            tuple(item for item in flat if isinstance(item, Input))
            if all(isinstance(item, (Input, Output)) for item in flat)
            else None
        )
        keys = row.keys
        assert keys is not None
        return InputGroupView(
            row.id,
            rows,
            row.value is None,
            keys.key if isinstance(keys, Numbered) else keys.names.__getitem__,
        )
    return None


def choice_scope_rows(choice: Choice) -> tuple[BodyRow | Use, ...]:
    return (*choice.shared, *(row for option in choice.options for row in option.rows))


def formula_names(rows: tuple[BodyRow | Use, ...]) -> FormulaNames:
    """What a formula may name in one scope, from that scope's own rows."""
    found = FormulaNames()
    for row in scope_rows(rows):
        if isinstance(row, (Input, State)):
            found.add(
                row.id,
                _row_kind(row),
                is_input=isinstance(row, Input),
                is_asset=row.type.kind == "asset" or row.type == TypeExpr.concrete(ASSET_TYPE),
            )
        elif isinstance(row, Choice):
            found.add(row.id, "choice", is_input=False)
            found.nested(row.id, formula_names(choice_scope_rows(row)), across_members=False)
        elif isinstance(row, Group):
            found.add(row.id, "group", is_input=False)
            if row.member is not None:
                found.nested(row.id, formula_names(row.member), across_members=True)
        elif isinstance(row, TypeChoice):
            found.add(row.id, "opaque", is_input=True)
            found.nested(row.id, formula_names(row.rows), across_members=False)
        elif isinstance(row, Described):
            found.add(row.id, "opaque", is_input=False)
    return found


@dataclass(frozen=True)
class _Steps:
    steps: tuple[Step, ...]


_STEP_FIELDS: Final = frozenset({"steps", "fix"})


def _formulas_in(value: object) -> Iterator[Formula | Lookup | _Steps]:
    """Formulas, lookups and step lists in a row's own fields, not in the rows it holds."""
    if isinstance(value, (Formula, Lookup)):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in cast("tuple[object, ...]", value):
            yield from _formulas_in(item)
    elif isinstance(value, Mapping):
        for item in cast("Mapping[object, object]", value).values():
            yield from _formulas_in(item)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        yield from _own_fields(value, rows_apart=False)


def _own_fields(
    row: object, skip: frozenset[str] = frozenset(), *, rows_apart: bool = True
) -> Iterator[Formula | Lookup | _Steps]:
    """With ``rows_apart``, the fields holding rows are left to the scope those rows open."""
    for item in dataclasses.fields(cast("Any", row)):
        if (rows_apart and item.name in _CHILD_FIELDS) or item.name in skip:
            continue
        inner = getattr(row, item.name)
        if item.name in _STEP_FIELDS:
            yield _Steps(cast("tuple[Step, ...]", inner))
        else:
            yield from _formulas_in(inner)


def _calls_key(formula: CompiledFormula) -> bool:
    """Whether a formula, or a derived formula it uses, calls ``key()``."""
    for node in ast.walk(formula.tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "key":
            return True
    return any(_calls_key(derived) for derived in formula.derived.values())


class _FormulaCheck:
    """Checks every formula and lookup argument against the scope of its row."""

    def __init__(self, tables: Mapping[str, object], derived: Mapping[str, str]) -> None:
        self.tables = tables
        self.derived = derived

    def scope(
        self, rows: tuple[BodyRow | Use, ...], outer: FormulaScope | None, in_member: bool
    ) -> FormulaScope:
        own = formula_names(rows)
        names = dict(outer.names) if outer is not None else {}
        names.update(own.names)
        inputs = set(outer.inputs) if outer is not None else set[str]()
        assets = set(outer.assets) if outer is not None else set[str]()
        for name in own.names:
            inputs.discard(name)
            assets.discard(name)
        scope = FormulaScope(
            names=names,
            inputs=frozenset(inputs | own.inputs),
            assets=frozenset(assets | own.assets),
            tables=self.tables,
            derived=self.derived,
            in_member=in_member,
        )
        for row in scope_rows(rows):
            self.row(row, scope)
        for row in rows:
            if isinstance(row, Use):
                self.check(_formulas_in(row), scope)
        return scope

    def row(self, row: BodyRow, scope: FormulaScope) -> None:
        if isinstance(row, Group):
            self.check(_own_fields(row, frozenset({"member_label"})), scope)
            member_rows: tuple[BodyRow, ...] = (
                row.member if row.member is not None else (cast("BodyRow", row.value),)
            )
            member_scope = self.scope(member_rows, scope, in_member=True)
            self.check(_formulas_in(row.member_label), member_scope)
            if isinstance(row.keys, Numbered) and isinstance(row.member_label, Formula):
                if _calls_key(parse_formula(row.member_label.text, member_scope)):
                    raise ValueError(
                        f"group {row.id}: a numbered group's member_label cannot use key(); "
                        "a member is shown by its position, index() + 1"
                    )
            return
        self.check(_own_fields(row), scope)
        if isinstance(row, Note) and isinstance(row.text, Lookup):
            _check_answer(row.text.function, lambda answer: answer is str, "str")
        if isinstance(row, Choice):
            for option in row.options:
                self.check(_own_fields(option), scope)
            self.scope(choice_scope_rows(row), scope, scope.in_member)
        elif isinstance(row, TypeChoice):
            self.scope(row.rows, scope, scope.in_member)
            for variant in row.variants or ():
                self.scope((*row.rows, *variant.rows), scope, scope.in_member)
        elif isinstance(row, Described):
            for kind in row.kinds:
                self.check(_own_fields(kind), scope)

    def check(self, found: Iterator[Formula | Lookup | _Steps], scope: FormulaScope) -> None:
        for item in found:
            if isinstance(item, _Steps):
                self.steps(item.steps, scope)
            elif isinstance(item, Formula):
                parse_formula(item.text, scope)
            else:
                self.arguments(item.function, item.args, scope, pack=False)

    def arguments(
        self,
        function: DeclaredFunction,
        args: tuple[Argument, ...],
        scope: FormulaScope,
        *,
        pack: bool,
    ) -> None:
        name = function_name(function)
        if isinstance(function, str):
            # Read back from the wire: the pack checked the fit when it loaded.
            for argument in args:
                if isinstance(argument, Formula):
                    parse_formula(argument.text, scope)
                elif argument not in scope.names:
                    raise ValueError(f"{name}: argument {argument!r} names no row")
            return
        parameters = _parameters(function, pack=pack)
        required = sum(1 for _, _, has_default in parameters if not has_default)
        if not required <= len(args) <= len(parameters):
            raise ValueError(
                f"{name}: takes {required} to {len(parameters)} arguments, given {len(args)}"
            )
        for argument, (parameter, annotation, _) in zip(args, parameters, strict=False):
            if isinstance(argument, Formula):
                parse_formula(argument.text, scope)
                continue
            kind = scope.names.get(argument)
            if kind is None:
                raise ValueError(f"{name}: argument {argument!r} names no row")
            if not _fits(kind, argument in scope.assets, annotation):
                raise ValueError(
                    f"{name}: field {argument!r} does not fit parameter {parameter} "
                    f"({_annotation_text(annotation)})"
                )

    def steps(self, steps: tuple[Step, ...], scope: FormulaScope) -> None:
        """Check a button's steps in order; a ``Pick`` or ``Ask`` adds names for later steps."""
        for step in steps:
            for target in _step_fields(step):
                kind = scope.names.get(target)
                if kind is None or kind == "group" or kind.endswith("[]"):
                    raise ValueError(f"step {type(step).__name__}: {target!r} names no field")
            group = _step_group(step)
            if group is not None:
                if scope.names.get(group) != "group":
                    raise ValueError(f"step {type(step).__name__}: {group!r} names no group")
                members = (
                    (step.field,)
                    if isinstance(step, SetEach)
                    else tuple(step.values)
                    if isinstance(step, AddMember)
                    else ()
                )
                for member in members:
                    if f"{group}.{member}" not in scope.names:
                        raise ValueError(
                            f"step {type(step).__name__}: {member!r} is no row of group {group}"
                        )
            if isinstance(step, Call):
                self.arguments(step.function, step.args, scope, pack=True)
            else:
                if isinstance(step, Pick) and isinstance(step.options, Lookup):
                    _check_answer(step.options.function, _fits_options, "a list of options")
                self.check(_formulas_in(step), scope)
            added = (
                {step.name: "choice"}
                if isinstance(step, Pick)
                else {row.id: _value_kind(row.type) for row in step.rows}
                if isinstance(step, Ask)
                else {}
            )
            for name in added:
                if name in scope.names:
                    raise ValueError(
                        f"step {type(step).__name__}: {name!r} is also the id of a field"
                    )
            if added:
                scope = dataclasses.replace(scope, names={**scope.names, **added})


def _step_fields(step: Step) -> tuple[str, ...]:
    """The fields a step reads or writes by id."""
    if isinstance(step, Swap):
        return (step.a, step.b)
    if isinstance(step, Disconnect):
        return (step.input,)
    if isinstance(step, Call):
        return step.writes
    if isinstance(
        step,
        (Set, Toggle, Reset, Append, Randomize, SetMode, OpenPicker, OpenEditor, PasteImage),
    ):
        return (step.field,)
    return ()


def _step_group(step: Step) -> str | None:
    if isinstance(step, (AddMember, RemoveMember, ClearMembers, SetEach)):
        return step.group
    return None


ASSET_REF_CLASS: Final = ("dinkster_assets.model", "AssetRef")
"""The asset class a lookup parameter names, matched by module and name (no dependency)."""

_SCALARS: Final = {"int": (int, float), "float": (float,), "bool": (bool,), "text": (str,)}


def _annotation_text(annotation: object) -> str:
    if isinstance(annotation, type):
        return annotation.__qualname__
    return repr(annotation)


def _is_asset_ref(annotation: object) -> bool:
    return (
        isinstance(annotation, type)
        and (annotation.__module__, annotation.__qualname__) == ASSET_REF_CLASS
    )


def _fits(kind: str, is_asset: bool, annotation: object) -> bool:
    """Whether a field of this formula kind may be handed to a parameter so annotated."""
    if is_asset:
        return _is_asset_ref(annotation)
    if kind.endswith("[]"):
        if typing.get_origin(annotation) is not list:
            return False
        (item,) = typing.get_args(annotation) or (None,)
        return _fits(kind[:-2], False, item)
    allowed = _SCALARS.get("text" if kind == "choice" else kind, ())
    return any(annotation is candidate for candidate in allowed)


def _allowed_parameter(annotation: object) -> bool:
    if any(annotation is scalar for scalar in (int, float, bool, str)) or _is_asset_ref(annotation):
        return True
    if typing.get_origin(annotation) is list:
        return typing.get_args(annotation) in ((int,), (float,), (bool,), (str,))
    return False


def _parameters(function: Callable[..., object], *, pack: bool) -> list[tuple[str, object, bool]]:
    """The positional parameters of a lookup or pack function, with their annotations."""
    name = function_name(function)
    try:
        signature = inspect.signature(function, eval_str=True)
    except (NameError, SyntaxError, TypeError, AttributeError) as exc:
        raise ValueError(f"{name}: its annotations cannot be resolved: {exc}") from None
    parameters: list[tuple[str, object, bool]] = []
    for parameter in signature.parameters.values():
        has_default = parameter.default is not inspect.Parameter.empty
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError(f"{name}: *args and **kwargs cannot be filled")
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            if not has_default:
                raise ValueError(f"{name}: keyword-only parameter {parameter.name} has no default")
            continue
        annotation = parameter.annotation
        if annotation is inspect.Parameter.empty:
            raise ValueError(f"{name}: parameter {parameter.name} has no annotation")
        if not _allowed_parameter(annotation):
            if pack:
                raise ValueError(
                    f"{name}: parameter {parameter.name} ({_annotation_text(annotation)}) is "
                    "not int, float, bool, str, a list of one of those, or AssetRef. Progress, "
                    "cancelling and secrets for pack functions are not built yet"
                )
            raise ValueError(
                f"{name}: parameter {parameter.name} ({_annotation_text(annotation)}) is not "
                "int, float, bool, str, a list of one of those, or AssetRef"
            )
        parameters.append((parameter.name, annotation, has_default))
    return parameters


def _check_answer(
    function: DeclaredFunction, fits: Callable[[object], bool], expected: str
) -> None:
    if isinstance(function, str):
        return
    name = function_name(function)
    try:
        answer = inspect.signature(function, eval_str=True).return_annotation
    except (NameError, SyntaxError, TypeError, AttributeError) as exc:
        raise ValueError(f"{name}: its annotations cannot be resolved: {exc}") from None
    if answer is inspect.Signature.empty:
        raise ValueError(f"{name}: its answer has no annotation")
    if not fits(answer):
        raise ValueError(f"{name}: answers {_annotation_text(answer)} where {expected} is used")


def _fits_options(answer: object) -> bool:
    if typing.get_origin(answer) is not list:
        return False
    return typing.get_args(answer) in ((str,), (int,), (float,), (ComboOption,))


def top_level_rows(
    body: tuple[BodyRow, ...],
) -> tuple[tuple[Input, ...], tuple[Output, ...]]:
    """The inputs and outputs of the body's own scope that every instance has."""
    inputs = tuple(
        row.as_input(row.id) if isinstance(row, State) else row
        for row in scope_rows(body)
        if isinstance(row, (Input, State))
    )
    outputs = tuple(row for row in scope_rows(body) if isinstance(row, Output))
    return inputs, outputs


def check_body(
    body: tuple[BodyRow, ...],
    *,
    tables: Mapping[str, object] | None = None,
    derived: Mapping[str, Formula] | None = None,
    validate: tuple[Rule, ...] = (),
    title_suffix: TextOrFormula = "",
    menu: tuple[Button, ...] = (),
) -> None:
    """Refuse a declaration whose body, node rules, title suffix or menu break the contract.

    Checks ids, nesting, box count and ``Use`` rows, then every formula,
    lookup and step against the scope of the row it sits on; the node's
    rules, title suffix and menu sit in the body's scope."""
    _BodyCheck().scope(body, 0, "")
    tables = dict(tables or {})
    derived_text: dict[str, str] = {}
    for name, formula in (derived or {}).items():
        check_id(name, "derived formula name")
        if not isinstance(cast("object", formula), Formula):
            raise ValueError(f"derived {name}: is a Formula")
        derived_text[name] = formula.text
    for name in tables:
        check_id(name, "table name")
    top = formula_names(body)
    check_scope(FormulaScope(names=top.names, tables=tables, derived=derived_text))
    for rule in validate:
        if not isinstance(cast("object", rule), Rule):
            raise ValueError("a node's validate holds Rule rows")
    _check_text_or_formula(title_suffix, "title_suffix")
    menu_ids = [button.id for button in menu]
    if len(set(menu_ids)) != len(menu_ids):
        raise ValueError("menu: two buttons share an id")
    check = _FormulaCheck(tables, derived_text)
    scope = check.scope(body, None, in_member=False)
    check.check(_formulas_in((validate, title_suffix)), scope)
    for button in menu:
        if not isinstance(cast("object", button), Button):
            raise ValueError("a node's menu holds Button rows")
        check.row(button, scope)
