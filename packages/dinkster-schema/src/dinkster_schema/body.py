"""The rows a node declares in ``NodeSchema.body``, and the declared data they carry.

A body is an ordered tuple of rows. ``Input``, ``Output`` and ``State`` hold
values; ``Section`` and ``Row`` are layout boxes that open no scope; ``Group``,
``Choice``, ``TypeChoice`` and ``Described`` are structure that decide which
flattened inputs and outputs exist; ``Note``, ``Button`` and ``Editor`` are layout.

Each class checks only what it can see by itself here. Checks that need the
whole declaration (scopes and ids, formulas against their scope, size limits,
lookup and step arguments) run when the schema is checked as a whole.

Lookups and button steps are carried as declared data: the editor runs a
button's steps, and the host calls a ``@pack_function`` for its ``Call`` step.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import keyword
import math
import re
import typing
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Final, Literal, cast, overload

from dinkster_values import ASSET_TYPE, CustomWidgetDescriptor
from dinkster_values.builtin_types import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    HISTOGRAM_TYPE,
)

from .formula import Formula as CompiledFormula
from .formula import FormulaScope, check_scope, parse_formula
from .model import (
    AFTER_RUN_MODES,
    OPTION_KEY_PATTERN,
    VARIANT_KEY_PATTERN,
    ComboOption,
    ComboWidget,
    CompositorWidget,
    CurveWidget,
    Input,
    MultiComboWidget,
    NumberWidget,
    Output,
    OutputProbeSpec,
    SaveTargetWidget,
    StringWidget,
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


def check_option_key(value: object, what: str) -> str:
    """An option key is a stored value, never part of a path: words of printable
    ASCII separated by single spaces, as ComfyUI's option names are."""
    if not isinstance(value, str) or not OPTION_KEY_PATTERN.fullmatch(value):
        raise ValueError(
            f"{what} must be printable ASCII words separated by single spaces: {value!r}"
        )
    return value


def check_variant_key(value: object, what: str) -> str:
    if not isinstance(value, str) or not VARIANT_KEY_PATTERN.fullmatch(value):
        raise ValueError(f"{what} must match [A-Za-z0-9_-]+: {value!r}")
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


def check_text_or_formula(value: object, what: str) -> None:
    if not isinstance(value, (str, Formula)):
        raise ValueError(f"{what} must be text or a Formula, got {type(value).__name__}")


def check_flag_or_formula(value: object, what: str) -> None:
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


@overload
def lookup[F: Callable[..., object]](function: F) -> F: ...
@overload
def lookup[F: Callable[..., object]](
    *, watches: tuple[str, ...] = (), after_run: bool = False
) -> Callable[[F], F]: ...
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


@typing.overload
def pack_function[F: Callable[..., object]](function: F, *, time_limit: float = 30.0) -> F: ...


@typing.overload
def pack_function[F: Callable[..., object]](
    function: None = None, *, time_limit: float = 30.0
) -> Callable[[F], F]: ...


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


class CallContext:
    """What the host hands a ``@pack_function`` that declares it as its first parameter.

    Progress and notes reach only the person who pressed the button.
    ``cancelled`` turns true when that person cancels or the time limit
    passes; the host has stopped waiting by then, and the function should
    return soon. Its answer is discarded."""

    def __init__(
        self,
        progress: Callable[[float, str], None],
        note: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> None:
        self._progress = progress
        self._note = note
        self._cancelled = cancelled

    def progress(self, fraction: float, text: str = "") -> None:
        """Report how far the work is, from 0 to 1, with an optional line of text."""
        if type(fraction) not in (int, float) or not 0 <= fraction <= 1:
            raise ValueError("progress fraction is a number from 0 to 1")
        if not isinstance(cast("object", text), str):
            raise ValueError("progress text is text")
        self._progress(float(fraction), text)

    def note(self, text: str) -> None:
        """A line shown at the button with the result, such as what was left out."""
        if not isinstance(cast("object", text), str):
            raise ValueError("a note is text")
        self._note(text)

    @property
    def cancelled(self) -> bool:
        return self._cancelled()


def takes_call_context(function: Callable[..., object]) -> bool:
    """Whether a pack function's first positional parameter is the host's ``CallContext``."""
    try:
        signature = inspect.signature(function, eval_str=True)
        annotations = typing.get_type_hints(function, include_extras=True)
    except (NameError, SyntaxError, TypeError, AttributeError):
        return False
    first = next(iter(signature.parameters.values()), None)
    return (
        first is not None
        and first.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and annotations.get(first.name) is CallContext
    )


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


type AutomaticKind = Literal["int", "float", "number", "text", "choice", "bool"]

_AUTOMATIC_KINDS: Final[Mapping[TypeExpr, AutomaticKind]] = MappingProxyType(
    {
        TypeExpr.concrete(CORE_INT): "int",
        TypeExpr.concrete(CORE_FLOAT): "float",
        TypeExpr.union(CORE_INT, CORE_FLOAT): "number",
        TypeExpr.union(CORE_FLOAT, CORE_INT): "number",
        TypeExpr.concrete(CORE_STRING): "text",
        TypeExpr.concrete(CORE_COMBO): "choice",
        TypeExpr.concrete(CORE_BOOLEAN): "bool",
    }
)

_AUTOMATIC_ANSWERS: Final[Mapping[AutomaticKind, tuple[type, ...]]] = MappingProxyType(
    {
        "int": (int,),
        "float": (float, int),
        "number": (int, float),
        "text": (str,),
        "choice": (str,),
        "bool": (bool,),
    }
)


def automatic_kind(expr: TypeExpr) -> AutomaticKind | None:
    """The kind of field an automatic value fills, or None for a type with no such field."""
    return _AUTOMATIC_KINDS.get(expr)


def automatic_value_fits(kind: AutomaticKind, value: object) -> bool:
    """Whether a worked-out value is a value of a field of this kind.

    A whole number fits a decimal field, as on a wire; a decimal fits a
    whole-number field only when it is exactly whole."""
    if kind == "bool":
        return type(value) is bool
    if kind in ("text", "choice"):
        return type(value) is str
    if kind == "int":
        return type(value) is int or (type(value) is float and value.is_integer())
    return type(value) is int or (type(value) is float and math.isfinite(value))


def check_automatic(row: Input) -> None:
    """Refuse an ``automatic`` the field cannot hold, by what the input itself shows."""
    automatic = cast("object", row.automatic)
    if automatic is None:
        return
    if not isinstance(automatic, (Formula, Lookup)):
        raise ValueError(f"input {row.id}: automatic is a Formula or a Lookup")
    if row.default is None:
        raise ValueError(
            f"input {row.id}: automatic needs a default, used when the value cannot be worked out"
        )
    kind = automatic_kind(row.type)
    if kind is None:
        raise ValueError(
            f"input {row.id}: automatic needs a whole number, decimal number, text, choice or "
            "yes or no field"
        )
    if isinstance(automatic, Lookup):
        answers = _AUTOMATIC_ANSWERS[kind]
        _check_answer(
            automatic.function,
            lambda answer: any(answer is allowed for allowed in answers),
            " or ".join(allowed.__name__ for allowed in answers),
        )


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
    values: Mapping[str, StepValue] = field(default_factory=dict[str, StepValue])

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


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
    values: Mapping[str, StepValue] = field(default_factory=dict[str, StepValue])

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


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
    """A check that must hold, its message, and an optional labeled fix of button steps."""

    check: Formula
    message: str
    fix: tuple[Step, ...] = ()
    fix_label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.check), Formula):
            raise ValueError("a Rule's check is a Formula")
        if not isinstance(cast("object", self.message), str) or not self.message:
            raise ValueError("a Rule needs a message")
        _check_steps(self.fix, "rule fix")
        if self.fix and (
            not isinstance(cast("object", self.fix_label), str) or not self.fix_label.strip()
        ):
            raise ValueError("a Rule's fix needs a non-empty label")
        if any(isinstance(cast("object", step), Call) for step in self.fix):
            raise ValueError("a Rule's fix cannot Call a pack function; a button's Call can")


# --- rows ---------------------------------------------------------------------


@dataclass(frozen=True)
class State:
    """A stored value with no pin and no field, written by an editor or a step.

    With ``part_of_run=False`` it is kept with the workflow only: a run neither
    receives it nor keys its cache on it, and no formula the engine works out
    or that decides structure may read it."""

    id: str
    type: TypeExpr
    default: object = None
    part_of_run: bool = True

    def __post_init__(self) -> None:
        check_id(self.id, "state id")
        if type(self.part_of_run) is not bool:
            raise ValueError(f"state {self.id}: part_of_run must be a bool")

    def as_input(self, path: str) -> Input:
        """The flat input a run receives: stored and sent, never drawn."""
        return Input(path, self.type, required=False, default=self.default, hidden=True, pin=False)


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
        check_flag_or_formula(self.visible, f"note {self.id} visible")


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
        check_flag_or_formula(self.enabled, f"button {self.id} enabled")
        check_text_or_formula(self.disabled_reason, f"button {self.id} disabled_reason")
        check_flag_or_formula(self.visible, f"button {self.id} visible")
        if self.icon is not None and self.icon not in BUTTON_ICONS:
            raise ValueError(
                f"button {self.id}: icon {self.icon!r} is not one of "
                f"{', '.join(sorted(BUTTON_ICONS))}"
            )


type EditorWidget = CustomWidgetDescriptor | CompositorWidget | SaveTargetWidget


@dataclass(frozen=True)
class Editor:
    """A widget with no value of its own, on its own line. Layout, like a note.

    Its widget's ``writes`` and ``reads`` name the rows it touches."""

    id: str
    widget: EditorWidget
    visible: FlagOrFormula = True

    def __post_init__(self) -> None:
        check_layout_id(self.id, "editor id")
        if not isinstance(
            cast("object", self.widget),
            (CustomWidgetDescriptor, CompositorWidget, SaveTargetWidget),
        ):
            raise ValueError(
                f"editor {self.id}: widget is a CustomWidgetDescriptor, CompositorWidget "
                "or SaveTargetWidget"
            )
        if not self.widget.writes and not self.widget.reads:
            raise ValueError(f"editor {self.id}: its widget names no row in writes or reads")
        check_flag_or_formula(self.visible, f"editor {self.id} visible")


@dataclass(frozen=True)
class Section:
    """A titled box that opens and closes. Layout only; it opens no scope."""

    id: str
    rows: tuple[BodyRow | Use, ...]
    label: str = ""
    tooltip: str = ""
    visible: FlagOrFormula = True
    collapsed: bool = False
    open_name: bool = True
    """False: no header row while open; the closed row still shows the label."""

    def __post_init__(self) -> None:
        check_layout_id(self.id, "section id")
        check_flag_or_formula(self.visible, f"section {self.id} visible")
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
                check_option_key(key, f"Use {self.id} only")
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
        check_option_key(self.key, "option key")


_OPTIONAL_CHOICE_DEFAULT: Final = (
    "an optional choice cannot declare a default; it always starts with nothing chosen. "
    "A choice that should start on an option and still offer none is a required choice "
    "with an option of its own for none"
)


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
    on_edit: tuple[Step, ...] = ()
    """Steps the host applies with the user's own change of the choice."""

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
        if not self.required and self.default is not None:
            raise ValueError(f"choice {self.id}: {_OPTIONAL_CHOICE_DEFAULT}")
        check_flag_or_formula(self.visible, f"choice {self.id} visible")

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
        check_text_or_formula(self.member_label, f"group {self.id} member_label")
        check_flag_or_formula(self.visible, f"group {self.id} visible")

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
        check_variant_key(self.key, "variant key")
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
        check_flag_or_formula(self.visible, f"type choice {self.id} visible")
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
        check_flag_or_formula(self.visible, f"described {self.id} visible")

    def kind(self, kind_id: str) -> Input | Output | None:
        return next((kind for kind in self.kinds if kind.id == kind_id), None)


type BodyRow = (
    Input
    | Output
    | State
    | Section
    | Row
    | Group
    | Choice
    | TypeChoice
    | Described
    | Note
    | Button
    | Editor
)
"""A row of a body."""


# --- the whole body -------------------------------------------------------------

type _Box = Section | Row | Group | Choice | TypeChoice | Described


type _Ids = dict[str, set[str]]
"""The ids taken in one scope, by kind: input, output, section, button, note, editor."""


def _new_ids() -> _Ids:
    kinds = ("input", "output", "section", "button", "note", "editor")
    return {kind: set[str]() for kind in kinds}


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
            elif isinstance(row, Editor):
                _claim(ids["editor"], row.id, where, "editor")
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
        for row in choice.shared:
            kind = type(row).__name__
            if isinstance(row, (Section, Row)):
                raise ValueError(
                    f"choice {where[:-1]}: a {kind} in shared is layout; write it under the "
                    "option, where the Use stands, with the Use inside it"
                )
            if isinstance(row, (Note, Button, Editor)):
                raise ValueError(
                    f"choice {where[:-1]}: a {kind} in shared holds no value to share; write "
                    "it under each option that shows it"
                )
        ids = _new_ids()
        self.rows(choice.shared, depth, where, ids, {})
        shared = shared_rows(choice)
        for option in choice.options:
            used = [row.id for row in option.rows if isinstance(row, Use)]
            if len(set(used)) != len(used):
                raise ValueError(f"choice {choice.id}: option {option.key} uses a row twice")
            # Different options may each hold a layout row with one id: only one
            # option is ever shown, and a section's saved open state is shared.
            layout = {kind: set[str]() for kind in ("section", "button", "note", "editor")}
            self.rows(option.rows, depth, where, {**ids, **layout}, shared)

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
    required = row.required if use.required is UNSET else use.required
    default = row.default if use.default is UNSET else use.default
    if not required and default is not None:
        raise ValueError(f"Use {where}: {_OPTIONAL_CHOICE_DEFAULT}")
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
    outside_run: set[str] = field(default_factory=set[str])
    """States with ``part_of_run=False`` and fields with ``sent=False``."""
    fields: set[str] = field(default_factory=set[str])
    """Names that hold an ``Input`` or a ``State``."""
    counted: set[str] = field(default_factory=set[str])
    """A group counted by a formula and its rows across members."""

    def add(
        self,
        name: str,
        kind: str,
        *,
        is_input: bool,
        is_asset: bool = False,
        outside_run: bool = False,
        is_field: bool = False,
    ) -> None:
        self.names.setdefault(name, kind)
        if is_field:
            self.fields.add(name)
        if is_input:
            self.inputs.add(name)
        if is_asset:
            self.assets.add(name)
        if outside_run:
            self.outside_run.add(name)

    def nested(self, prefix: str, inner: FormulaNames, *, across_members: bool) -> None:
        for name, kind in inner.names.items():
            if across_members:
                if kind not in ("group", "opaque") and not kind.endswith("[]"):
                    self.add(
                        f"{prefix}.{name}",
                        f"{kind}[]",
                        is_input=False,
                        outside_run=name in inner.outside_run,
                    )
                continue
            self.add(
                f"{prefix}.{name}",
                kind,
                is_input=name in inner.inputs,
                is_asset=name in inner.assets,
                outside_run=name in inner.outside_run,
                is_field=name in inner.fields,
            )
            if name in inner.counted:
                self.counted.add(f"{prefix}.{name}")


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


def declared_calls(body: tuple[BodyRow, ...], menu: tuple[Button, ...] = ()) -> Iterator[Call]:
    """Every ``Call`` step of the node's buttons, in its body at every depth and in its menu."""
    buttons = [row for row in all_rows(body) if isinstance(row, Button)]
    for button in (*buttons, *menu):
        for step in button.steps:
            if isinstance(step, Call):
                yield step


def declared_calls_with_secrets(
    body: tuple[BodyRow, ...], menu: tuple[Button, ...] = ()
) -> Iterator[tuple[Call, frozenset[int]]]:
    """Calls and their direct secret arguments, respecting each button's field scope."""

    def visit(
        rows: tuple[BodyRow | Use, ...], inherited: frozenset[str], buttons: tuple[Button, ...] = ()
    ) -> Iterator[tuple[Call, frozenset[int]]]:
        scoped = tuple(scope_rows(rows))
        secrets = (inherited - formula_names(rows).names.keys()) | {
            row.id for row in scoped if isinstance(row, Input) and row.secret
        }
        for row in (*scoped, *buttons):
            if isinstance(row, Button):
                for step in row.steps:
                    if isinstance(step, Call):
                        yield (
                            step,
                            frozenset(
                                index
                                for index, argument in enumerate(step.args)
                                if isinstance(argument, str) and argument in secrets
                            ),
                        )
            elif isinstance(row, Group):
                yield from visit(
                    (row.value,) if row.value is not None else row.member or (), secrets
                )
            elif isinstance(row, Choice):
                for option in row.options:
                    yield from visit(option_rows(row, option.key), secrets)
            elif isinstance(row, TypeChoice):
                if row.variants is None:
                    yield from visit(row.rows, secrets)
                for variant in row.variants or ():
                    yield from visit((*row.rows, *variant.rows), secrets)
            elif isinstance(row, Described):
                yield from visit(row.kinds, secrets)

    yield from visit(body, frozenset(), menu)


def replace_rows[R: BodyRow | Use](
    rows: tuple[R, ...], by_id: Mapping[str, Input]
) -> tuple[R, ...]:
    """The rows of one scope with the inputs named in ``by_id`` replaced, wherever
    a section or a Row holds them."""

    def one(row: BodyRow | Use) -> BodyRow | Use:
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

    # Each row keeps its kind, so the tuple keeps its element type.
    return cast("tuple[R, ...]", tuple(one(row) for row in rows))


@dataclass(frozen=True)
class InputGroupView:
    """A group of the body's own scope as the replacement language's input family.

    ``rows`` is ``None`` when a member holds anything but plain inputs.
    ``records`` says whether a member's row id is part of its flattened id.
    ``limit`` is the most members the group admits: its declared maximum,
    its number of names, and the engine's cap, whichever is least."""

    id: str
    rows: tuple[Input, ...] | None
    records: bool
    position_key: Callable[[int], str] | None
    min_members: int
    limit: int


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
        limit = MAX_MEMBERS if row.max_members is None else row.max_members
        return InputGroupView(
            row.id,
            rows,
            row.value is None,
            keys.key if isinstance(keys, Numbered) else keys.names.__getitem__,
            row.min_members,
            limit if isinstance(keys, Numbered) else min(limit, len(keys.names)),
        )
    return None


def shared_rows(choice: Choice) -> dict[str, BodyRow]:
    """The rows of a choice's ``shared`` by id, the rows a ``Use`` names.

    The pack-load check refuses layout rows there, so each has an id."""
    return {cast("Any", row).id: row for row in choice.shared}


def option_rows(choice: Choice, key: str) -> tuple[BodyRow, ...]:
    """The rows of one option's scope, through the layout boxes, with the
    shared rows its ``Use`` rows place."""
    option = choice.option(key)
    assert option is not None
    shared = shared_rows(choice)
    found: list[BodyRow] = []

    def walk(rows: tuple[BodyRow | Use, ...]) -> None:
        for row in rows:
            if isinstance(row, Use):
                found.append(shared[row.id])
            elif isinstance(row, (Section, Row)):
                walk(row.rows)

    walk(option.rows)
    return (*scope_rows(option.rows), *found)


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
                outside_run=not (row.part_of_run if isinstance(row, State) else row.sent),
                is_field=True,
            )
        elif isinstance(row, Choice):
            found.add(row.id, "choice", is_input=False)
            found.nested(row.id, formula_names(choice_scope_rows(row)), across_members=False)
        elif isinstance(row, Group):
            found.add(row.id, "group", is_input=False)
            if row.member is not None:
                found.nested(row.id, formula_names(row.member), across_members=True)
            if isinstance(row.grows, Count):
                head = f"{row.id}."
                found.counted.update(name for name in found.names if name.startswith(head))
                found.counted.add(row.id)
        elif isinstance(row, TypeChoice):
            found.add(row.id, "opaque", is_input=True)
            found.nested(row.id, formula_names(row.rows), across_members=False)
        elif isinstance(row, Described):
            found.add(row.id, "opaque", is_input=False)
    return found


@dataclass(frozen=True)
class _Steps:
    steps: tuple[Step, ...]
    on_edit: bool = False
    owner: str = ""


_STEP_FIELDS: Final = frozenset({"steps", "fix", "on_edit"})
ON_EDIT_STEPS: Final = (Set, Reset, Swap, SetEach)
"""The steps a field's change may apply: those that set fields and nothing else."""


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
            yield _Steps(
                cast("tuple[Step, ...]", inner),
                on_edit=item.name == "on_edit",
                owner=cast("str", getattr(row, "id", "")),
            )
        else:
            yield from _formulas_in(inner)


def _refuse_outside_run(formula: Formula, scope: FormulaScope, where: str) -> None:
    """Refuse a formula the engine works out that reads, itself or through a
    derived formula, a value that is not part of a run."""
    if not scope.outside_run:
        return
    pending = [parse_formula(formula.text, scope)]
    while pending:
        compiled = pending.pop()
        for node in ast.walk(compiled.tree):
            name = ast.unparse(node) if isinstance(node, (ast.Name, ast.Attribute)) else None
            if name in scope.outside_run:
                raise ValueError(
                    f"{where}: reads {name!r}, which is not part of a run (a State with "
                    "part_of_run=False or a field with sent=False); a value that is not part "
                    "of a run cannot decide structure or be worked out by the engine"
                )
        pending.extend(compiled.derived.values())


def _refuse_automatic_outside_run(row: Input, scope: FormulaScope) -> None:
    """The engine works an automatic value out too, so neither its formula nor
    a lookup's arguments may read a value that is not part of a run."""
    where = f"input {row.id} automatic"
    if isinstance(row.automatic, Formula):
        _refuse_outside_run(row.automatic, scope, where)
    elif isinstance(row.automatic, Lookup):
        for argument in row.automatic.args:
            if isinstance(argument, Formula):
                _refuse_outside_run(argument, scope, where)
            elif argument in scope.outside_run:
                raise ValueError(
                    f"{where}: reads {argument!r}, which is not part of a run (a State with "
                    "part_of_run=False or a field with sent=False); a value that is not part "
                    "of a run cannot be worked out by the engine"
                )


def _names_read(formula: Formula, scope: FormulaScope) -> set[str]:
    """The names a formula reads, itself or through a derived formula."""
    names: set[str] = set()
    pending = [parse_formula(formula.text, scope)]
    while pending:
        compiled = pending.pop()
        for node in ast.walk(compiled.tree):
            if isinstance(node, (ast.Name, ast.Attribute)):
                names.add(ast.unparse(node))
        pending.extend(compiled.derived.values())
    return names


def _refuse_automatic_circle(rows: tuple[BodyRow | Use, ...], scope: FormulaScope) -> None:
    """Refuse automatic fields of one scope that read each other, or one that
    reads itself: none of them could be worked out."""
    automatic = {
        row.id: row.automatic
        for row in scope_rows(rows)
        if isinstance(row, Input) and row.automatic is not None
    }
    reads: dict[str, set[str]] = {}
    for input_id, declared in automatic.items():
        if isinstance(declared, Formula):
            names = _names_read(declared, scope)
        else:
            assert isinstance(declared, Lookup)
            names = set[str]()
            for argument in declared.args:
                if isinstance(argument, Formula):
                    names |= _names_read(argument, scope)
                else:
                    names.add(argument)
        reads[input_id] = names & automatic.keys()
    done: set[str] = set()

    def visit(input_id: str, path: tuple[str, ...]) -> None:
        if input_id in path:
            circle = " -> ".join((*path[path.index(input_id) :], input_id))
            raise ValueError(
                f"input {input_id} automatic: automatic fields read each other ({circle}); "
                "none of them could be worked out"
            )
        if input_id in done:
            return
        for read in sorted(reads[input_id]):
            visit(read, (*path, input_id))
        done.add(input_id)

    for input_id in automatic:
        visit(input_id, ())


def _refuse_counted(formula: CompiledFormula, scope: FormulaScope) -> None:
    """Refuse a formula that reads, itself or through a derived formula, the
    size or the members of a group whose count is a formula."""
    if not scope.counted:
        return
    pending = [formula]
    while pending:
        compiled = pending.pop()
        for node in ast.walk(compiled.tree):
            name = ast.unparse(node) if isinstance(node, (ast.Name, ast.Attribute)) else None
            if name in scope.counted:
                group = next(
                    item
                    for item in sorted(scope.counted, key=len)
                    if f"{name}.".startswith(f"{item}.")
                )
                raise ValueError(
                    f"formula {formula.text!r} reads {name!r}: reading the size or the members "
                    f"of group {group!r}, whose count is a formula, is not built yet"
                )
        pending.extend(compiled.derived.values())


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
        self.members: dict[str, tuple[BodyRow, ...]] = {}
        """The member rows of each group a step may name here, by its id."""
        self.inputs: dict[str, Input] = {}
        """The Input rows a step may name here, by id."""

    def scope(
        self,
        rows: tuple[BodyRow | Use, ...],
        outer: FormulaScope | None,
        in_member: bool,
        *,
        check_rows: bool = True,
    ) -> FormulaScope:
        own = formula_names(rows)
        names = dict(outer.names) if outer is not None else {}
        names.update(own.names)
        inputs = set(outer.inputs) if outer is not None else set[str]()
        assets = set(outer.assets) if outer is not None else set[str]()
        outside_run = set(outer.outside_run) if outer is not None else set[str]()
        fields = set(outer.fields) if outer is not None else set[str]()
        counted = set(outer.counted) if outer is not None else set[str]()
        for name in own.names:
            inputs.discard(name)
            assets.discard(name)
            outside_run.discard(name)
            fields.discard(name)
            counted.discard(name)
        scope = FormulaScope(
            names=names,
            inputs=frozenset(inputs | own.inputs),
            assets=frozenset(assets | own.assets),
            tables=self.tables,
            derived=self.derived,
            in_member=in_member,
            outside_run=frozenset(outside_run | own.outside_run),
            fields=frozenset(fields | own.fields),
            counted=frozenset(counted | own.counted),
        )
        if not check_rows:
            return scope
        _check_curve_histograms(rows)
        outer_members = dict(self.members)
        outer_inputs = dict(self.inputs)
        for name in own.names:
            self.inputs.pop(name, None)
        for row in scope_rows(rows):
            if isinstance(row, Group):
                self.members[row.id] = (
                    row.member if row.member is not None else (cast("BodyRow", row.value),)
                )
            elif isinstance(row, Input):
                self.inputs[row.id] = row
        for row in scope_rows(rows):
            self.row(row, scope)
        for row in rows:
            if isinstance(row, Use):
                self.check(_formulas_in(row), scope)
        _refuse_automatic_circle(rows, scope)
        if outer is not None:
            # The top scope keeps its rows: the node's menu is checked in it afterwards.
            self.members = outer_members
            self.inputs = outer_inputs
        return scope

    def row(self, row: BodyRow, scope: FormulaScope) -> None:
        if isinstance(row, Group):
            self.check(_own_fields(row, frozenset({"member_label"})), scope)
            if isinstance(row.grows, Count):
                _refuse_outside_run(row.grows.formula, scope, f"group {row.id} count")
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
        if isinstance(row, Editor):
            _check_editor_names(row.widget, scope, f"editor {row.id}", own=None)
        elif isinstance(row, Input):
            for widget in _shown_widgets(row.widget):
                if isinstance(widget, (CustomWidgetDescriptor, CompositorWidget, SaveTargetWidget)):
                    _check_editor_names(widget, scope, f"input {row.id} widget", own=row.id)
                _check_widget_answers(widget)
            _refuse_automatic_outside_run(row, scope)
        if isinstance(row, Choice) and row.leader is not None:
            _refuse_outside_run(row.leader, scope, f"choice {row.id} leader")
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
                self.steps(item.steps, scope, on_edit=item.owner if item.on_edit else None)
            elif isinstance(item, Formula):
                _refuse_counted(parse_formula(item.text, scope), scope)
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
        for argument in args:
            if isinstance(argument, Formula):
                _refuse_counted(parse_formula(argument.text, scope), scope)
            elif argument in scope.counted:
                raise ValueError(
                    f"{name}: argument {argument!r} reads the members of a group whose count "
                    "is a formula, which is not built yet"
                )
        if isinstance(function, str):
            # Read back from the wire: the pack checked the fit when it loaded.
            for argument in args:
                if not isinstance(argument, Formula) and argument not in scope.names:
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
                continue
            kind = scope.names.get(argument)
            if kind is None:
                raise ValueError(f"{name}: argument {argument!r} names no row")
            if not _fits(kind, argument in scope.assets, annotation):
                raise ValueError(
                    f"{name}: field {argument!r} does not fit parameter {parameter} "
                    f"({_annotation_text(annotation)})"
                )

    def randomize(self, step: Randomize, scope: FormulaScope) -> None:
        """A number, a yes or no field, or a choice whose options the declaration writes.

        Only a number takes ``low`` and ``high``."""
        kind = scope.names[step.field]
        if step.field not in scope.fields:
            raise ValueError(f"step Randomize: {step.field!r} names no Input or State")
        if kind in ("int", "float"):
            return
        if kind == "choice":
            row = self.inputs.get(step.field)
            written = row is not None and any(
                isinstance(widget, ComboWidget)
                and isinstance(widget.options, tuple)
                and widget.options
                and not widget.remote_route
                for widget in _shown_widgets(row.widget)
            )
            if not written:
                raise ValueError(
                    f"step Randomize: {step.field!r} is a choice whose options are not written "
                    "in the declaration; Randomize picks only from written options"
                )
        elif kind != "bool":
            raise ValueError(
                f"step Randomize: {step.field!r} is a {kind} field; Randomize takes a whole or "
                "decimal number, a yes or no field, or a choice whose options are written "
                "in the declaration"
            )
        if step.low is not None or step.high is not None:
            what = "yes or no field" if kind == "bool" else "choice"
            raise ValueError(
                f"step Randomize: {step.field!r} is a {what}; low and high are only for a number"
            )

    def steps(
        self, steps: tuple[Step, ...], scope: FormulaScope, *, on_edit: str | None = None
    ) -> None:
        """Check a button's steps in order; a ``Pick`` or ``Ask`` adds names for later steps.

        ``on_edit`` names the field whose change applies the steps: they only
        set fields, and their formulas may read ``previous()``."""
        if on_edit is not None:
            for step in steps:
                if not isinstance(cast("object", step), ON_EDIT_STEPS):
                    raise ValueError(
                        f"{on_edit} on_edit: {type(step).__name__} is not allowed; a field's "
                        "change applies "
                        f"only {', '.join(cls.__name__ for cls in ON_EDIT_STEPS)}"
                    )
            scope = dataclasses.replace(scope, in_on_edit=True)
        for step in steps:
            for target in _step_fields(step):
                kind = scope.names.get(target)
                if kind is None or kind == "group" or kind.endswith("[]"):
                    raise ValueError(f"step {type(step).__name__}: {target!r} names no field")
            pin = (
                step.input
                if isinstance(step, Disconnect)
                else step.to
                if isinstance(step, AddNode)
                else None
            )
            if pin is not None and pin not in scope.inputs:
                raise ValueError(f"step {type(step).__name__}: {pin!r} names no input of this node")
            if pin is not None and (row := self.inputs.get(pin)) is not None and not row.pin:
                raise ValueError(f"step {type(step).__name__}: input {pin!r} has no pin")
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
            if isinstance(step, Randomize):
                self.randomize(step, scope)
            if isinstance(step, Call):
                self.arguments(step.function, step.args, scope, pack=True)
            elif isinstance(step, SetEach):
                # The value is worked out once per member, in that member's scope.
                member_rows = self.members.get(step.group)
                if member_rows is None:
                    raise ValueError(f"step SetEach: {step.group!r} is a group of another scope")
                member_scope = self.scope(member_rows, scope, in_member=True, check_rows=False)
                self.check(_formulas_in(step.value), member_scope)
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


def _shown_widgets(widget: object) -> tuple[object, ...]:
    if isinstance(widget, WidgetRepresentations):
        return tuple(item.widget for item in widget.representations)
    return () if widget is None else (widget,)


def _check_curve_histograms(rows: tuple[BodyRow | Use, ...]) -> None:
    """Each curve's histogram is an input of the curve's own scope that is a
    HISTOGRAM and may be left empty, and no two curves share one."""
    inputs = {row.id: row for row in scope_rows(rows) if isinstance(row, Input)}
    taken: dict[str, str] = {}
    for curve in inputs.values():
        names = {
            widget.histogram
            for widget in _shown_widgets(curve.widget)
            if isinstance(widget, CurveWidget) and widget.histogram
        }
        if len(names) > 1:
            raise ValueError(f"curve {curve.id}: its representations name more than one histogram")
        for name in names:
            _check_curve_histogram(curve.id, name, inputs, taken)


def _check_curve_histogram(
    curve: str, name: str, inputs: Mapping[str, Input], taken: dict[str, str]
) -> None:
    what = f"curve {curve}: histogram {name!r}"
    target = inputs.get(name)
    if target is None:
        raise ValueError(f"{what} names no input beside the curve")
    if target.type != TypeExpr.concrete(HISTOGRAM_TYPE):
        raise ValueError(f"{what} is not a HISTOGRAM input")
    if target.required:
        raise ValueError(f"{what} must be an input that may be left empty")
    if name in taken:
        raise ValueError(f"{what} is already the histogram of curve {taken[name]}")
    taken[name] = curve


def _check_editor_names(
    widget: EditorWidget, scope: FormulaScope, what: str, *, own: str | None
) -> None:
    """An editor's row ids exist where it stands; it writes only an Input or a State.

    A widget on an input lists only the other rows: its own needs no entry."""
    for mapping, writes in ((widget.writes, True), (widget.reads, False)):
        for part, row_id in mapping.items():
            if row_id == own:
                raise ValueError(f"{what}: {part} names its own input, which needs no entry")
            if row_id not in scope.names:
                raise ValueError(f"{what}: {part} names no row here: {row_id!r}")
            if writes and row_id not in scope.fields:
                raise ValueError(f"{what}: {part} writes {row_id!r}, which is no Input or State")


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
        annotations = typing.get_type_hints(function, include_extras=True) if pack else {}
    except (NameError, SyntaxError, TypeError, AttributeError) as exc:
        raise ValueError(f"{name}: its annotations cannot be resolved: {exc}") from None
    if pack and signature.return_annotation is inspect.Signature.empty:
        raise ValueError(f"{name}: its answer has no annotation")
    parameters: list[tuple[str, object, bool]] = []
    declared = list(signature.parameters.values())
    if pack and takes_call_context(function):
        declared = declared[1:]
    for parameter in declared:
        has_default = parameter.default is not inspect.Parameter.empty
        if parameter.annotation is CallContext:
            raise ValueError(
                f"{name}: the CallContext parameter is the first one"
                if pack
                else f"{name}: a lookup gets no CallContext; only a pack function does"
            )
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError(f"{name}: *args and **kwargs cannot be filled")
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            if not has_default:
                raise ValueError(f"{name}: keyword-only parameter {parameter.name} has no default")
            if not pack:
                continue
        annotation = annotations.get(parameter.name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            raise ValueError(f"{name}: parameter {parameter.name} has no annotation")
        if not _allowed_parameter(annotation):
            if pack:
                raise ValueError(
                    f"{name}: parameter {parameter.name} ({_annotation_text(annotation)}) is "
                    "not int, float, bool, str, a list of one of those, or AssetRef, "
                    "after an optional first CallContext"
                )
            raise ValueError(
                f"{name}: parameter {parameter.name} ({_annotation_text(annotation)}) is not "
                "int, float, bool, str, a list of one of those, or AssetRef"
            )
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            continue
        parameters.append((parameter.name, annotation, has_default))
    return parameters


def pack_function_parameters(function: Callable[..., object]) -> list[tuple[str, object, bool]]:
    """A pack function's positional parameters after its CallContext: name, annotation, default."""
    return _parameters(function, pack=True)


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


def _fits_limit(answer: object) -> bool:
    """A number, or None for no bound."""
    if answer is int or answer is float:
        return True
    members = set(typing.get_args(answer))
    return bool(members) and members <= {int, float, type(None)} and members != {type(None)}


def _check_widget_answers(widget: object) -> None:
    """The answer of each lookup a field's widget uses must fit where it is used."""
    if isinstance(widget, ComboWidget) and isinstance(widget.options, Lookup):
        _check_answer(widget.options.function, _fits_options, "a list of options")
    elif isinstance(widget, StringWidget) and isinstance(widget.completions, Lookup):
        _check_answer(widget.completions.function, _fits_options, "a list of completions")
    elif isinstance(widget, NumberWidget):
        for name in ("min", "max", "step", "offset"):
            bound = getattr(widget, name)
            if isinstance(bound, Lookup):
                _check_answer(bound.function, _fits_limit, f"a number or None for {name}")


def top_level_rows(
    body: tuple[BodyRow, ...],
) -> tuple[tuple[Input, ...], tuple[Output, ...]]:
    """The inputs and outputs of the body's own scope that every instance has."""
    inputs = tuple(
        row.as_input(row.id) if isinstance(row, State) else row
        for row in scope_rows(body)
        if (isinstance(row, Input) and row.sent) or (isinstance(row, State) and row.part_of_run)
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
    check_text_or_formula(title_suffix, "title_suffix")
    menu_ids = [button.id for button in menu]
    if len(set(menu_ids)) != len(menu_ids):
        raise ValueError("menu: two buttons share an id")
    check = _FormulaCheck(tables, derived_text)
    scope = check.scope(body, None, in_member=False)
    check.check(_formulas_in((validate, title_suffix)), scope)
    for rule in validate:
        _refuse_outside_run(rule.check, scope, "validate rule")
    for button in menu:
        if not isinstance(cast("object", button), Button):
            raise ValueError("a node's menu holds Button rows")
        check.row(button, scope)
