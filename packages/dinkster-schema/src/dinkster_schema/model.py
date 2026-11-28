"""Node schema model - the single source of truth (hazard H1).

The engine, validator, cache-key builder, wire encoder, and isolation layer
all consume these objects. There is no INPUT_TYPES-style dict anywhere.

The type-expression model is deliberately small and closed (one canonical
representation, mirroring Dinkster-Frontend's TypeExpr):
concrete | union | wildcard | variable | list | asset | stream | tuple.

Structured kinds use a recursive ``element`` expression instead of flat
type ids. Lists (DESIGN 3.13): at runtime a list value's
type id is the canonical parametric string ``list<element_type_id>``
(runtime lists always have a concrete element type); the expression level
is where ``list<T>`` with a template-variable element lives. The two never
mix: ``TypeExpr.concrete("list<...>")`` is rejected so a list type has
exactly one schema representation.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast

# The list and asset type-id grammars are owned by dinkster-values (type ids
# are registry namespace); importing them here is the one place schema
# reaches sideways in the bottom layer, so expression-level types and
# runtime type ids can never drift apart.
from dinkster_values import (
    ASSET_TYPE,
    COLOR_TYPE,
    COMPOSITOR_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    CURVE_TYPE,
    MOUNT_TYPE,
    SAVE_TARGET_TYPE,
    CustomWidgetDescriptor,
    asset_type_id,
    list_type_id,
    parse_asset_type_id,
    parse_list_type_id,
)
from dinkster_values import (
    runtime_type_atom as _runtime_type_atom,
)
from dinkster_values.model import editor_row_names
from dinkster_values.streams import parse_stream_type_id, stream_type_id
from dinkster_values.tuples import parse_tuple_type_id, tuple_type_id

from .names import validate_name
from .replace import ReplacementRule

if TYPE_CHECKING:
    from .body import (
        BodyRow,
        Button,
        FlagOrFormula,
        Formula,
        Lookup,
        Rule,
        Step,
        TextOrFormula,
    )

TypeExprKind = Literal[
    "concrete",
    "union",
    "wildcard",
    "variable",
    "list",
    "asset",
    "stream",
    "tuple",
    "tuple_element",
]

Cardinality = Literal["scalar", "list", "unknown"]

_COMBO_CHOICE_MAX_ENTRIES = 10_000
_COMBO_CHOICE_MAX_VALUE_BYTES = 4_096
_COMBO_CHOICE_MAX_RESPONSE_BYTES = 2_097_152
_REMOTE_CHOICE_ROUTE_PREFIX = "/api/choices/"


def _remote_choice_id(route: str) -> str:
    """Return the choice id from the one canonical remote COMBO route."""
    choice_id = route.removeprefix(_REMOTE_CHOICE_ROUTE_PREFIX)
    if (
        not route.startswith(_REMOTE_CHOICE_ROUTE_PREFIX)
        or route != _REMOTE_CHOICE_ROUTE_PREFIX + choice_id
        or validate_name(choice_id) is not None
    ):
        raise ValueError(
            "combo widget remote route must be canonical "
            f"/api/choices/{{choice_id}}, got: {route!r}"
        )
    return choice_id


def combo_choices_json_bytes(values: Sequence[object], *, subject: str) -> bytes:
    """Validate one provider-ordered choice list and encode its exact body."""
    if len(values) > _COMBO_CHOICE_MAX_ENTRIES:
        raise ValueError(
            f"{subject} has {len(values)} entries; maximum is {_COMBO_CHOICE_MAX_ENTRIES}"
        )
    checked: list[str] = []
    seen: set[str] = set()
    for index, raw_value in enumerate(values):
        if not isinstance(raw_value, str) or not raw_value:
            raise ValueError(f"{subject} value {index} must be a non-empty string")
        if "\0" in raw_value:
            raise ValueError(f"{subject} value {index} must not contain NUL")
        try:
            encoded = raw_value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{subject} value {index} must be valid UTF-8") from exc
        if len(encoded) > _COMBO_CHOICE_MAX_VALUE_BYTES:
            raise ValueError(
                f"{subject} value {index} is {len(encoded)} encoded bytes; "
                f"maximum is {_COMBO_CHOICE_MAX_VALUE_BYTES}"
            )
        if raw_value in seen:
            raise ValueError(f"{subject} has duplicate value {raw_value!r}")
        seen.add(raw_value)
        checked.append(raw_value)
    body = json.dumps(checked, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(body) > _COMBO_CHOICE_MAX_RESPONSE_BYTES:
        raise ValueError(
            f"{subject} compact JSON response exceeds {_COMBO_CHOICE_MAX_RESPONSE_BYTES} bytes"
        )
    return body


@dataclass(frozen=True)
class TypeExpr:
    kind: TypeExprKind
    types: tuple[str, ...] = ()
    template_id: str = ""
    element: TypeExpr | None = None
    elements: tuple[TypeExpr, ...] = ()
    input_id: str = ""
    index: int | str | None = None

    def __post_init__(self) -> None:
        if self.kind not in (
            "concrete",
            "union",
            "wildcard",
            "variable",
            "list",
            "asset",
            "stream",
            "tuple",
            "tuple_element",
        ):
            raise ValueError(f"unknown TypeExpr kind: {self.kind!r}")
        if self.kind == "concrete":
            if len(self.types) != 1:
                raise ValueError("concrete TypeExpr requires exactly one type id")
            if parse_tuple_type_id(self.types[0]) is not None:
                raise ValueError("tuple types require TypeExpr.tuple_of(...)")
            if parse_stream_type_id(self.types[0]) is not None:
                raise ValueError("stream types require TypeExpr.stream_of(...)")
            if parse_list_type_id(self.types[0]) is not None:
                raise ValueError(
                    f"list types are spelled TypeExpr.list_of(...), never "
                    f"concrete({self.types[0]!r}) - one schema representation"
                )
            if parse_asset_type_id(self.types[0]) is not None:
                raise ValueError(
                    f"asset types are spelled TypeExpr.asset_of(...), never "
                    f"concrete({self.types[0]!r}) - one schema representation"
                )
        if self.kind == "union" and len(self.types) < 2:
            raise ValueError("union TypeExpr requires at least two type ids")
        if self.kind in ("union", "variable"):
            for entry in self.types:
                if parse_tuple_type_id(entry) is not None:
                    raise ValueError(f"{self.kind} entries are atom type ids, not tuple types")
                if parse_stream_type_id(entry) is not None:
                    raise ValueError(
                        f"{self.kind} TypeExpr entries are atom type ids; stream "
                        f"types are spelled TypeExpr.stream_of(...), got {entry!r}"
                    )
                if parse_list_type_id(entry) is not None:
                    raise ValueError(
                        f"{self.kind} TypeExpr entries are atom type ids; list "
                        f"types are spelled TypeExpr.list_of(...), got {entry!r}"
                    )
                if parse_asset_type_id(entry) is not None:
                    raise ValueError(
                        f"{self.kind} TypeExpr entries are atom type ids; asset "
                        f"types are spelled TypeExpr.asset_of(...), got {entry!r}"
                    )
        if self.kind == "variable" and not self.template_id:
            raise ValueError("variable TypeExpr requires a template_id")
        if self.kind == "wildcard" and self.types:
            raise ValueError("wildcard TypeExpr carries no type ids")
        if self.kind in ("list", "asset", "stream"):
            if self.element is None:
                raise ValueError(f"{self.kind} TypeExpr requires an element expression")
            if self.types or self.template_id:
                raise ValueError(f"{self.kind} TypeExpr carries only an element expression")
        elif self.element is not None and self.kind != "tuple_element":
            raise ValueError(f"{self.kind} TypeExpr carries no element expression")
        if self.kind == "tuple":
            if self.types or self.template_id or not isinstance(cast(object, self.elements), tuple):
                raise ValueError("tuple TypeExpr carries only an immutable elements tuple")
            if any(
                not isinstance(item, TypeExpr) for item in cast(tuple[object, ...], self.elements)
            ):
                raise ValueError("tuple positions must be TypeExprs")
        elif self.elements:
            raise ValueError(f"{self.kind} TypeExpr carries no positional elements")
        if self.kind == "tuple_element":
            if bool(self.input_id) == (self.element is not None):
                raise ValueError("tuple element requires one input reference or element expression")
            if self.types or self.template_id or type(self.index) not in (int, str):
                raise ValueError("tuple element requires an integer index or named literal input")
            if self.index == "":
                raise ValueError("tuple element index input must be nonempty")
        elif self.input_id or self.index is not None:
            raise ValueError(f"{self.kind} TypeExpr carries no input or index reference")

    @staticmethod
    def concrete(type_id: str) -> TypeExpr:
        return TypeExpr(kind="concrete", types=(type_id,))

    @staticmethod
    def union(*type_ids: str) -> TypeExpr:
        return TypeExpr(kind="union", types=tuple(type_ids))

    @staticmethod
    def wildcard() -> TypeExpr:
        return TypeExpr(kind="wildcard")

    @staticmethod
    def variable(template_id: str, allowed: tuple[str, ...] = ()) -> TypeExpr:
        return TypeExpr(kind="variable", types=allowed, template_id=template_id)

    @staticmethod
    def list_of(element: TypeExpr) -> TypeExpr:
        return TypeExpr(kind="list", element=element)

    @staticmethod
    def tuple_of(*elements: TypeExpr) -> TypeExpr:
        return TypeExpr(kind="tuple", elements=elements)

    @staticmethod
    def tuple_element(tuple_input: str, index_input: str) -> TypeExpr:
        """An output position selected by a document-stored integer input."""
        return TypeExpr(kind="tuple_element", input_id=tuple_input, index=index_input)

    @property
    def requires_elaboration(self) -> bool:
        return bool(
            self.input_id
            or isinstance(self.index, str)
            or (self.element is not None and self.element.requires_elaboration)
            or any(item.requires_elaboration for item in self.elements)
        )

    @staticmethod
    def stream_of(element: TypeExpr) -> TypeExpr:
        return TypeExpr(kind="stream", element=element)

    @staticmethod
    def asset_of(element: TypeExpr) -> TypeExpr:
        """An asset that decodes to ``element``.

        The VALUE is one AssetRef - what the decode yields is
        the element expression, so ``asset_of(list_of(...))`` is one asset
        decoding to a list, distinct from ``list_of(asset_of(...))``."""
        return TypeExpr(kind="asset", element=element)

    def accepts_concrete(self, type_id: str) -> bool:
        """Advisory compatibility check (type compat is advisory, structure is
        authoritative - shared principle with the frontend)."""
        if self.kind == "wildcard":
            return True
        if self.kind == "variable":
            return not self.types or type_id in self.types
        if self.kind == "tuple":
            positions = parse_tuple_type_id(type_id)
            return (
                positions is not None
                and len(positions) == len(self.elements)
                and all(
                    expr.accepts_concrete(position)
                    for expr, position in zip(self.elements, positions, strict=True)
                )
            )
        if self.kind == "stream":
            inner = parse_stream_type_id(type_id)
            assert self.element is not None
            return inner is not None and self.element.accepts_concrete(inner)
        if self.kind == "list":
            inner = parse_list_type_id(type_id)
            assert self.element is not None  # post_init invariant
            return inner is not None and self.element.accepts_concrete(inner)
        if self.kind == "asset":
            inner = parse_asset_type_id(type_id)
            assert self.element is not None  # post_init invariant
            return inner is not None and self.element.accepts_concrete(inner)
        return type_id in self.types

    def runtime_type_id(self) -> str | None:
        """The runtime type id this expression denotes when it is recursively
        fully concrete, else None. This is what workers wrap outputs with and
        what the engine wraps literals with - the single bridge between the
        expression level and runtime value type ids."""
        if self.kind == "concrete":
            return self.types[0]
        if self.kind == "tuple":
            positions = tuple(item.runtime_type_id() for item in self.elements)
            if any(item is None for item in positions):
                return None
            return tuple_type_id(*cast("tuple[str, ...]", positions))
        if self.kind == "stream":
            assert self.element is not None
            inner = self.element.runtime_type_id()
            return None if inner is None else stream_type_id(inner)
        if self.kind == "list":
            assert self.element is not None
            if self.element.kind == "wildcard":
                return list_type_id("*")
            inner = self.element.runtime_type_id()
            return None if inner is None else list_type_id(inner)
        if self.kind == "asset":
            assert self.element is not None
            inner = self.element.runtime_type_id()
            return None if inner is None else asset_type_id(inner)
        return None

    @staticmethod
    def runtime_type_atom(type_id: str) -> str | None:
        """The concrete atom inside a runtime type id: peels ``list<...>``
        ``asset<...>``, and ``stream<...>`` layers and returns the innermost
        atom when the id is well-formed, else None. The inverse direction of
        :meth:`runtime_type_id` - the same closed grammar. Angle brackets are
        constructor syntax, never part of an atom. Delegates to
        dinkster_values, which owns the grammar. Consumers that know the
        registered atoms (the engine's registry) can then check the atom is
        real; this method alone only checks shape."""
        return _runtime_type_atom(type_id)

    @staticmethod
    def runtime_cardinality(type_id: str) -> Cardinality:
        """Structural shape of a runtime type id: an outer ``list<...>`` is
        a list; everything else - atoms and ``asset<...>``, which is ONE
        AssetRef regardless of its decode target - is one value. The
        runtime-id mirror of :meth:`cardinality`, for consumers holding a
        stamp instead of an expression (typed literals)."""
        return "list" if parse_list_type_id(type_id) is not None else "scalar"

    def cardinality(self) -> Cardinality:
        """Structural shape of this expression: cardinality is authoritative
        (DESIGN 3.13), so a definitely-list edge into a definitely-scalar
        socket is a document-time error, while wildcard/variable stay
        unknown and never trigger structural rejection. An asset is one
        value (scalar) regardless of what it decodes to."""
        if self.kind == "list":
            return "list"
        if self.kind in ("concrete", "union", "asset", "stream", "tuple"):
            return "scalar"
        return "unknown"


AbsentPolicy = Literal["skip", "accept", "fail", "omit"]

ABSENT_POLICIES: frozenset[str] = frozenset({"skip", "accept", "fail", "omit"})


ASSET_KIND_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*(/[a-z0-9][a-z0-9-]*)+$")
"""The open namespaced asset-kind grammar (mirrors dinkster-assets, which this
package cannot import): '/'-separated lowercase [a-z0-9-] segments, at
least two ('media/image', 'model/lora'). The vocabulary is open - any
grammatical kind is valid - only the grammar is closed."""


SOURCE_FILENAME_KINDS = frozenset(
    {"media/image", "media/audio", "media/video", "data/latent", "media/model3d"}
)
SOURCE_FILENAME_CATEGORIES = frozenset({"input", "output", "temp"})


@dataclass(frozen=True)
class SourceFilenameSpec:
    """Execution binding for a staged Comfy source filename."""

    kind: Literal["media/image", "media/audio", "media/video", "data/latent", "media/model3d"]
    category: Literal["input", "output", "temp"]

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in SOURCE_FILENAME_KINDS:
            raise ValueError(
                "source filename kind must be one of "
                f"{sorted(SOURCE_FILENAME_KINDS)}, got {self.kind!r}"
            )
        if type(self.category) is not str or self.category not in SOURCE_FILENAME_CATEGORIES:
            raise ValueError(
                "source filename category must be one of "
                f"{sorted(SOURCE_FILENAME_CATEGORIES)}, got {self.category!r}"
            )


@dataclass(frozen=True)
class AssetWidget:
    """Presentation request for an asset picker/upload control.

    This is closed data rather than a generic metadata dictionary because
    widget kinds are a frontend/backend contract. New kinds can join the
    Widget union without making arbitrary presentation data executable or
    silently accepted by older code.

    ``kind`` optionally names the namespaced asset kind the picker should
    filter to ('model/lora', 'media/image') - a coarser, purpose-level
    filter than ``accept``'s media types. Presentation only: like the whole
    widget, it never joins schema signatures or execution identity."""

    accept: tuple[str, ...] = ()
    kind: str = ""
    allow_upload: bool = False
    selection: bool = False

    def __post_init__(self) -> None:
        if any(not media_type for media_type in self.accept):
            raise ValueError("asset widget accept entries must be non-empty")
        if self.kind and not ASSET_KIND_PATTERN.fullmatch(self.kind):
            raise ValueError(
                "asset widget kind must be a namespaced asset kind like "
                f"'model/lora', got: {self.kind!r}"
            )
        if type(self.allow_upload) is not bool:
            raise ValueError("asset widget allow_upload must be a bool")
        if type(self.selection) is not bool:
            raise ValueError("asset widget selection must be a bool")
        if self.selection and (
            self.kind
            not in {"model/vae", "model/diffusers", "directory/input", "directory/dataset"}
            or self.allow_upload
        ):
            raise ValueError("asset selection requires a supported exact kind and forbids upload")


@dataclass(frozen=True)
class MountWidget:
    """Select one writable mount id for MOUNT, STRING or their union."""


@dataclass(frozen=True)
class SaveTargetWidget:
    """Presentation request for a save-destination picker: choose a
    readwrite mount and a relative prefix (subfolders + filename stem).
    On an input it produces a ``dinkster.save_target`` wire mapping
    {"mount": ..., "prefix": ...} - structured, never a raw path string.
    On an Editor its parts write the existing rows named by ``writes``.

    ``suffix`` is advisory display data (what extension the node will
    append, e.g. ".png"); the node owns the actual format.
    ``writes`` and ``reads`` map an editor's parts to existing row ids."""

    suffix: str = ""
    writes: Mapping[str, str] = field(default_factory=lambda: cast("Mapping[str, str]", {}))
    reads: Mapping[str, str] = field(default_factory=lambda: cast("Mapping[str, str]", {}))

    def __post_init__(self) -> None:
        writes, reads = editor_row_names(self.writes, self.reads, "save target widget")
        object.__setattr__(self, "writes", writes)
        object.__setattr__(self, "reads", reads)
        if not self.suffix:
            return
        if not self.suffix.startswith(".") or "/" in self.suffix or "\\" in self.suffix:
            raise ValueError("save target widget suffix must be a bare extension like '.png'")


@dataclass(frozen=True)
class ComboOption:
    """One combo value plus optional dropdown presentation metadata."""

    value: str | bool | int | float
    label: str | None = None
    info: str | None = None
    folder: str | None = None

    def __post_init__(self) -> None:
        if not (
            (type(self.value) is str and bool(self.value))
            or type(self.value) is bool
            or (type(self.value) is int and abs(self.value) <= _JSON_SAFE_INT)
            or (type(self.value) is float and math.isfinite(self.value))
        ):
            raise ValueError(
                "combo option value must be a non-empty string, boolean, JSON-safe integer "
                "or finite float"
            )
        for field_name in ("label", "info"):
            field_value = getattr(self, field_name)
            if field_value is not None and (type(field_value) is not str or not field_value):
                raise ValueError(f"combo option {field_name} must be a non-empty string")
        if self.folder is None:
            return
        if type(self.folder) is not str or not self.folder or "\\" in self.folder:
            raise ValueError("combo option folder must be a relative /-separated path")
        if any(segment in {"", ".", ".."} for segment in self.folder.split("/")):
            raise ValueError(
                "combo option folder must contain non-empty segments other than '.' or '..'"
            )


def _validate_static_combo_options(
    options: tuple[str | bool | int | float | ComboOption, ...],
    *,
    subject: str,
    strings_only: bool = False,
) -> None:
    if not isinstance(cast("object", options), tuple) or any(
        not (
            (type(option) is ComboOption and (not strings_only or type(option.value) is str))
            or (type(option) is str and bool(option))
            or (
                not strings_only
                and (
                    type(option) is bool
                    or (type(option) is int and abs(option) <= _JSON_SAFE_INT)
                    or (type(option) is float and math.isfinite(option))
                )
            )
        )
        for option in cast("tuple[object, ...]", options)
    ):
        choices = (
            "non-empty strings"
            if strings_only
            else "non-empty strings, booleans, JSON-safe integers or finite floats"
        )
        raise ValueError(f"{subject} options must be {choices} or ComboOption values")


@dataclass(frozen=True)
class NamedList:
    """Options from a list several packs add to, such as ``comfy.samplers``."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.name), str) or not self.name:
            raise ValueError("NamedList needs a list name")


@dataclass(frozen=True)
class ComboWidget:
    """Presentation request for a combo-choice dropdown.

    The socket has concrete ``COMBO`` identity while its runtime value,
    literal, cache payload, and node-facing object stay a string, boolean or
    finite number without type conversion.
    Choice vocabulary is never identity within combos: different option
    lists remain compatible, and a stored value outside the current options
    is the frontend's diagnostic to surface, not a schema error. This
    descriptor never joins schema signatures or execution identity. Input
    enforces the concrete COMBO binding.

    ``options`` are static choices, rendered immediately, or a
    ``NamedList``: the choices of a list packs register, which the widget
    reads from that list's route, so ``remote_route`` is set from it.
    ``remote_route``
    optionally names a server route (GET, JSON array of strings) the
    frontend queries at editor-open - the fit for lists that vary by
    installation, like registered samplers: a pack registers one, the
    route enumerates it, the dropdown updates, no schema change.
    ``refresh_button`` asks the frontend for a user-triggered re-fetch
    control on remote combos. The optional remote policy fields declare
    initial client IO behavior: selection after a manual refresh, request
    timeout, retries after the initial attempt, and automatic refresh
    interval. Their absence means the client defaults (4096 ms, two
    retries, and no automatic expiry); like the route itself, these facts
    remain presentation-only and never join execution identity.

    Contract edges agreed with the frontend (wire v9): when both are
    present, a fetched remote result REPLACES the static options (remote
    is authoritative; static is the immediate render set, typically a
    snapshot of the route's core entries). A remote-only combo whose
    route has not answered yet is a legal transient state - the frontend
    renders a loading/empty dropdown, and a stored value outside the
    current option set is a frontend diagnostic, never a reset.

    ``options`` may also be a ``Formula`` the editor works out or a
    ``Lookup`` the host calls; such a list is the only source of choices
    and, like the others, never decides the stored value.
    """

    options: tuple[str | bool | int | float | ComboOption, ...] | NamedList | Formula | Lookup = ()
    remote_route: str = ""
    refresh_button: bool = False
    control_after_refresh: Literal["first", "last"] | None = None
    remote_timeout_ms: int | None = None
    remote_max_retries: int | None = None
    remote_refresh_ms: int | None = None
    editable_options: bool = False
    options_input: str = ""
    """With ``editable_options``, names a top-level hidden ``list<STRING>``
    input that the editor fills with the authored options, in order, on every
    submission, so the node can read them. Empty means the options stay in the
    editor."""

    @property
    def static_options(self) -> tuple[str | bool | int | float | ComboOption, ...]:
        """The static choices; a ``NamedList`` given as ``options`` has become the route."""
        return self.options if isinstance(self.options, tuple) else ()

    def __post_init__(self) -> None:
        from .body import Formula, Lookup

        if isinstance(self.options, (Formula, Lookup)):
            remote_policy = (
                self.control_after_refresh,
                self.remote_timeout_ms,
                self.remote_max_retries,
                self.remote_refresh_ms,
            )
            if (
                self.remote_route
                or self.editable_options
                or self.refresh_button
                or any(value is not None for value in remote_policy)
            ):
                raise ValueError(
                    "combo widget options from a formula or a lookup take no other option "
                    "source and no refresh button"
                )
            return
        if isinstance(self.options, NamedList):
            if self.remote_route:
                raise ValueError("combo widget takes a NamedList or a remote route, not both")
            route = _REMOTE_CHOICE_ROUTE_PREFIX + self.options.name
            _remote_choice_id(route)
            object.__setattr__(self, "remote_route", route)
            object.__setattr__(self, "options", ())
        _validate_static_combo_options(self.static_options, subject="combo widget")
        if type(self.editable_options) is not bool:
            raise ValueError("combo widget editable_options must be a bool")
        if self.editable_options and self.remote_route:
            raise ValueError("editable combo options cannot have another option source")
        if type(self.options_input) is not str:
            raise ValueError("combo widget options_input must be a string")
        if self.options_input and not self.editable_options:
            raise ValueError("combo widget options_input requires editable options")
        if self.remote_route:
            _remote_choice_id(self.remote_route)
        if not self.options and not self.remote_route and not self.editable_options:
            raise ValueError(
                "combo widget needs static options, a remote route, or editable options"
            )
        if self.refresh_button and not self.remote_route:
            raise ValueError(
                "combo widget refresh button requires a remote route "
                "(static options have nothing to re-fetch)"
            )
        remote_policy = (
            self.control_after_refresh,
            self.remote_timeout_ms,
            self.remote_max_retries,
            self.remote_refresh_ms,
        )
        if any(value is not None for value in remote_policy) and not self.remote_route:
            raise ValueError("combo widget remote policy requires a remote route")
        if self.control_after_refresh is not None and (
            type(self.control_after_refresh) is not str
            or self.control_after_refresh not in {"first", "last"}
        ):
            raise ValueError(
                "combo widget control_after_refresh must be 'first' or 'last', "
                f"got {self.control_after_refresh!r}"
            )
        if self.control_after_refresh is not None and not self.refresh_button:
            raise ValueError("combo widget control_after_refresh requires a refresh button")
        for field_name, value, lower, upper in (
            ("remote_timeout_ms", self.remote_timeout_ms, 1, 60_000),
            ("remote_max_retries", self.remote_max_retries, 0, 5),
            ("remote_refresh_ms", self.remote_refresh_ms, 0, 86_400_000),
        ):
            if value is not None and (type(value) is not int or not lower <= value <= upper):
                raise ValueError(
                    f"combo widget {field_name} must be an integer in {lower}..{upper}, "
                    f"got {value!r}"
                )


@dataclass(frozen=True)
class MultiComboWidget:
    """Presentation request for a list-valued combo selector.

    Unlike ``ComboWidget``, this descriptor owns one canonical
    ``list<COMBO>`` value. Options and remote policy are presentation
    vocabulary only; selected order and duplicate multiplicity remain value,
    fingerprint, and cache identity. There is deliberately no
    control-after-generate field because upstream defines no deterministic
    list transformation for it."""

    options: tuple[str | ComboOption, ...] = ()
    remote_route: str = ""
    refresh_button: bool = False
    control_after_refresh: Literal["first", "last"] | None = None
    remote_timeout_ms: int | None = None
    remote_max_retries: int | None = None
    remote_refresh_ms: int | None = None
    placeholder: str | None = None
    chip: bool | None = None

    def __post_init__(self) -> None:
        _validate_static_combo_options(
            self.options, subject="multi combo widget", strings_only=True
        )
        if self.remote_route:
            _remote_choice_id(self.remote_route)
        if not self.options and not self.remote_route:
            raise ValueError("multi combo widget needs static options, a remote route, or both")
        if self.refresh_button and not self.remote_route:
            raise ValueError("multi combo widget refresh button requires a remote route")
        remote_policy = (
            self.control_after_refresh,
            self.remote_timeout_ms,
            self.remote_max_retries,
            self.remote_refresh_ms,
        )
        if any(value is not None for value in remote_policy) and not self.remote_route:
            raise ValueError("multi combo widget remote policy requires a remote route")
        if self.control_after_refresh is not None and (
            type(self.control_after_refresh) is not str
            or self.control_after_refresh not in {"first", "last"}
        ):
            raise ValueError(
                "multi combo widget control_after_refresh must be 'first' or 'last', "
                f"got {self.control_after_refresh!r}"
            )
        if self.control_after_refresh is not None and not self.refresh_button:
            raise ValueError("multi combo widget control_after_refresh requires a refresh button")
        for field_name, value, lower, upper in (
            ("remote_timeout_ms", self.remote_timeout_ms, 1, 60_000),
            ("remote_max_retries", self.remote_max_retries, 0, 5),
            ("remote_refresh_ms", self.remote_refresh_ms, 0, 86_400_000),
        ):
            if value is not None and (type(value) is not int or not lower <= value <= upper):
                raise ValueError(
                    f"multi combo widget {field_name} must be an integer in {lower}..{upper}, "
                    f"got {value!r}"
                )
        if self.placeholder is not None and type(self.placeholder) is not str:
            raise ValueError("multi combo widget placeholder must be a string")
        if self.chip is not None and type(self.chip) is not bool:
            raise ValueError("multi combo widget chip must be a bool")


@dataclass(frozen=True)
class BooleanWidget:
    """Presentation request for a boolean toggle with custom state labels.

    Only meaningful on ``BOOLEAN`` inputs. A label-less boolean
    carries no widget at all - the type alone means "render a toggle" -
    so this widget exists purely to carry ComfyUI-style label_on/
    label_off prose ("enable"/"disable" on a mute toggle). Presentation
    only: never in signatures, never execution identity."""

    label_on: str = ""
    label_off: str = ""

    def __post_init__(self) -> None:
        if not self.label_on and not self.label_off:
            raise ValueError(
                "boolean widget needs at least one custom label "
                "(a label-less boolean input carries no widget)"
            )


AfterRunMode = Literal["fixed", "increment", "decrement", "randomize"]
"""The INITIAL mode of an input's after-run controller: what the editor does
to the field's value after each submitted run. The mode seeds the control and
never overwrites it - the live mode is client-owned mutable state, so later
schema reads must not reset a user's chosen mode.

The vocabulary is closed: any addition is a wire bump, not an in-place grow.
The backend treats an unknown mode as a decode error."""

AFTER_RUN_MODES: Final = ("fixed", "increment", "decrement", "randomize")

# The largest integer a JSON double represents exactly. Larger integer
# constraints use canonical decimal strings.
_JSON_SAFE_INT = 2**53 - 1
_DECIMAL_WIRE_INT_MIN = -(2**63)
_DECIMAL_WIRE_INT_MAX = 2**64 - 1


@dataclass(frozen=True)
class NumberWidget:
    """Presentation request for a numeric input's editor.

    Numeric sockets and their exact union use this presentation without
    changing their runtime value domains.
    Out-of-range stored values are the frontend's diagnostic to surface, never a schema error;
    a collaborative session write that changes a number to below ``min``, above ``max`` or,
    for a whole-number input, off its step is refused by the server unless the number is
    one of ``special``. Absent min/max means unbounded. Integer
    constraints through signed 64-bit minimum and unsigned 64-bit maximum are
    lossless as canonical decimal strings outside the JSON-double-safe range.

    ``display`` is an explicit editor presentation when present. Bounds and
    step are independent constraints: they never imply a slider or any other
    presentation. ``round`` is an independent finite positive rendering
    precision, or literal False to disable rounding, using FLOAT semantics;
    it never implies display or changes bounds/step. A whole number is on a
    step when it is ``offset`` plus a multiple of ``step``; ``offset`` is
    ``min`` when not given, and 0 when there is no ``min``. ``special`` maps
    values allowed outside the limits and off the step to their labels.
    ``reason`` is the short text that explains a limit. Absence of the descriptor means "no
    presentation metadata", and the frontend still renders a number editor
    by primitive type inference; it never means the input is widgetless
    (pinned with the frontend).

    Socket binding: this descriptor is only
    legal on concrete ``INT``/``FLOAT`` inputs, their one-level lists,
    and the scalar ``INT``/``FLOAT`` union.
    Constraints use the socket's domain (FLOAT for the union) - an int socket with a
    fractional min/max/step is malformed. Enforced by Input, which is
    the first place the widget meets its socket type."""

    min: int | float | Formula | Lookup | None = None
    max: int | float | Formula | Lookup | None = None
    step: int | float | Formula | Lookup | None = None
    display: Literal["number", "slider", "knob", "gradientslider", "color"] | None = None
    round: int | float | Literal[False] | None = None
    offset: int | float | Formula | Lookup | None = None
    special: Mapping[int | float, str] = field(
        default_factory=lambda: cast("Mapping[int | float, str]", {})
    )
    reason: str | None = None
    """``min``, ``max``, ``step`` and ``offset`` may also be a ``Formula`` the
    editor works out or a ``Lookup`` the host calls; ``None`` from either means
    no bound and no offset. While an offset is not known the editor draws no
    step grid. The number checks below apply to literal numbers only."""

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.special), Mapping):
            raise ValueError("number widget special must map each value to its label")
        special = dict(cast("Mapping[object, object]", self.special))
        for value, label in special.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"number widget special value must be a number, got {value!r}")
            if not isinstance(label, str) or not label:
                raise ValueError(
                    f"number widget special label for {value!r} must be non-empty text"
                )
        object.__setattr__(
            self,
            "special",
            dict(sorted(cast("dict[int | float, str]", special).items())),
        )
        if self.reason is not None and (
            not isinstance(cast("object", self.reason), str) or not self.reason
        ):
            raise ValueError("number widget reason must be non-empty text")
        if (
            self.min is None
            and self.max is None
            and self.step is None
            and self.round is None
            and self.display is None
            and self.offset is None
            and not self.special
            and self.reason is None
        ):
            raise ValueError(
                "number widget needs at least one constraint or a display "
                "(a metadata-less numeric input carries no widget)"
            )
        from .body import Formula, Lookup

        for name in ("min", "max", "step", "offset"):
            value = getattr(self, name)
            if value is not None and type(value) not in (int, float):
                if not isinstance(value, (Formula, Lookup)):
                    raise ValueError(
                        f"number widget {name} must be a number, a Formula or a Lookup, "
                        f"got {value!r}"
                    )
        for name, bound in self.numbers():
            if bound is None:
                continue
            if isinstance(bound, bool):
                # bool is an int subclass; True/False here is always a
                # call-site mistake, never a numeric constraint.
                raise ValueError(f"number widget {name} must be a number, got {bound}")
            if not math.isfinite(bound):
                # NaN would also slip past the ordering/positivity checks
                # below, and none of these survive canonical JSON anyway.
                raise ValueError(f"number widget {name} must be finite, got {bound}")
            if (
                isinstance(bound, int)
                and not _DECIMAL_WIRE_INT_MIN <= bound <= _DECIMAL_WIRE_INT_MAX
            ):
                raise ValueError(
                    f"number widget {name} must fit the supported integer range "
                    f"({_DECIMAL_WIRE_INT_MIN}..{_DECIMAL_WIRE_INT_MAX}), got {bound}"
                )
        if (
            self.round is not None
            and self.round is not False
            and (
                type(self.round) not in (int, float)
                or not math.isfinite(self.round)
                or self.round <= 0
            )
        ):
            raise ValueError(
                f"number widget round must be False or a finite positive number, got {self.round!r}"
            )
        literal = dict(self.numbers())
        low, high, step = literal["min"], literal["max"], literal["step"]
        if low is not None and high is not None and low > high:
            raise ValueError(f"number widget min ({low}) must not exceed max ({high})")
        if step is not None and step <= 0:
            raise ValueError(f"number widget step must be positive, got {step}")
        if self.offset is not None and self.step is None:
            raise ValueError("number widget offset needs a step to count from")
        if self.reason is not None and self.min is None and self.max is None and self.step is None:
            raise ValueError("number widget reason needs a min, a max or a step to explain")
        if self.display is not None and (
            type(self.display) is not str
            or self.display not in {"number", "slider", "knob", "gradientslider", "color"}
        ):
            raise ValueError(
                "number widget display must be one of "
                "['color', 'gradientslider', 'knob', 'number', 'slider'], "
                f"got {self.display!r}"
            )

    def numbers(self) -> tuple[tuple[str, int | float | None], ...]:
        """Each literal number the widget declares, named for messages: limits,
        offset, special values. A limit or offset given by a formula or a
        lookup is None here."""

        def literal(value: object) -> int | float | None:
            return cast("int | float", value) if type(value) in (int, float) else None

        return (
            ("min", literal(self.min)),
            ("max", literal(self.max)),
            ("step", literal(self.step)),
            ("offset", literal(self.offset)),
            *((f"special value {value!r}", value) for value in self.special),
        )


TextCompletionKind = Literal["identifier", "operator"]


@dataclass(frozen=True)
class TextCompletionItem:
    """One schema-declared text completion candidate.

    ``value`` is the stable candidate and filter text. ``insert_text`` and
    ``label`` default to that value. ``kind`` selects the token class whose
    exact range the frontend replaces. Completion metadata is presentation
    only and never joins schema signatures or execution identity."""

    value: str
    label: str = ""
    insert_text: str = ""
    detail: str = ""
    kind: TextCompletionKind = "identifier"

    def __post_init__(self) -> None:
        if type(self.value) is not str or not self.value:
            raise ValueError("text completion value must be a non-empty string")
        for field_name in ("label", "insert_text", "detail"):
            field_value = getattr(self, field_name)
            if type(field_value) is not str:
                raise ValueError(f"text completion {field_name} must be a string")
        if self.kind not in ("identifier", "operator"):
            raise ValueError("text completion kind must be 'identifier' or 'operator'")


@dataclass(frozen=True)
class TextCompletions:
    """Static completion candidates."""

    items: tuple[TextCompletionItem, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.items), tuple) or any(
            type(item) is not TextCompletionItem for item in cast("tuple[object, ...]", self.items)
        ):
            raise ValueError("text completion items must be an immutable tuple")
        if not self.items:
            raise ValueError("text completions need at least one item")


@dataclass(frozen=True)
class StringWidget:
    """Presentation request for a string editor.

    Only legal on concrete ``STRING`` inputs (enforced by Input).
    Explicit ``multiline=False`` lets a named representation set compose both
    one-line and multiline editors from the same closed
    descriptor. Absence still means the original inferred one-line editor,
    and the default stays ``True`` so every existing declaration is unchanged.
    ``placeholder`` and ``completions`` are presentation-only.
    ``dynamic_prompts`` is tri-state
    and joins schema identity because it controls client prompt serialization;
    the backend receives and executes only the resulting literal string."""

    multiline: bool | None = True
    placeholder: str | None = None
    dynamic_prompts: bool | None = None
    completions: TextCompletions | Formula | Lookup | None = None
    """Literal ``TextCompletions``, or a ``Formula`` or ``Lookup`` whose
    list the editor filters while the user types."""

    def __post_init__(self) -> None:
        if self.multiline is not None and type(self.multiline) is not bool:
            raise ValueError("string widget multiline must be a bool")
        if self.placeholder is not None and type(self.placeholder) is not str:
            raise ValueError("string widget placeholder must be a string")
        if self.dynamic_prompts is not None and type(self.dynamic_prompts) is not bool:
            raise ValueError("string widget dynamic_prompts must be a bool")
        from .body import Formula, Lookup

        if self.completions is not None and not isinstance(
            cast("object", self.completions), (TextCompletions, Formula, Lookup)
        ):
            raise ValueError(
                "string widget completions must be TextCompletions, a Formula or a Lookup"
            )
        if (
            self.multiline is None
            and self.placeholder is None
            and self.dynamic_prompts is None
            and self.completions is None
        ):
            raise ValueError("string widget needs at least one presentation field")


@dataclass(frozen=True)
class ColorWidget:
    """Color-picker presentation over a canonical STRING value.

    ``alpha`` says the node uses opacity, so the picker offers it."""

    alpha: bool = False


_STRIP_COLOR: Final = re.compile(r"#[0-9a-fA-F]{6}")
"""A color of a curve's strip: "#rrggbb", either case of hex digit."""


def _strip_colors(value: object, where: str) -> tuple[str, str]:
    """A strip's ``(low, high)`` colors in lower case, or ValueError."""

    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{where} must be a pair of #rrggbb colors")
    colors = cast("Sequence[object]", value)
    if len(colors) != 2 or not all(
        isinstance(color, str) and _STRIP_COLOR.fullmatch(color) for color in colors
    ):
        raise ValueError(f"{where} must be a pair of #rrggbb colors")
    return (cast("str", colors[0]).lower(), cast("str", colors[1]).lower())


@dataclass(frozen=True)
class CurveWidget:
    """Curve-editor presentation over a canonical dinkster.curve value.

    ``input_strip`` draws a color strip under the plot, along the curve's
    input, and ``output_strip`` one at its left, along its output. Each is a
    ``(low, high)`` pair of "#rrggbb" colors, stored in lower case, or None
    for no strip. They are layout: neither signature includes them.

    ``histogram`` names the input beside the curve that is its histogram: an
    optional HISTOGRAM input of the curve's own scope, whose pin the editor
    draws inside the curve's editor. Empty means the curve has none. The
    engine sends the value arriving there to this curve's editor. Like the
    strips, it is layout: neither signature includes it. ``histogram_visible``
    sets the line's default visibility; a wire always shows it and a saved
    workflow may override it. It is layout too.

    ``interpolations`` lists the editor's choices in order. A single kind
    fixes the editor's plot and hides the choice; it does not restrict values
    arriving by wire. Like the other presentation settings, it is layout."""

    input_strip: tuple[str, str] | None = None
    output_strip: tuple[str, str] | None = None
    histogram: str = ""
    histogram_visible: bool = True
    interpolations: tuple[Literal["linear", "monotone_cubic", "step"], ...] = (
        "linear",
        "monotone_cubic",
        "step",
    )

    def __post_init__(self) -> None:
        if type(self.histogram) is not str:
            raise ValueError("curve widget histogram must be a string")
        if type(self.histogram_visible) is not bool:
            raise ValueError("curve widget histogram_visible must be a bool")
        kinds = cast("object", self.interpolations)
        if not isinstance(kinds, tuple):
            raise ValueError("curve widget interpolations must be distinct supported kinds")
        choices = cast("tuple[object, ...]", kinds)
        if (
            not choices
            or any(kind not in ("linear", "monotone_cubic", "step") for kind in choices)
            or len(set(choices)) != len(choices)
        ):
            raise ValueError("curve widget interpolations must be distinct supported kinds")
        for name in ("input_strip", "output_strip"):
            value = cast("object", getattr(self, name))
            if value is not None:
                object.__setattr__(self, name, _strip_colors(value, f"curve widget {name}"))


@dataclass(frozen=True)
class CompositorWidget:
    """Compositor presentation over a canonical dinkster.compositor value.

    ``writes`` and ``reads`` map the editor's name for a part to a row id."""

    writes: Mapping[str, str] = field(default_factory=lambda: cast("Mapping[str, str]", {}))
    reads: Mapping[str, str] = field(default_factory=lambda: cast("Mapping[str, str]", {}))

    def __post_init__(self) -> None:
        writes, reads = editor_row_names(self.writes, self.reads, "compositor widget")
        object.__setattr__(self, "writes", writes)
        object.__setattr__(self, "reads", reads)


WidgetDescriptor = (
    CustomWidgetDescriptor
    | AssetWidget
    | MountWidget
    | SaveTargetWidget
    | ComboWidget
    | MultiComboWidget
    | BooleanWidget
    | NumberWidget
    | StringWidget
    | ColorWidget
    | CurveWidget
    | CompositorWidget
)
"""One input presentation descriptor."""


def _widget_value_domain(widget: WidgetDescriptor) -> str:
    """The canonical value/literal domain a descriptor presents.

    Representations may vary rendering inside one domain, but may not switch
    between editors that construct different canonical value shapes.
    """

    if isinstance(widget, CustomWidgetDescriptor):
        return widget.widget_type
    if isinstance(widget, AssetWidget):
        return "asset-selection" if widget.selection else "asset"
    if isinstance(widget, SaveTargetWidget):
        return "save-target"
    if isinstance(widget, ComboWidget):
        return "combo"
    if isinstance(widget, MultiComboWidget):
        return "multi-combo"
    if isinstance(widget, BooleanWidget):
        return "boolean"
    if isinstance(widget, NumberWidget):
        return "number"
    if isinstance(widget, ColorWidget):
        return "string"
    if isinstance(widget, CurveWidget):
        return "curve"
    if isinstance(widget, CompositorWidget):
        return "compositor"
    return "string"


STRUCTURAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
"""Schema-authored identifiers that stay stable and wire-safe."""


def _validate_structural_id(value: str, field_name: str) -> None:
    if not STRUCTURAL_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must match [A-Za-z0-9_-]+: {value!r}")


@dataclass(frozen=True)
class WidgetRepresentation:
    """One stable name for one presentation of an input's canonical value."""

    id: str
    widget: WidgetDescriptor
    display_name: str = ""

    def __post_init__(self) -> None:
        _validate_structural_id(self.id, "widget representation id")
        raw_widget = cast("object", self.widget)
        if not isinstance(
            raw_widget,
            (
                CustomWidgetDescriptor,
                AssetWidget,
                MountWidget,
                SaveTargetWidget,
                ComboWidget,
                MultiComboWidget,
                BooleanWidget,
                NumberWidget,
                StringWidget,
                ColorWidget,
                CurveWidget,
                CompositorWidget,
            ),
        ):
            raise ValueError("widget representation must contain one widget descriptor")
        if not isinstance(cast("object", self.display_name), str):
            raise ValueError("widget representation display_name must be a string")


@dataclass(frozen=True)
class WidgetRepresentations:
    """Finite named presentation choices for one canonical input.

    ``default`` is the initial presentation. ``user_switchable`` explicitly
    grants or forbids a client-side presentation switch. Normally neither
    field, nor a client's selected id, changes input or cache identity. When
    any member declares ``dynamic_prompts``, the declaration map and default
    join schema identity because they determine client serialization. The
    resulting submitted literal carries ordinary value/cache identity;
    ``user_switchable`` and the live selected id remain client-owned state.
    """

    representations: tuple[WidgetRepresentation, ...]
    default: str
    user_switchable: bool

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.representations), tuple):
            raise ValueError("widget representations must be an immutable tuple")
        if not self.representations:
            raise ValueError("widget representations must not be empty")
        if not all(
            isinstance(item, WidgetRepresentation)
            for item in cast("tuple[object, ...]", self.representations)
        ):
            raise ValueError("widget representations must contain WidgetRepresentation values")
        ids = tuple(item.id for item in self.representations)
        if len(set(ids)) != len(ids):
            raise ValueError("widget representation ids must be unique")
        domains = {_widget_value_domain(item.widget) for item in self.representations}
        if len(domains) != 1:
            raise ValueError("widget representations must share one canonical value domain")
        if domains == {"asset"}:
            kinds = {cast("AssetWidget", item.widget).kind for item in self.representations}
            if len(kinds) != 1:
                raise ValueError("asset widget representations must share one asset kind")
        if not isinstance(cast("object", self.default), str):
            raise ValueError("widget representations default must be a string")
        if self.default not in ids:
            raise ValueError("widget representations default must name a declared representation")
        if type(self.user_switchable) is not bool:
            raise ValueError("widget representations user_switchable must be a bool")

    def default_representation(self) -> WidgetRepresentation:
        return next(item for item in self.representations if item.id == self.default)


Widget = WidgetDescriptor | WidgetRepresentations
"""A singular descriptor or a finite named set of descriptors."""


def _validate_widget_binding(input_id: str, input_type: TypeExpr, widget: WidgetDescriptor) -> None:
    """Bind one descriptor to the canonical socket/value domain."""

    primitive_type = (
        cast("TypeExpr", input_type.element) if input_type.kind == "list" else input_type
    )
    if isinstance(widget, NumberWidget):
        socket = primitive_type.types[0] if primitive_type.kind == "concrete" else None
        if input_type.kind == "union" and set(input_type.types) == {CORE_INT, CORE_FLOAT}:
            socket = CORE_FLOAT
        if socket not in (CORE_INT, CORE_FLOAT):
            raise ValueError(
                f"input {input_id}: number widget requires a concrete "
                f"INT or FLOAT input or their scalar union, got {input_type.kind} "
                f"{socket or ''}".rstrip()
            )
        if socket == CORE_INT:
            for name, bound in widget.numbers():
                if bound is not None and bound != int(bound):
                    raise ValueError(
                        f"input {input_id}: number widget {name} must be integral "
                        f"on a INT input, got {bound}"
                    )
                if isinstance(bound, float) and abs(bound) > _JSON_SAFE_INT:
                    raise ValueError(
                        f"input {input_id}: number widget {name} must be exactly representable "
                        f"on a INT input, got {bound}"
                    )
        if socket == CORE_FLOAT:
            for name, bound in widget.numbers():
                if isinstance(bound, int) and abs(bound) > _JSON_SAFE_INT:
                    raise ValueError(
                        f"input {input_id}: number widget {name} must use a float "
                        f"outside the safe integer range on a FLOAT input, got {bound}"
                    )
        if widget.offset is not None and socket == CORE_FLOAT:
            raise ValueError(
                f"input {input_id}: number widget offset applies to whole numbers, "
                "not a FLOAT input"
            )
        if widget.round is not None and socket != CORE_FLOAT:
            raise ValueError(
                f"input {input_id}: number widget round requires a concrete FLOAT input"
            )
    if isinstance(widget, StringWidget) and primitive_type != TypeExpr.concrete(CORE_STRING):
        raise ValueError(f"input {input_id}: string widget requires a concrete STRING input")
    if (
        isinstance(widget, MountWidget)
        and input_type
        not in (
            TypeExpr.concrete(MOUNT_TYPE),
            TypeExpr.concrete(CORE_STRING),
        )
        and not (input_type.kind == "union" and set(input_type.types) == {MOUNT_TYPE, CORE_STRING})
    ):
        raise ValueError(f"input {input_id}: mount widget requires MOUNT, STRING or their union")
    if isinstance(widget, ColorWidget) and input_type not in (
        TypeExpr.concrete(CORE_STRING),
        TypeExpr.concrete(COLOR_TYPE),
    ):
        raise ValueError(
            f"input {input_id}: color widget requires a concrete STRING or COLOR input"
        )
    if isinstance(widget, CurveWidget) and input_type != TypeExpr.concrete(CURVE_TYPE):
        raise ValueError(f"input {input_id}: curve widget requires a concrete {CURVE_TYPE} input")
    if isinstance(widget, CompositorWidget) and input_type != TypeExpr.concrete(COMPOSITOR_TYPE):
        raise ValueError(
            f"input {input_id}: compositor widget requires a concrete {COMPOSITOR_TYPE} input"
        )
    if isinstance(widget, BooleanWidget) and primitive_type != TypeExpr.concrete(CORE_BOOLEAN):
        raise ValueError(f"input {input_id}: boolean widget requires a concrete BOOLEAN input")
    if isinstance(widget, ComboWidget) and primitive_type != TypeExpr.concrete(CORE_COMBO):
        raise ValueError(f"input {input_id}: combo widget requires a concrete COMBO input")
    if isinstance(widget, MultiComboWidget) and input_type != TypeExpr.list_of(
        TypeExpr.concrete(CORE_COMBO)
    ):
        raise ValueError(f"input {input_id}: multi combo widget requires a list<COMBO> input")
    selection_union = (
        isinstance(widget, AssetWidget)
        and widget.selection
        and widget.kind == "model/vae"
        and input_type.kind == "union"
        and set(input_type.types) == {CORE_COMBO, ASSET_TYPE}
    )
    if (
        isinstance(widget, AssetWidget)
        and widget.selection
        and not (input_type == TypeExpr.concrete(CORE_COMBO) or selection_union)
    ):
        raise ValueError(
            f"input {input_id}: asset selection requires a scalar COMBO input "
            "or the model/vae COMBO/ASSET union"
        )
    if isinstance(widget, AssetWidget) and not (
        selection_union
        or input_type == TypeExpr.concrete(ASSET_TYPE)
        or (input_type.kind == "asset" and input_type.runtime_type_id() is not None)
        or input_type.kind == "concrete"
        or (widget.allow_upload and input_type == TypeExpr.list_of(TypeExpr.concrete(ASSET_TYPE)))
    ):
        # Asset destinations receive one ref directly. Concrete scalar
        # destinations receive an asset-stamped literal that the worker
        # decodes (or a stamped list that it terminally merges). The exact
        # upload-enabled list<ASSET> spelling supports ordered source
        # uploads; other lists and unresolved asset variables cannot determine
        # a contract.
        raise ValueError(
            f"input {input_id}: asset widget requires a concrete "
            f"scalar, bare {ASSET_TYPE}, upload-enabled list<{ASSET_TYPE}>, or "
            "fully-concrete asset<...> input; other list inputs and "
            "asset<variable> are not supported"
        )
    if isinstance(widget, SaveTargetWidget) and input_type != TypeExpr.concrete(SAVE_TARGET_TYPE):
        raise ValueError(
            f"input {input_id}: save target widget requires a concrete {SAVE_TARGET_TYPE} input"
        )


@dataclass(frozen=True)
class Input:
    id: str
    type: TypeExpr
    required: bool = True
    default: object = None
    tooltip: str = ""
    on_absent: AbsentPolicy | None = None
    widget: Widget | None = None
    label: TextOrFormula = ""
    pin_first: bool = False
    hidden: bool = False
    lazy: bool = False
    source_filename: SourceFilenameSpec | None = None
    accepts_storage: bool = False
    accepts_stream: bool = False
    accepts_list: bool = False
    alpha_policy: Literal["preserve", "require", "create_if_missing", "drop"] = "preserve"
    mask_polarity: Literal["coverage", "transparency"] | None = None
    mask_semantic: Literal["alpha", "selection", "other"] | None = None
    pin: bool = True
    on_edit: tuple[Step, ...] = ()
    after_run: AfterRunMode | None = None
    visible: FlagOrFormula = True
    read_only: FlagOrFormula = False
    disabled: FlagOrFormula = False
    disabled_reason: TextOrFormula = ""
    automatic: Formula | Lookup | None = None
    sent: bool = True
    list_count_of: str | None = None
    """Store a known list input's count with the causing editor command.
    Both fields belong to the node's top-level scope.
    Unknown counts leave the stored value alone; runs never update it."""
    """``label`` is presentation only, text or a formula; empty means "derive
    from the id". Like
    ``tooltip`` and ``widget`` it never
    joins schema signatures or execution identity (hazard H15: presentation
    is pixels, never identity).

    ``pin=False`` omits this field's incoming connection affordance without
    changing its value domain, declaration order, or execution identity.
    ``pin_first`` shows the field as a pin until the user converts it.

    ``visible`` shows or hides the field, by a flag or a formula; a hidden
    field is still stored and still sent to ``execute``.

    ``read_only`` shows the field at full contrast but lets nobody type into
    it; ``disabled`` dims it and lets nobody type into it, and
    ``disabled_reason`` says why. Each is a literal or a formula the editor
    works out; while a formula is unknown or fails the field is neither. They
    limit the user's own edits only: the value is still sent, a wire still
    works, and a step or an after-run write may still change it. Like
    ``visible`` they are layout and never join signatures or execution
    identity.

    ``sent=False`` keeps the field with the workflow only, like a ``State``
    with ``part_of_run=False``: it is stored and edited, but a run neither
    receives it nor keys its cache on it, and no formula the engine works out
    may read it. It has no pin.

    ``after_run`` shows an after-run controller on a number or combo field,
    seeded with this mode: what the editor does to the value after each run.

    ``automatic`` is the value used while nothing is stored for the field: a
    formula or a lookup the editor works out to show and the host works out
    again when a run is queued. When it cannot be worked out the field's
    ``default`` is used, so a field that declares it must declare a default.
    A formula form joins both signatures; a lookup form joins neither. The
    value worked out is the field's value for the run.

    ``on_edit`` holds the steps the host applies with the user's own change of
    this field, in the same undo step. The engine never runs them.

    ``lazy`` marks an input whose linked producer is deferred until the node's
    worker-bound hook requests it. Unlike the nearby presentation fields, it
    changes scheduling and therefore joins schema signatures.

    ``accepts_storage`` declares that the consumer accepts compact storage
    dtypes without input conversion. It joins execution identity only when true.

    ``accepts_list`` admits a whole list of the declared input type as well as
    one value. The node consumes the unchanged envelope; this never requests
    repeated execution or scalar wrapping.

    ``on_absent`` is what happens when this input's *linked* value arrives absent
    (DESIGN 3.15). Applied by the engine before invocation - node code never
    sees an absent envelope unless it opted in:

    - ``"skip"``: the node does not execute; its outputs become absent with
      the origin's provenance. Default for required inputs.
    - ``"omit"``: the input is treated as unconnected (dropped before the
      cache key, so it hits the same entries). Default for optional inputs;
      invalid on required ones (execute() has no default for the parameter).
    - ``"accept"``: the node executes and receives plain ``None``.
    - ``"fail"``: the run fails loudly, naming the origin.

    None means "use the default for this input's required-ness"."""

    def __post_init__(self) -> None:
        if self.list_count_of is not None:
            _validate_structural_id(self.list_count_of, "input list count source")
            if self.type != TypeExpr.concrete(CORE_INT) or self.pin:
                raise ValueError(f"input {self.id}: list_count_of needs an INT field with no pin")
        if type(self.pin) is not bool:
            raise ValueError(f"input {self.id}: pin must be a bool")
        if not self.pin and self.pin_first:
            raise ValueError(f"input {self.id}: pin=False conflicts with pin_first")
        if type(self.sent) is not bool:
            raise ValueError(f"input {self.id}: sent must be a bool")
        if not self.sent and self.pin:
            raise ValueError(
                f"input {self.id}: a field that is not sent cannot be wired; "
                "declare it with pin=False"
            )
        if type(self.hidden) is not bool:
            raise ValueError(f"input {self.id}: hidden must be a bool")
        if type(self.on_edit) is not tuple:
            raise ValueError(f"input {self.id}: on_edit is a tuple of steps")
        from .body import check_automatic, check_flag_or_formula, check_text_or_formula

        check_text_or_formula(self.label, f"input {self.id} label")
        check_flag_or_formula(self.visible, f"input {self.id} visible")
        check_flag_or_formula(self.read_only, f"input {self.id} read_only")
        check_flag_or_formula(self.disabled, f"input {self.id} disabled")
        check_text_or_formula(self.disabled_reason, f"input {self.id} disabled_reason")
        check_automatic(self)
        if self.after_run is not None:
            if self.after_run not in AFTER_RUN_MODES:
                raise ValueError(
                    f"input {self.id}: after_run is one of {', '.join(AFTER_RUN_MODES)}"
                )
            shown = (
                tuple(item.widget for item in self.widget.representations)
                if isinstance(self.widget, WidgetRepresentations)
                else (self.widget,)
            )
            numeric = self.type in (
                TypeExpr.concrete(CORE_INT),
                TypeExpr.concrete(CORE_FLOAT),
                TypeExpr.concrete(CORE_COMBO),
            )
            if not all(
                isinstance(widget, (NumberWidget, ComboWidget)) or (widget is None and numeric)
                for widget in shown
            ):
                raise ValueError(f"input {self.id}: after_run needs a number or combo field")
        if type(self.lazy) is not bool:
            raise ValueError(f"input {self.id}: lazy must be a bool")
        if type(self.accepts_storage) is not bool:
            raise ValueError(f"input {self.id}: accepts_storage must be a bool")
        if type(self.accepts_stream) is not bool:
            raise ValueError(f"input {self.id}: accepts_stream must be a bool")
        if type(self.accepts_list) is not bool:
            raise ValueError(f"input {self.id}: accepts_list must be a bool")
        _validate_media_policy(self.alpha_policy, self.mask_polarity, self.mask_semantic)
        descriptors = (
            tuple(item.widget for item in self.widget.representations)
            if isinstance(self.widget, WidgetRepresentations)
            else (self.widget,)
        )
        if self.source_filename is not None:
            source_type = TypeExpr.concrete(ASSET_TYPE)
            typed_asset = self.type.kind == "asset" and self.type.runtime_type_id() is not None
            if not typed_asset and self.type not in (source_type, TypeExpr.list_of(source_type)):
                raise ValueError(
                    f"input {self.id}: source filename requires a scalar asset or list of assets"
                )
            if (
                not isinstance(self.widget, AssetWidget)
                or not self.widget.allow_upload
                or self.widget.kind != self.source_filename.kind
            ):
                raise ValueError(
                    f"input {self.id}: source filename requires an upload-enabled ASSET widget "
                    "with the same media kind"
                )
        elif any(isinstance(widget, AssetWidget) and widget.allow_upload for widget in descriptors):
            # Applies to typed assets too: the frontend rejects any
            # upload-enabled ASSET widget without a source filename binding,
            # which silently drops the node from the catalog.
            raise ValueError(
                f"input {self.id}: upload-enabled ASSET widget requires a source filename binding"
            )
        if self.on_absent is not None and self.on_absent not in ABSENT_POLICIES:
            raise ValueError(f"input {self.id}: unknown on_absent {self.on_absent!r}")
        if self.on_absent == "omit" and self.required:
            raise ValueError(
                f"input {self.id}: on_absent='omit' requires an optional input "
                "(a required execute() parameter cannot be omitted)"
            )
        # Every representation binds to this same canonical socket/value
        # domain. The wrapper contributes no execution state of its own.
        if isinstance(self.widget, WidgetRepresentations):
            for representation in self.widget.representations:
                _validate_widget_binding(self.id, self.type, representation.widget)
        elif self.widget is not None:
            _validate_widget_binding(self.id, self.type, self.widget)
        default: object = self.default
        if (
            self.type == TypeExpr.concrete(CORE_COMBO)
            and default is not None
            and not (
                type(default) in (str, bool)
                or (type(default) is int and abs(default) <= _JSON_SAFE_INT)
                or (type(default) is float and math.isfinite(default))
            )
        ):
            raise ValueError(
                f"input {self.id}: combo default must be a string, boolean, JSON-safe integer "
                "or finite float"
            )
        if (
            any(isinstance(widget, MultiComboWidget) for widget in descriptors)
            and (
                type(default) is not list
                or any(type(value) is not str for value in cast("list[object]", default))
            )
            and default is not None
        ):
            raise ValueError(f"input {self.id}: multi combo default must be an array of strings")

    def absent_policy(self) -> AbsentPolicy:
        """The effective policy: the declaration, or the sane default -
        skip for required inputs, omit (treat as unconnected) for optional."""
        if self.on_absent is not None:
            return self.on_absent
        return "skip" if self.required else "omit"


def _freeze_applies(
    applies: Mapping[str, tuple[str, ...]] | None,
    subject: str,
) -> Mapping[str, tuple[str, ...]] | None:
    if applies is None:
        return None
    if not isinstance(cast("object", applies), Mapping):
        raise ValueError(f"{subject} applies must be a mapping")
    raw_applies = dict(cast("Mapping[object, object]", applies))
    if not raw_applies:
        raise ValueError(f"{subject} applies must not be empty; omit it instead")
    checked: dict[str, tuple[str, ...]] = {}
    for combo_id, raw_values in raw_applies.items():
        if not isinstance(combo_id, str) or not combo_id:
            raise ValueError(f"{subject} applies keys must be non-empty combo ids")
        if isinstance(raw_values, str) or not isinstance(raw_values, Sequence):
            raise ValueError(f"{subject} applies[{combo_id!r}] must be a sequence of options")
        values = tuple(cast("Sequence[object]", raw_values))
        if not values:
            raise ValueError(f"{subject} applies[{combo_id!r}] must cover at least one option")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{subject} applies[{combo_id!r}] options must be non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError(f"{subject} applies[{combo_id!r}] has duplicate options")
        checked[combo_id] = cast("tuple[str, ...]", values)
    return MappingProxyType(checked)


@dataclass(frozen=True)
class OutputRepresents:
    """One selected asset that can estimate an output before execution.

    ``input`` names a top-level asset-widget input. ``rendition`` names the
    client-side conversion from that asset to this output; clients render
    only rendition kinds they understand. ``applies`` limits the promise to
    declared required combo values. This is presentation metadata only:
    estimates never become execution values and authoritative outputs always
    replace them.
    """

    input: str
    rendition: str
    applies: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.input), str) or not self.input:
            raise ValueError("output represents input must be a non-empty string")
        if not isinstance(cast("object", self.rendition), str) or not self.rendition:
            raise ValueError("output represents rendition must be a non-empty string")
        object.__setattr__(
            self,
            "applies",
            _freeze_applies(self.applies, "output represents"),
        )


KNOWN_VALUE_LIMITS: Final = ("min", "max", "step", "default")
"""The limits a known value may name, in their wire order."""


@dataclass(frozen=True)
class OutputListCount:
    """An output's item count follows an input, without reading its values.

    The output and its input or group belong to the node's top-level scope.
    With ``group``, the output joins that group's ``input`` across members:
    a scalar contributes one and a list contributes its item count. An
    unknown list count makes the total unknown. Without ``group``, the
    output preserves the named list input's count.
    """

    input: str
    group: str | None = None

    def __post_init__(self) -> None:
        _validate_structural_id(self.input, "output list count input")
        if self.group is not None:
            _validate_structural_id(self.group, "output list count group")


@dataclass(frozen=True)
class OutputKnownValue:
    """An output equal to one typed primitive input before execution.

    The input and output share one of INT, FLOAT, STRING and BOOLEAN, or the
    input is a COMBO and the output is a COMBO or a STRING, or both are typed
    by the same type variable. A variable's known value applies only once the
    variable is bound to one of those types; otherwise the node runs.

    ``limits`` maps ``min``, ``max``, ``step`` and ``default`` to other optional
    inputs of the value's type that hold them, so the editor can show the
    value's own range and reset value. ``min``, ``max`` and ``step`` need an INT
    or FLOAT value.
    """

    input: str
    limits: Mapping[str, str] = field(
        default_factory=lambda: cast("Mapping[str, str]", MappingProxyType({})), hash=False
    )

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.input), str) or not self.input:
            raise ValueError("output known value input must be a non-empty string")
        raw = cast("object", self.limits)
        if not isinstance(raw, Mapping):
            raise ValueError("output known value limits must be a mapping")
        limits = cast("Mapping[object, object]", raw)
        for key, target in limits.items():
            if key not in KNOWN_VALUE_LIMITS:
                raise ValueError(f"unknown output known value limit: {key!r}")
            if not isinstance(target, str) or not target:
                raise ValueError(f"output known value limit {key} must name an input")
        object.__setattr__(
            self,
            "limits",
            MappingProxyType({key: limits[key] for key in KNOWN_VALUE_LIMITS if key in limits}),
        )


def _validate_media_policy(alpha: str, polarity: str | None, semantic: str | None) -> None:
    if alpha not in {"preserve", "require", "create_if_missing", "drop"}:
        raise ValueError(f"unknown alpha policy: {alpha!r}")
    if polarity is not None and polarity not in {"coverage", "transparency"}:
        raise ValueError(f"unknown mask polarity: {polarity!r}")
    if semantic is not None and semantic not in {"alpha", "selection", "other"}:
        raise ValueError(f"unknown mask semantic: {semantic!r}")


@dataclass(frozen=True)
class Output:
    """Outputs have real ids, never positions (no RETURN_TYPES tuples).

    ``preview`` is narrow final-output discovery intent. It helps clients
    choose an already-produced output to inspect after a run; it does not
    request execution, alter scheduling/cache identity, select a renderer,
    or describe the separate ephemeral ``report_preview`` channel.
    """

    id: str
    type: TypeExpr
    tooltip: str = ""
    optional: bool = False
    """This output may deliberately carry no value at runtime (DESIGN 3.15):
    execute() returns the ABSENT marker for it (a loader with no VAE, a
    detector with no match). Declared, so the frontend can render the edge
    as maybe-absent and validation can warn when it feeds a fail-policy
    input. Only optional outputs may return ABSENT - anything else is a
    contract error at the worker."""
    preview: bool = False
    """Generic final-output intent/discovery only. It does not promise a
    renderer, rendition, retention, purity, cacheability, or liveness."""
    represents: OutputRepresents | None = None
    """Optional pre-execution estimate sourced from one selected asset.
    Presentation only: never submitted, hashed, scheduled, or executed."""
    known_value: OutputKnownValue | None = None
    """Optional exact primitive identity sourced from one top-level input."""
    label: TextOrFormula = ""
    """Presentation only, text or a formula; empty means "derive from the id"."""
    visible: FlagOrFormula = True
    """Shows or hides the pin, by a flag or a formula; a hidden output is
    still produced and still linkable."""
    alpha_policy: Literal["preserve", "require", "create_if_missing", "drop"] = "preserve"
    mask_polarity: Literal["coverage", "transparency"] | None = None
    mask_semantic: Literal["alpha", "selection", "other"] | None = None
    list_count: int | OutputListCount | Formula | None = None
    """An item count, its input source, or a formula, separate from the type.
    File-derived formula answers are estimates, not document-known counts.
    Editor metadata only: never an execution value or part of cache identity."""

    def __post_init__(self) -> None:
        from .body import Formula, check_flag_or_formula, check_text_or_formula

        _validate_media_policy(self.alpha_policy, self.mask_polarity, self.mask_semantic)
        check_text_or_formula(self.label, f"output {self.id} label")
        check_flag_or_formula(self.visible, f"output {self.id} visible")
        if type(self.preview) is not bool:
            raise ValueError(f"output {self.id}: preview must be a bool")
        if self.represents is not None and not isinstance(
            cast("object", self.represents), OutputRepresents
        ):
            raise ValueError(f"output {self.id}: represents must be an OutputRepresents")
        if self.known_value is not None and not isinstance(
            cast("object", self.known_value), OutputKnownValue
        ):
            raise ValueError(f"output {self.id}: known_value must be an OutputKnownValue")
        if self.list_count is not None:
            if self.type.kind != "list":
                raise ValueError(f"output {self.id}: list_count needs a list output")
            if not isinstance(self.list_count, (OutputListCount, Formula)) and not (
                type(self.list_count) is int and 0 <= self.list_count <= 2**53 - 1
            ):
                raise ValueError(
                    f"output {self.id}: list_count needs a safe nonnegative integer "
                    "or OutputListCount or Formula"
                )


RESERVED_INPUT_IDS = frozenset({"execution_context", "input_values"})
"""Framework-reserved execute() parameter names. ``execution_context`` carries
live run control; ``input_values`` provides read-only input envelopes. These
are not row ids."""

SearchVisibility = Literal["normal", "deprecated", "hidden"]
"""How node search lists a type. Presentation/lifecycle metadata, never
validity: a "hidden" node is fully instantiable and executable in existing
workflows - it just stops being offered for new ones. Distinct axis from
``NodeSchema.deprecation``: a node can be deprecated-but-listed (demoted in
search, badge visible) or hidden outright while keeping a genuinely unique
behavior alive."""

SEARCH_VISIBILITIES: frozenset[str] = frozenset({"normal", "deprecated", "hidden"})


@dataclass(frozen=True)
class Deprecation:
    """Structured author-declared deprecation for a node type.

    ``message`` is author prose for the deprecation badge popover ("going
    away soon, use X because Y"). ``replacement`` names the successor node
    type for a click-to-replace affordance - naming only; how a document
    migrates (input mappings, guards) is replacement-rule data, not this.
    Closed data, never code, and never part of the schema signature:
    deprecating a node must not invalidate its caches."""

    message: str
    since: str = ""  # pack version that declared it, e.g. "2.4.0"
    replacement: str = ""  # successor node_type

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("deprecation requires a message")


@dataclass(frozen=True)
class OutputProbeSpec:
    """Bind a descriptor document to a revisioned, host-owned asset probe."""

    input: str
    kind: str
    revision: str

    def __post_init__(self) -> None:
        for value in (self.input, self.kind, self.revision):
            _validate_structural_id(value, "output probe field")
        if self.kind != "model":
            raise ValueError("output probe kind must be 'model'")


VARIANT_KEY_PATTERN = STRUCTURAL_ID_PATTERN
"""A type choice's variant keys."""
OPTION_KEY_PATTERN = re.compile(r"[!-~]+( [!-~]+)*")
"""A choice's option keys admit printable non-space ASCII tokens
separated by single interior spaces (upstream BFL keys "Flux.2 [pro]" and
"Flux.2 [max]"); no leading, trailing, or consecutive spaces; keys are
exact-string identity - never trimmed, case-folded, or normalized."""


def _type_variables(expr: TypeExpr) -> Iterable[TypeExpr]:
    if expr.kind == "variable":
        yield expr
    elif expr.kind in ("list", "asset", "stream"):
        assert expr.element is not None
        yield from _type_variables(expr.element)
    elif expr.kind == "tuple":
        for position in expr.elements:
            yield from _type_variables(position)


@dataclass(frozen=True)
class SlotValue:
    """What execute() receives for a dynamic slot: the connected value, the
    active variant key (explicit dispatch - node code never sniffs runtime
    types), and the variant's dependent inputs under their LOCAL names."""

    variant: str
    value: object
    options: Mapping[str, object] = field(default_factory=dict[str, object])


@dataclass(frozen=True)
class SelectorSpec:
    """Engine-owned routing: the ``input`` row picks one member of the values
    group ``group`` and the node's one output passes that member on. The
    node's ``execute`` never runs.

    The body holds a top-level ``input`` typed ``INT | BOOLEAN``, a top-level
    group ``group`` of lazy inputs that keeps at least one member, and exactly
    one output whose type every member shares.

    The members are the branches, in the node's stored order. A stored pick is
    a member key; a key that is no longer a member, or no pick, picks the
    first member. A wired pick, or a stored whole number such as a known
    value a client wrote in, is a position counting from 0; a boolean picks
    between exactly two members, false first. Any other position fails the
    Switch. A picked member with nothing plugged in passes nothing on.

    At submission ``lower_selectors`` resolves every top-level Switch whose
    pick is stored and removes the other members' exclusive cones before
    validation and asset checks; a wired pick keeps every member and the
    engine demands only the picked one.

    ``names`` optionally names a top-level hidden ``list<STRING>`` input that
    the client fills with the branch names, in member order, when the pick
    is a text or choice wire whose value is not known before the run. The
    pick input is then typed ``INT | BOOLEAN | STRING | COMBO``. Text on a
    wire picks the branch with exactly that name, then the member with that
    key, and otherwise fails the Switch; it never falls back to the first
    member."""

    input: str
    group: str
    names: str | None = None

    def __post_init__(self) -> None:
        values = cast("tuple[object, ...]", (self.input, self.group))
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError("selector input and group must be non-empty strings")
        if self.input == self.group:
            raise ValueError("selector input and group must be distinct")
        if self.names is not None and (
            not isinstance(cast("object", self.names), str)
            or not self.names
            or self.names in values
        ):
            raise ValueError(
                "selector names must be a non-empty string distinct from its input and group"
            )

    def members(self, inputs: Iterable[Input]) -> tuple[str, ...]:
        """The flattened input ids of the members of an effective interface."""
        prefix = f"{self.group}."
        return tuple(spec.id for spec in inputs if spec.id.startswith(prefix))

    def pick(
        self,
        members: Sequence[str],
        value: object,
        *,
        linked: bool = False,
        names: Sequence[str] | None = None,
    ) -> str:
        """The flattened input id ``value`` picks among ``members``.

        ``linked`` says the pick arrived over a wire; ``names`` are the branch
        names in member order, when the client sent them.

        Raises ``ValueError`` with the message the Switch shows on its pick."""
        count = len(members)
        if not count:
            raise ValueError("Switch has no branches.")
        branches = "There is 1 branch" if count == 1 else f"There are {count} branches"
        if value is None:
            return members[0]
        if isinstance(value, str):
            member = f"{self.group}.{value}"
            if not linked:
                return member if member in members else members[0]
            if names is not None and len(names) != count:
                raise ValueError(f"Switch received {len(names)} branch names for {count} branches.")
            if names is not None and value in names:
                return members[list(names).index(value)]
            if member in members:
                return member
            shown = (
                list(names) if names is not None else [m[len(self.group) + 1 :] for m in members]
            )
            listed = shown[0] if count == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]
            verb = "The branch is" if count == 1 else "The branches are"
            raise ValueError(f'Switch received "{value}". {verb} {listed}.')
        if type(value) is bool:
            if count != 2:
                raise ValueError(
                    f"Switch received {str(value).lower()}, which picks between exactly two "
                    f"branches. {branches}; wire a position from 0 to {count - 1} instead."
                )
            return members[int(value)]
        if type(value) is int:
            if not 0 <= value < count:
                positions = {1: "0", 2: "0 or 1"}.get(count, f"0 to {count - 1}")
                raise ValueError(f"Switch received position {value}. {branches}; use {positions}.")
            return members[value]
        raise ValueError(
            f"Switch received {type(value).__name__}; a pick is a branch, a position or "
            "true or false."
        )


MirrorKind = Literal["glsl", "expression"]
"""How a frontend renders a mirror estimate: a GLSL ES 3.00 fragment shader
carried inline in the declaration, or the deterministic expression grammar
identified by its grammar version."""

MIRROR_KINDS: frozenset[str] = frozenset({"glsl", "expression"})

MirrorPrecision = Literal["exact", "bounded"]
"""How closely a mirror estimate matches the authoritative result: ``exact``
promises bit-identical IEEE-754 binary64 results (restricted to the
correctly-rounded operation subset), ``bounded`` promises agreement within
the declared tolerance."""

MIRROR_PRECISIONS: frozenset[str] = frozenset({"exact", "bounded"})

MAX_MIRROR_SOURCE_BYTES = 16384
"""Upper bound on inline mirror shader source (UTF-8 bytes)."""


@dataclass(frozen=True)
class MirrorTolerance:
    """Declared agreement bound for a bounded-precision mirror.

    ``relative`` bounds scalar results: |estimate - authoritative| <=
    relative * max(|estimate|, |authoritative|). ``per_channel`` bounds image
    results: maximum absolute per-channel delta after quantization at preview
    resolution. At least one bound is required; a declaration may state both
    when a mirror produces both result shapes."""

    relative: float | None = None
    per_channel: float | None = None

    def __post_init__(self) -> None:
        if self.relative is None and self.per_channel is None:
            raise ValueError("mirror tolerance requires relative or per_channel")
        for name in ("relative", "per_channel"):
            value = getattr(self, name)
            if value is None:
                continue
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"mirror tolerance {name} must be a finite positive float")


@dataclass(frozen=True)
class MirrorSpec:
    """A frontend-renderable mirror of this node's transform.

    Presentation metadata only: a client MAY run the declared mirror over
    client-resident inputs to render an instant preview estimate. The node's
    ``execute()`` remains the sole authoritative implementation - nothing in
    the engine, scheduler, or worker boundary reads this field, mirrored
    results never become outputs or values, and authoritative results always
    replace estimates. Like ``emits_previews``, never part of the schema
    signature: declaring a mirror must not invalidate caches.

    ``kind`` selects the mirror runtime. ``expression`` mirrors carry
    ``grammar_version`` (the deterministic expression grammar the client
    must implement); ``glsl`` mirrors carry ``source`` (an inline GLSL ES
    3.00 fragment shader, at most ``MAX_MIRROR_SOURCE_BYTES`` UTF-8 bytes).
    ``precision`` states the parity contract; ``bounded`` mirrors must
    declare a ``tolerance`` and ``exact`` mirrors must not.

    A GLSL mirror binds an input with a combo widget, and a required choice,
    as ``uniform int <id>``: the index of the selected value, or of the
    chosen option in the choice's declared order. A row inside the chosen
    option binds by its own declared id. An optional choice with nothing
    chosen gives no estimate.

    ``applies`` scopes the mirror to a subset of the node's choices: each
    key names a required choice at the top of the body and maps to the
    option keys the mirror covers. A client renders an estimate only when
    every named choice's chosen option is in its covered set, and renders
    none otherwise - a mirror has no abstain channel of its own, so the
    declaration is what keeps a partially mirrorable node (some operations
    shader-expressible, others not) from producing wrong estimates. An
    absent mapping means the mirror covers the node's whole input space."""

    kind: MirrorKind
    precision: MirrorPrecision
    tolerance: MirrorTolerance | None = None
    grammar_version: int | None = None
    source: str | None = None
    applies: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.kind), str) or self.kind not in MIRROR_KINDS:
            raise ValueError(f"unknown mirror kind: {self.kind!r}")
        if (
            not isinstance(cast("object", self.precision), str)
            or self.precision not in MIRROR_PRECISIONS
        ):
            raise ValueError(f"unknown mirror precision: {self.precision!r}")
        if self.tolerance is not None and not isinstance(
            cast("object", self.tolerance), MirrorTolerance
        ):
            raise ValueError("mirror tolerance must be a MirrorTolerance")
        if self.source is not None and not isinstance(cast("object", self.source), str):
            raise ValueError("mirror shader source must be a string")
        if self.precision == "bounded" and self.tolerance is None:
            raise ValueError("bounded mirrors require a tolerance")
        if self.precision == "exact" and self.tolerance is not None:
            raise ValueError("exact mirrors must not declare a tolerance")
        if self.kind == "expression":
            if self.source is not None:
                raise ValueError("expression mirrors must not carry shader source")
            if type(self.grammar_version) is not int or self.grammar_version < 1:
                raise ValueError("expression mirrors require a positive grammar_version")
        if self.kind == "glsl":
            if self.grammar_version is not None:
                raise ValueError("glsl mirrors must not carry a grammar_version")
            if not self.source:
                raise ValueError("glsl mirrors require shader source")
            if len(self.source.encode("utf-8")) > MAX_MIRROR_SOURCE_BYTES:
                raise ValueError(f"mirror shader source exceeds {MAX_MIRROR_SOURCE_BYTES} bytes")
        if self.applies is not None:
            object.__setattr__(self, "applies", _freeze_applies(self.applies, "mirror"))


def _validate_applies_against_combos(
    node_type: str,
    subject: str,
    applies: Mapping[str, tuple[str, ...]],
    choices: Mapping[str, tuple[bool, tuple[str, ...]]],
) -> None:
    """``choices`` maps each top-level choice to whether it is required and its option keys."""
    for applies_id, covered in applies.items():
        choice = choices.get(applies_id)
        if choice is None:
            raise ValueError(
                f"{node_type}: {subject} applies key {applies_id!r} does not name a declared choice"
            )
        required, keys = choice
        if not required:
            raise ValueError(
                f"{node_type}: {subject} applies key {applies_id!r} must name a required choice"
            )
        declared_keys = set(keys)
        undeclared = sorted(set(covered) - declared_keys)
        if undeclared:
            raise ValueError(
                f"{node_type}: {subject} applies[{applies_id!r}] covers "
                f"undeclared option(s): {', '.join(undeclared)}"
            )


class _BodyRows[T](tuple[T, ...]):
    """The top-level rows a NodeSchema derives from its body."""

    __slots__ = ()


@dataclass(frozen=True)
class NodeSchema:
    node_type: str
    version: int = 1
    display_name: str = ""
    category: str = ""
    description: str = ""
    editor_role: str | None = None
    """Frontend capability role. Presentation metadata only."""
    inputs: tuple[Input, ...] = ()
    outputs: tuple[Output, ...] = ()
    slot_choices: tuple[tuple[str, str], ...] = ()
    """The chosen key of each choice and type choice, set ONLY on elaborated
    schemas. Part of the wire form and therefore of the schema signature -
    the chosen option is computation identity even when two options share
    an interface shape (hazard H4)."""
    idempotent: bool = True  # False -> never cached (side-effecting nodes)
    occupies: tuple[str, ...] = ()
    """Abstract resource kinds this node's execution occupies while running,
    e.g. ("gpu",). A declaration, never code (hazard H12): how many nodes may
    occupy a kind concurrently is engine/placement configuration (default: 1
    per kind - most GPUs run one node at a time; a powerful one can be
    configured higher). Kinds stay abstract here because the concrete
    instance is a fact of the *inputs*, not the node type: which GPU a
    sampler occupies depends on where the model value it receives lives, so
    the engine binds each kind to the instance(s) declared in the input
    envelopes' resources meta at admission time. An admission hint only - it
    never joins the schema signature or cache keys (hazard H4: the same
    computation yields the same result wherever and however it is
    scheduled)."""
    io_bound: bool = False
    """This node's execution waits (network, external service) rather than
    computes - the partner/API-node case. io_bound nodes are exempt from the
    engine's implicit "compute" admission lane, so they overlap freely (still
    bounded by max_concurrency) while ordinary local nodes keep the limited
    default. Mutually exclusive with occupies: a node either waits, occupies
    declared hardware, or shares the default compute lane. Scheduling
    metadata like occupies - never part of the schema signature."""
    dispatch_affinity: Literal["native"] | None = None
    """Execution-arm affinity declared by the schema owner. ``native``
    selects a native body when one is available; absence leaves ordinary
    owner planning intact. Scheduling metadata only - never part of the
    schema signature or cache keys."""
    deprecation: Deprecation | None = None
    """Author-declared deprecation (badge + click-to-replace affordance).
    Lifecycle metadata: like search_visibility, never part of the schema
    signature - deprecating a node must not invalidate its caches."""
    search_visibility: SearchVisibility = "normal"
    """How node search lists this type; see SearchVisibility. Orthogonal to
    deprecation by design (deprecated-but-listed and hidden-but-valid are
    both real states), and never validity: any visibility executes."""
    replacements: tuple[ReplacementRule, ...] = ()
    """Declarative migration rules this schema carries, keyed by each rule's
    ``from_type`` (the predecessor). A successor's schema is the natural
    carrier, but a rule's cases name their own targets - guarded fan-out to
    multiple successors is legitimate, so cases are never constrained to this
    schema's node_type. Closed serializable data executed by the frontend's
    replacement engine (mirror of Dinkster-Frontend replace/model.ts); like
    deprecation, lifecycle metadata that never joins the schema signature -
    shipping migration advice must not invalidate caches."""
    search_terms: tuple[str, ...] = ()
    """Extra names node search also matches when offering this type
    (ComfyUI's search-alias concept): legacy class names, synonyms,
    common misspellings. Discovery metadata only, frontend-consumed:
    search terms never resolve anything, never
    join the schema signature, and collisions across schemas are
    meaningless - many nodes may legitimately answer to "blur"."""
    output_node: bool = False
    """This node acts as a sink/output of a workflow (v1 OUTPUT_NODE, a
    save/preview node). Submission formats without explicit targets select
    these nodes. Region bodies also select them each iteration, and may omit
    exported outputs when they contain a declared sink. Not part of the
    schema signature."""
    emits_previews: bool = False
    """This node's execution emits live previews (report_preview during
    sampling or other long-running work). One declaration, two consumers:
    the frontend shows per-node preview affordances only for flagged types,
    and the engine resolves the effective preview mode to "off" for
    unflagged nodes so no preview emitter, provider matching, or decode
    work is ever set up for them. A node that emits previews at runtime
    without declaring this flag loses both (accepted tradeoff of the
    declarative design). Capability metadata only: never validity and
    never part of the schema signature."""
    may_expand_graph: bool = False
    """Execution may return a validated runtime graph expansion and inhibit
    nodes in the run overlay. Capability metadata, not cache identity."""
    admission_validation: bool = False
    """Ask the owning native worker to validate literal values and declared
    linked-input types before producers execute. Opt-in only: nodes without
    this hook retain advisory declared-type mismatch behavior."""
    execution_context: bool = False
    """Inject the hidden run-scoped execution_context parameter. Such nodes
    observe live run state and therefore never use invocation-result caches."""
    input_values: bool = False
    """Inject read-only input envelopes without live run-control access.
    Execute may omit decoded input parameters to handle only their envelopes.
    Caching includes metadata throughout each input value tree."""
    mirror: MirrorSpec | None = None
    """Optional frontend-renderable mirror of this node's transform (see
    MirrorSpec). Presentation metadata only: the engine never reads it,
    mirrored estimates never become outputs, and ``execute()`` stays the
    sole authoritative implementation. Never part of the schema signature -
    declaring, changing, or removing a mirror must not invalidate caches."""
    selector: SelectorSpec | None = None
    chunk_safe: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    """Media ranges; ordinary inputs promise independent, cardinality-preserving execution."""
    chunk_safe_applies: Mapping[str, tuple[str, ...]] | None = None
    additional_inputs: TypeExpr | None = None
    """Opt-in type declaration for otherwise undeclared input names.

    Supplied names elaborate to ordinary optional inputs. Declared dynamic
    namespaces remain reserved even when their inputs are inactive. Wildcard
    literals require explicit registered type stamps, just like any other
    wildcard input; this declaration never bypasses value admission.
    """
    body: tuple[BodyRow, ...] = ()
    """The node's rows as a tree (``dinkster_schema.body``); elaboration flattens it."""
    tables: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}), hash=False)
    """Constants formulas may name: nested dictionaries and lists of plain values."""
    derived: Mapping[str, Formula] = field(default_factory=lambda: MappingProxyType({}), hash=False)
    """Named formulas other formulas use by name."""
    validate: tuple[Rule, ...] = ()
    """Rules about the node as a whole, shown on its title.

    When a run is queued, a rule whose check is false on the stored values
    fails the run before any node runs, or, when the rule declares a fix,
    leaves a warning on the node and the run goes on; the engine never runs
    the fix. A check that reads a wired input has no answer then and passes."""
    title_suffix: TextOrFormula = ""
    """Text or a formula added to the node's title."""
    menu: tuple[Button, ...] = ()
    """Buttons offered as items of the node's menu."""
    effective: bool = False
    """True for the interface ``elaborate`` works out for one node: flat
    ``inputs`` and ``outputs`` and no body. A declared schema writes its rows
    in ``body``."""

    def _applies_choices(self) -> dict[str, tuple[bool, tuple[str, ...]]]:
        """The top-level choices an ``applies`` key may name."""
        if self.body:
            from .body import Choice, scope_rows

            return {
                row.id: (row.required, tuple(option.key for option in row.options))
                for row in scope_rows(self.body)
                if isinstance(row, Choice)
            }
        return {}

    def _check_body(self) -> None:
        from .body import check_body

        has_body = bool(self.body or self.validate or self.menu or self.title_suffix)
        has_body = has_body or bool(self.tables or self.derived)
        if type(self.effective) is not bool:
            raise ValueError(f"{self.node_type}: effective must be a bool")
        if not has_body and not self.effective:
            for name in ("inputs", "outputs"):
                rows = getattr(self, name)
                if rows and not isinstance(rows, _BodyRows):
                    raise ValueError(
                        f"{self.node_type}: {name}= is not a declaration; "
                        "write the rows in body=(...)"
                    )
        if has_body:
            if self.effective:
                raise ValueError(f"{self.node_type}: an effective schema has no body")
            from .body import top_level_rows

            inputs, outputs = top_level_rows(self.body)

            def stated(rows: tuple[object, ...], derived: tuple[object, ...]) -> bool:
                # Rows derived from a body come back through dataclasses.replace.
                return not isinstance(rows, _BodyRows) and rows not in ((), derived)

            if stated(self.inputs, inputs) or stated(self.outputs, outputs):
                raise ValueError(
                    f"{self.node_type}: declares body or the separate row lists, not both"
                )
            object.__setattr__(self, "inputs", _BodyRows(inputs))
            object.__setattr__(self, "outputs", _BodyRows(outputs))
        object.__setattr__(self, "tables", MappingProxyType(dict(self.tables)))
        object.__setattr__(self, "derived", MappingProxyType(dict(self.derived)))
        if has_body:
            try:
                check_body(
                    self.body,
                    tables=self.tables,
                    derived=self.derived,
                    validate=self.validate,
                    title_suffix=self.title_suffix,
                    menu=self.menu,
                )
            except ValueError as exc:
                raise ValueError(f"{self.node_type}: {exc}") from None

    def _check_selector(self, selector: SelectorSpec) -> None:
        from .body import Group, scope_rows

        if len(self.outputs) != 1:
            raise ValueError(f"{self.node_type}: selector schemas require exactly one output")
        output_type = self.outputs[0].type
        pick = next((spec for spec in self.inputs if spec.id == selector.input), None)
        pick_types = {CORE_INT, CORE_BOOLEAN}
        if selector.names is not None:
            pick_types |= {CORE_STRING, CORE_COMBO}
        if pick is None or pick.type.kind != "union" or set(pick.type.types) != pick_types:
            expected = " | ".join(sorted(pick_types))
            raise ValueError(f"{self.node_type}: selector input must name a {expected} union input")
        if selector.names is not None:
            names = next((spec for spec in self.inputs if spec.id == selector.names), None)
            in_body = self.effective or any(
                isinstance(row, Input) and row.id == selector.names for row in self.body
            )
            if (
                names is None
                or not in_body
                or not names.hidden
                or names.required
                or names.widget is not None
                or names.type != TypeExpr.list_of(TypeExpr.concrete(CORE_STRING))
            ):
                raise ValueError(
                    f"{self.node_type}: selector names {selector.names!r} must name a top-level "
                    "hidden, not required list<STRING> input with no widget, outside any "
                    "group, choice or section"
                )
        if self.effective:
            member_types = {
                spec.type for spec in self.inputs if spec.id in selector.members(self.inputs)
            }
        else:
            group = next(
                (
                    row
                    for row in scope_rows(self.body)
                    if isinstance(row, Group) and row.id == selector.group
                ),
                None,
            )
            if group is None or not isinstance(group.value, Input) or not group.value.lazy:
                raise ValueError(
                    f"{self.node_type}: selector group must name a top-level group of lazy inputs"
                )
            if group.min_members < 1:
                raise ValueError(f"{self.node_type}: a selector group keeps at least one branch")
            if _body_has_structure(self.body, allowed=group):
                raise ValueError(f"{self.node_type}: a selector body holds only rows and its group")
            member_types = {group.value.type}
        if member_types - {output_type}:
            raise ValueError(f"{self.node_type}: selector branches must share the output type")

    def __post_init__(self) -> None:
        if not self.node_type:
            raise ValueError("node_type is required")
        self._check_body()
        if self.additional_inputs is not None and not isinstance(
            cast("object", self.additional_inputs), TypeExpr
        ):
            raise ValueError("additional_inputs must be a TypeExpr")
        if self.additional_inputs is not None and self.additional_inputs.requires_elaboration:
            raise ValueError("additional_inputs cannot depend on another input")
        if type(self.execution_context) is not bool:
            raise ValueError("execution_context must be a bool")
        if type(self.input_values) is not bool:
            raise ValueError("input_values must be a bool")
        if self.editor_role is not None and (
            not isinstance(cast("object", self.editor_role), str) or not self.editor_role
        ):
            raise ValueError(f"{self.node_type}: editor_role must be a non-empty string")
        if self.chunk_safe is not None:
            inputs, outputs = self.chunk_safe
            inputs, outputs = tuple(inputs), tuple(outputs)
            object.__setattr__(self, "chunk_safe", (inputs, outputs))
            if not inputs or not outputs:
                raise ValueError("chunk_safe requires input and output IDs")
            if len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
                raise ValueError("chunk_safe IDs must be unique")
            if not set(inputs) <= {spec.id for spec in self.inputs}:
                raise ValueError("chunk_safe names an unknown input")
            if not set(outputs) <= {spec.id for spec in self.outputs}:
                raise ValueError("chunk_safe names an unknown output")
        if self.chunk_safe_applies is not None:
            if self.chunk_safe is None:
                raise ValueError("chunk_safe_applies requires chunk_safe")
            object.__setattr__(
                self, "chunk_safe_applies", _freeze_applies(self.chunk_safe_applies, "chunk_safe")
            )
            _validate_applies_against_combos(
                self.node_type, "chunk_safe", self.chunk_safe_applies, self._applies_choices()
            )
        if self.search_visibility not in SEARCH_VISIBILITIES:
            raise ValueError(
                f"{self.node_type}: unknown search_visibility: {self.search_visibility!r}"
            )
        if self.deprecation is not None and self.deprecation.replacement == self.node_type:
            raise ValueError(f"{self.node_type}: a node cannot replace itself")
        if self.mirror is not None and not isinstance(cast("object", self.mirror), MirrorSpec):
            raise ValueError(f"{self.node_type}: mirror must be a MirrorSpec")
        if self.mirror is not None and self.mirror.applies is not None:
            # A key that names nothing (or covers an option the combo never
            # declared) would make clients gate estimates on a value that can
            # never match - silently rendering none (or wrong ones after a
            # rename). Scoping is limited to required combos so elaboration
            # always sees a consumed choice to resolve the scope against: it
            # drops the mirror when the choice is uncovered and strips
            # ``applies`` when it is covered. A schema without the named combo
            # is therefore always a construction error - elaborated schemas
            # never carry an applies key.
            _validate_applies_against_combos(
                self.node_type,
                "mirror",
                self.mirror.applies,
                self._applies_choices(),
            )
        if len(set(self.search_terms)) != len(self.search_terms):
            raise ValueError(f"{self.node_type}: duplicate search terms")
        for term in self.search_terms:
            if not term:
                raise ValueError(f"{self.node_type}: search terms must be non-empty strings")
        if len(set(self.occupies)) != len(self.occupies):
            raise ValueError(f"{self.node_type}: duplicate occupies kinds")
        if self.io_bound and self.occupies:
            raise ValueError(f"{self.node_type}: io_bound and occupies are mutually exclusive")
        if self.dispatch_affinity not in (None, "native"):
            raise ValueError(
                f"{self.node_type}: unknown dispatch_affinity: {self.dispatch_affinity!r}"
            )
        for kind in self.occupies:
            if not kind:
                raise ValueError(f"{self.node_type}: occupies kinds must be non-empty strings")
        input_ids = [spec.id for spec in self.inputs]
        output_ids = [spec.id for spec in self.outputs]
        if len(set(input_ids)) != len(input_ids):
            raise ValueError(f"{self.node_type}: duplicate input ids")
        if len(set(output_ids)) != len(output_ids):
            raise ValueError(f"{self.node_type}: duplicate output ids")
        if self.selector is not None:
            self._check_selector(self.selector)
        reserved = RESERVED_INPUT_IDS & set(input_ids)
        if reserved:
            raise ValueError(
                f"{self.node_type}: reserved input id(s): {', '.join(sorted(reserved))}"
            )
        choice_ids = [slot_id for slot_id, _ in self.slot_choices]
        if len(set(choice_ids)) != len(choice_ids):
            raise ValueError(f"{self.node_type}: duplicate slot choice ids")
        if self.slot_choices and not self.effective:
            raise ValueError(f"{self.node_type}: slot_choices belong to elaborated schemas")
        inputs_by_id = {spec.id: spec for spec in self.inputs}

        def validate_output_type(expr: TypeExpr) -> None:
            if expr.kind == "tuple_element":
                if expr.input_id and expr.input_id not in inputs_by_id:
                    raise ValueError("tuple element references an unknown tuple input")
                if isinstance(expr.index, str):
                    selector = inputs_by_id.get(expr.index)
                    if selector is None or selector.type != TypeExpr.concrete(CORE_INT):
                        raise ValueError("tuple element selector must reference an INT input")
            if expr.element is not None:
                validate_output_type(expr.element)
            for position in expr.elements:
                validate_output_type(position)

        for spec in self.inputs:
            if spec.type.requires_elaboration:
                raise ValueError("input types cannot reference another input")
        for output in self.outputs:
            validate_output_type(output.type)
        self._check_options_inputs(inputs_by_id)
        count_inputs = self.inputs
        if not self.effective:
            from .body import all_rows, scope_rows

            count_inputs = tuple(row for row in scope_rows(self.body) if isinstance(row, Input))
            for row in scope_rows(self.body):
                if isinstance(row, (Input, Output)):
                    continue
                for nested in all_rows((row,)):
                    if (isinstance(nested, Input) and nested.list_count_of is not None) or (
                        isinstance(nested, Output)
                        and isinstance(nested.list_count, OutputListCount)
                    ):
                        raise ValueError(
                            f"{self.node_type}: count sources require top-level input/output rows"
                        )
        for input_spec in count_inputs:
            if input_spec.list_count_of is not None:
                source = inputs_by_id.get(input_spec.list_count_of)
                if source is None or source.type.kind != "list":
                    raise ValueError(
                        f"{self.node_type}: input {input_spec.id!r} "
                        "list_count_of must name a list input"
                    )
        for output in self.outputs:
            count = output.list_count
            if isinstance(count, OutputListCount):
                if count.group is None:
                    source = inputs_by_id.get(count.input)
                    if source is None or source.type.kind != "list":
                        raise ValueError(
                            f"{self.node_type}: output {output.id!r} "
                            "list_count must name a list input"
                        )
                elif not self.effective:
                    from .body import Group, scope_rows

                    group = next(
                        (
                            row
                            for row in scope_rows(self.body)
                            if isinstance(row, Group) and row.id == count.group
                        ),
                        None,
                    )
                    source = (
                        None
                        if group is None
                        else next(
                            (
                                row
                                for row in scope_rows(group.member or ())
                                if isinstance(row, Input) and row.id == count.input
                            ),
                            group.value
                            if isinstance(group.value, Input) and group.value.id == count.input
                            else None,
                        )
                    )
                    if source is None:
                        raise ValueError(
                            f"{self.node_type}: output {output.id!r} "
                            "list_count must name an input in its group"
                        )
            known_value = output.known_value
            if known_value is not None:
                known_input = inputs_by_id.get(known_value.input)
                if known_input is None:
                    raise ValueError(
                        f"{self.node_type}: output {output.id!r} known value input "
                        f"{known_value.input!r} is not a declared top-level input"
                    )
                primitive_types = {
                    TypeExpr.concrete(CORE_INT),
                    TypeExpr.concrete(CORE_FLOAT),
                    TypeExpr.concrete(CORE_STRING),
                    TypeExpr.concrete(CORE_BOOLEAN),
                }
                combo = TypeExpr.concrete(CORE_COMBO)
                same_primitive = known_input.type == output.type and output.type in primitive_types
                # A chosen option is its text, so a choice may also be known as a STRING.
                from_choice = known_input.type == combo and output.type in {
                    combo,
                    TypeExpr.concrete(CORE_STRING),
                }
                # The editor applies a variable's known value only once the
                # variable is bound to INT, FLOAT, STRING, BOOLEAN or COMBO.
                same_variable = known_input.type == output.type and output.type.kind == "variable"
                if not (same_primitive or from_choice or same_variable):
                    raise ValueError(
                        f"{self.node_type}: output {output.id!r} known value must reference "
                        "an input with the same concrete primitive type or type variable, "
                        "or a COMBO input for a COMBO or STRING output"
                    )
                numeric = {TypeExpr.concrete(CORE_INT), TypeExpr.concrete(CORE_FLOAT)}
                named: set[str] = set()
                for limit, target in known_value.limits.items():
                    if limit != "default" and known_input.type not in numeric:
                        raise ValueError(
                            f"{self.node_type}: output {output.id!r} known value limit "
                            f"{limit} needs an INT or FLOAT value"
                        )
                    holder = inputs_by_id.get(target)
                    if (
                        holder is None
                        or target == known_value.input
                        or holder.required
                        or holder.type != known_input.type
                        or target in named
                    ):
                        raise ValueError(
                            f"{self.node_type}: output {output.id!r} known value limit "
                            f"{limit} must name another optional input of the value's type, "
                            "and no input twice"
                        )
                    named.add(target)
            represents = output.represents
            if represents is None:
                continue
            represented_input = inputs_by_id.get(represents.input)
            if represented_input is None:
                raise ValueError(
                    f"{self.node_type}: output {output.id!r} represents input "
                    f"{represents.input!r} is not a declared top-level input"
                )
            widget = represented_input.widget
            has_asset_widget = isinstance(widget, AssetWidget) or (
                isinstance(widget, WidgetRepresentations)
                and all(
                    isinstance(representation.widget, AssetWidget)
                    for representation in widget.representations
                )
            )
            if not has_asset_widget:
                raise ValueError(
                    f"{self.node_type}: output {output.id!r} represents input "
                    f"{represents.input!r} must use an asset widget"
                )
            if represents.applies is not None:
                _validate_applies_against_combos(
                    self.node_type,
                    f"output {output.id!r} represents",
                    represents.applies,
                    self._applies_choices(),
                )

    def _check_options_inputs(self, inputs_by_id: Mapping[str, Input]) -> None:
        from .body import all_rows

        rows = (*self.inputs, *(row for row in all_rows(self.body) if isinstance(row, Input)))
        for row in rows:
            widget = row.widget
            widgets = (
                tuple(item.widget for item in widget.representations)
                if isinstance(widget, WidgetRepresentations)
                else (widget,)
            )
            for combo in widgets:
                if not isinstance(combo, ComboWidget) or not combo.options_input:
                    continue
                target = inputs_by_id.get(combo.options_input)
                if (
                    target is None
                    or not target.hidden
                    or target.required
                    or target.widget is not None
                    or target.type != TypeExpr.list_of(TypeExpr.concrete(CORE_STRING))
                ):
                    raise ValueError(
                        f"{self.node_type}: input {row.id!r} options_input "
                        f"{combo.options_input!r} must name a top-level hidden, not required "
                        "list<STRING> input with no widget"
                    )

    @property
    def is_static(self) -> bool:
        return (
            self.additional_inputs is None
            and not any(output.type.requires_elaboration for output in self.outputs)
            and not _body_has_structure(self.body)
        )

    def input(self, input_id: str) -> Input | None:
        for spec in self.inputs:
            if spec.id == input_id:
                return spec
        return None

    def output(self, output_id: str) -> Output | None:
        for spec in self.outputs:
            if spec.id == output_id:
                return spec
        return None


def _body_has_structure(body: Sequence[object], *, allowed: object = None) -> bool:
    from .body import Button, Editor, Note, Row, Section, State

    for row in body:
        if isinstance(row, (Section, Row)):
            if _body_has_structure(row.rows, allowed=allowed):
                return True
        elif row is not allowed and not isinstance(
            row, (Input, Output, State, Note, Button, Editor)
        ):
            return True
    return False


def _remote_choice_ids(schema: NodeSchema) -> tuple[str, ...]:
    """Collect remote COMBO ids from every recursive schema input position."""
    found: list[str] = []

    def visit_widget(widget: Widget | None) -> None:
        if isinstance(widget, ComboWidget) and widget.remote_route:
            found.append(_remote_choice_id(widget.remote_route))
        elif isinstance(widget, WidgetRepresentations):
            for representation in widget.representations:
                visit_widget(representation.widget)

    from .body import Choice, Row, Section, Use, all_rows

    def visit_option_rows(rows: Sequence[object]) -> None:
        for row in rows:
            if isinstance(row, Use) and isinstance(
                row.widget, (ComboWidget, WidgetRepresentations)
            ):
                visit_widget(row.widget)
            elif isinstance(row, (Section, Row)):
                visit_option_rows(row.rows)

    for spec in schema.inputs if not schema.body else ():
        visit_widget(spec.widget)
    for row in all_rows(schema.body):
        if isinstance(row, Input):
            visit_widget(row.widget)
        elif isinstance(row, Choice):
            for option in row.options:
                visit_option_rows(option.rows)
    return tuple(found)


def validate_remote_choice_authority(
    schemas: Mapping[str, NodeSchema],
    choices: Mapping[str, Sequence[object]],
    *,
    owner: str,
) -> None:
    """Require every published remote route to be owned by the same delta."""
    for node_type, schema in schemas.items():
        for choice_id in _remote_choice_ids(schema):
            if choice_id not in choices:
                raise ValueError(
                    f"{owner}: node {node_type!r} remote choice {choice_id!r} "
                    "is not registered by the same owner"
                )
