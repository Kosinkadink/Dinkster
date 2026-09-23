"""Node schema model - the single source of truth (hazard H1).

The engine, validator, cache-key builder, wire encoder, and isolation layer
all consume these objects. There is no INPUT_TYPES-style dict anywhere.

The type-expression model is deliberately small and closed (one canonical
representation, mirroring Dinkster-Frontend's TypeExpr):
concrete | union | wildcard | variable | list | asset | stream.

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
from typing import Literal, cast

# The list and asset type-id grammars are owned by dinkster-values (type ids
# are registry namespace); importing them here is the one place schema
# reaches sideways in the bottom layer, so expression-level types and
# runtime type ids can never drift apart.
from dinkster_values import (
    CustomWidgetDescriptor,
    asset_type_id,
    list_type_id,
    parse_asset_type_id,
    parse_list_type_id,
)
from dinkster_values import (
    runtime_type_atom as _runtime_type_atom,
)
from dinkster_values.streams import parse_stream_type_id, stream_type_id

from .names import validate_name
from .replace import ReplacementRule

TypeExprKind = Literal["concrete", "union", "wildcard", "variable", "list", "asset", "stream"]

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

    def __post_init__(self) -> None:
        if self.kind not in (
            "concrete",
            "union",
            "wildcard",
            "variable",
            "list",
            "asset",
            "stream",
        ):
            raise ValueError(f"unknown TypeExpr kind: {self.kind!r}")
        if self.kind == "concrete":
            if len(self.types) != 1:
                raise ValueError("concrete TypeExpr requires exactly one type id")
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
        elif self.element is not None:
            raise ValueError(f"{self.kind} TypeExpr carries no element expression")

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
    def stream_of(element: TypeExpr) -> TypeExpr:
        return TypeExpr(kind="stream", element=element)

    @staticmethod
    def asset_of(element: TypeExpr) -> TypeExpr:
        """An asset that decodes to ``element`` (typed assets, joint contract
        2026-07-26). The VALUE is one AssetRef - what the decode yields is
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
        if self.kind == "stream":
            assert self.element is not None
            inner = self.element.runtime_type_id()
            return None if inner is None else stream_type_id(inner)
        if self.kind == "list":
            assert self.element is not None
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
        if self.kind in ("concrete", "union", "asset", "stream"):
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


@dataclass(frozen=True)
class SaveTargetWidget:
    """Presentation request for a save-destination picker: choose a
    readwrite mount and a relative prefix (subfolders + filename stem).
    The value it produces is a ``dinkster.save_target`` wire mapping
    {"mount": ..., "prefix": ...} - structured, never a raw path string.

    ``suffix`` is advisory display data (what extension the node will
    append, e.g. ".png"); the node owns the actual format."""

    suffix: str = ""

    def __post_init__(self) -> None:
        if not self.suffix:
            return
        if not self.suffix.startswith(".") or "/" in self.suffix or "\\" in self.suffix:
            raise ValueError("save target widget suffix must be a bare extension like '.png'")


@dataclass(frozen=True)
class ComboOption:
    """One combo value plus optional dropdown presentation metadata."""

    value: str
    label: str | None = None
    info: str | None = None
    folder: str | None = None

    def __post_init__(self) -> None:
        if type(self.value) is not str or not self.value:
            raise ValueError("combo option value must be a non-empty string")
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


def _validate_static_combo_options(options: tuple[str | ComboOption, ...], *, subject: str) -> None:
    if not isinstance(cast("object", options), tuple) or any(
        not (type(option) is ComboOption or (type(option) is str and bool(option)))
        for option in cast("tuple[object, ...]", options)
    ):
        raise ValueError(f"{subject} options must be non-empty strings or ComboOption values")


@dataclass(frozen=True)
class InputFamilyOptionSource:
    """Combo choices derived from one top-level dynamic input family.

    Each family member suffix is the stable stored selection identity used by
    graph links, execution, and cache keys. The authored document owns a
    separate editable display name for that member; only that label is
    presentation data.
    Clients must reject duplicate display names or visibly disambiguate them;
    labels are never selection identity.
    """

    input_family: str

    def __post_init__(self) -> None:
        _validate_structural_id(self.input_family, "combo option source input family")


@dataclass(frozen=True)
class ComboWidget:
    """Presentation request for a combo-choice dropdown.

    The socket has concrete ``core.combo`` identity while its runtime value,
    literal, cache payload, and node-facing object stay a plain string.
    Choice vocabulary is never identity within combos: different option
    lists remain compatible, and a stored value outside the current options
    is the frontend's diagnostic to surface, not a schema error. This
    descriptor never joins schema signatures or execution identity. InputSpec
    enforces the concrete core.combo binding.

    ``options`` are static choices, rendered immediately. ``option_source``
    derives choices from the current members of a top-level input family:
    member suffixes are stable values and occurrence-authored member names are
    labels; clients reject or visibly disambiguate duplicate labels and never
    select by them. ``remote_route``
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
    ``control_after_generate`` is the optional closed initial controller
    mode, with the same client-owned live-state semantics as NumberWidget."""

    options: tuple[str | ComboOption, ...] = ()
    option_source: InputFamilyOptionSource | None = None
    remote_route: str = ""
    refresh_button: bool = False
    control_after_generate: ControlAfterGenerate | None = None
    control_after_refresh: Literal["first", "last"] | None = None
    remote_timeout_ms: int | None = None
    remote_max_retries: int | None = None
    remote_refresh_ms: int | None = None

    def __post_init__(self) -> None:
        _validate_static_combo_options(self.options, subject="combo widget")
        if self.option_source is not None and not isinstance(
            cast("object", self.option_source), InputFamilyOptionSource
        ):
            raise ValueError("combo widget option_source must be an InputFamilyOptionSource")
        if self.option_source is not None and (self.options or self.remote_route):
            raise ValueError(
                "combo widget input-family option source cannot be combined with "
                "static or remote options"
            )
        if self.remote_route:
            _remote_choice_id(self.remote_route)
        if not self.options and not self.remote_route and self.option_source is None:
            raise ValueError(
                "combo widget needs static options, a remote route, or an "
                "input-family option source"
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
        if self.control_after_generate is not None and (
            type(self.control_after_generate) is not str
            or self.control_after_generate not in CONTROL_AFTER_GENERATE
        ):
            raise ValueError(
                "combo widget control_after_generate must be one of "
                f"{sorted(CONTROL_AFTER_GENERATE)}, "
                f"got {self.control_after_generate!r}"
            )


@dataclass(frozen=True)
class MultiComboWidget:
    """Presentation request for a list-valued combo selector.

    Unlike ``ComboWidget``, this descriptor owns one canonical
    ``list<core.combo>`` value. Options and remote policy are presentation
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
        _validate_static_combo_options(self.options, subject="multi combo widget")
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

    Only meaningful on ``core.boolean`` inputs. A label-less boolean
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


ControlAfterGenerate = Literal["fixed", "increment", "decrement", "randomize"]
"""The INITIAL mode of a control-after-generate selector: what the frontend
does to the widget value after each submitted run. Presence of the field on a
NumberWidget or ComboWidget means "render the control"; the field seeds the
mode, never overwrites it - the live mode is client-owned mutable state, so
later schema reads must not reset a user's chosen mode.

The vocabulary is closed for v11: any addition is a wire bump, not an
in-place grow. The backend treats an unknown mode as a decode error (we
refuse to emit garbage); the frontend deliberately warns and drops the
malformed field while still rendering the number widget (it refuses to let
received garbage hide UI). That strict-emit/tolerant-render asymmetry is
the pinned contract."""

CONTROL_AFTER_GENERATE: frozenset[str] = frozenset({"fixed", "increment", "decrement", "randomize"})

# The largest integer a JSON double represents exactly. Larger integer
# constraints use canonical decimal strings.
_JSON_SAFE_INT = 2**53 - 1
_DECIMAL_WIRE_INT_MIN = -(2**63)
_DECIMAL_WIRE_INT_MAX = 2**64 - 1


@dataclass(frozen=True)
class NumberWidget:
    """Presentation request for a numeric input's editor.

    Only meaningful on ``core.int``/``core.float`` inputs; the socket type
    decides integer vs float rendering, so one widget kind covers both.
    All constraints are UI vocabulary: the runtime value stays a plain
    number and out-of-range stored values are the frontend's diagnostic to
    surface, never a schema error. Absent min/max means unbounded. Integer
    constraints through signed 64-bit minimum and unsigned 64-bit maximum are
    lossless as canonical decimal strings outside the JSON-double-safe range.

    ``display`` is an explicit editor presentation when present. Bounds and
    step are independent constraints: they never imply a slider or any other
    presentation. ``round`` is an independent finite positive rendering
    precision available only on core.float; it never implies display or
    changes bounds/step. ``control_after_generate`` names the INITIAL mode
    of the seed-style control-after-generate selector; setting it asks the
    frontend to render that control. Absence of the descriptor means "no
    presentation metadata", and the frontend still renders a number editor
    by primitive type inference; it never means the input is widgetless
    (pinned with the frontend).

    Socket binding: this descriptor is only
    legal on concrete ``core.int``/``core.float`` inputs, and constraints
    are interpreted in the socket's domain - an int socket with a
    fractional min/max/step is malformed. Enforced by InputSpec, which is
    the first place the widget meets its socket type."""

    min: int | float | None = None
    max: int | float | None = None
    step: int | float | None = None
    control_after_generate: ControlAfterGenerate | None = None
    display: Literal["number", "slider", "knob", "gradientslider"] | None = None
    round: int | float | None = None

    def __post_init__(self) -> None:
        if (
            self.min is None
            and self.max is None
            and self.step is None
            and self.round is None
            and self.control_after_generate is None
            and self.display is None
        ):
            raise ValueError(
                "number widget needs at least one constraint, display, or the "
                "control-after-generate mode (a metadata-less numeric "
                "input carries no widget)"
            )
        for name in ("min", "max", "step"):
            bound = getattr(self, name)
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
        if self.round is not None and (
            type(self.round) not in (int, float) or not math.isfinite(self.round) or self.round <= 0
        ):
            raise ValueError(
                f"number widget round must be a finite positive number, got {self.round!r}"
            )
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"number widget min ({self.min}) must not exceed max ({self.max})")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"number widget step must be positive, got {self.step}")
        if (
            self.control_after_generate is not None
            and self.control_after_generate not in CONTROL_AFTER_GENERATE
        ):
            raise ValueError(
                "number widget control_after_generate must be one of "
                f"{sorted(CONTROL_AFTER_GENERATE)}, "
                f"got {self.control_after_generate!r}"
            )
        if self.display is not None and (
            type(self.display) is not str
            or self.display not in {"number", "slider", "knob", "gradientslider"}
        ):
            raise ValueError(
                "number widget display must be one of "
                "['gradientslider', 'knob', 'number', 'slider'], "
                f"got {self.display!r}"
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
    """Static candidates plus current member names from dynamic input families."""

    items: tuple[TextCompletionItem, ...] = ()
    input_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.items), tuple) or any(
            type(item) is not TextCompletionItem for item in cast("tuple[object, ...]", self.items)
        ):
            raise ValueError("text completion items must be an immutable tuple")
        if not isinstance(cast("object", self.input_families), tuple):
            raise ValueError("text completion input_families must be an immutable tuple")
        if not self.items and not self.input_families:
            raise ValueError("text completions need at least one item or input family")
        if len(set(self.input_families)) != len(self.input_families):
            raise ValueError("text completion input_families must be unique")
        for family in self.input_families:
            _validate_structural_id(family, "text completion input family")


@dataclass(frozen=True)
class StringWidget:
    """Presentation request for a string editor.

    Only legal on concrete ``core.string`` inputs (enforced by InputSpec).
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
    completions: TextCompletions | None = None

    def __post_init__(self) -> None:
        if self.multiline is not None and type(self.multiline) is not bool:
            raise ValueError("string widget multiline must be a bool")
        if self.placeholder is not None and type(self.placeholder) is not str:
            raise ValueError("string widget placeholder must be a string")
        if self.dynamic_prompts is not None and type(self.dynamic_prompts) is not bool:
            raise ValueError("string widget dynamic_prompts must be a bool")
        if self.completions is not None and type(self.completions) is not TextCompletions:
            raise ValueError("string widget completions must be TextCompletions")
        if (
            self.multiline is None
            and self.placeholder is None
            and self.dynamic_prompts is None
            and self.completions is None
        ):
            raise ValueError("string widget needs at least one presentation field")


@dataclass(frozen=True)
class ColorWidget:
    """Fieldless color-picker presentation over a canonical core.string value."""


@dataclass(frozen=True)
class CurveWidget:
    """Fieldless curve-editor presentation over a canonical dinkster.curve value."""


@dataclass(frozen=True)
class CompositorWidget:
    """Fieldless compositor presentation over a canonical dinkster.compositor value."""


WidgetDescriptor = (
    CustomWidgetDescriptor
    | AssetWidget
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
        return "asset"
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

    if isinstance(widget, NumberWidget):
        socket = input_type.types[0] if input_type.kind == "concrete" else None
        if socket not in ("core.int", "core.float"):
            raise ValueError(
                f"input {input_id}: number widget requires a concrete "
                f"core.int or core.float input, got {input_type.kind} "
                f"{socket or ''}".rstrip()
            )
        if socket == "core.int":
            for name in ("min", "max", "step"):
                bound = getattr(widget, name)
                if bound is not None and bound != int(bound):
                    raise ValueError(
                        f"input {input_id}: number widget {name} must be integral "
                        f"on a core.int input, got {bound}"
                    )
                if isinstance(bound, float) and abs(bound) > _JSON_SAFE_INT:
                    raise ValueError(
                        f"input {input_id}: number widget {name} must be exactly representable "
                        f"on a core.int input, got {bound}"
                    )
        elif socket == "core.float":
            for name in ("min", "max", "step"):
                bound = getattr(widget, name)
                if isinstance(bound, int) and abs(bound) > _JSON_SAFE_INT:
                    raise ValueError(
                        f"input {input_id}: number widget {name} must fit the JSON-double-safe "
                        f"integer range on a core.float input, got {bound}"
                    )
        if widget.round is not None and socket != "core.float":
            raise ValueError(
                f"input {input_id}: number widget round requires a concrete core.float input"
            )
    if isinstance(widget, StringWidget) and input_type != TypeExpr.concrete("core.string"):
        raise ValueError(f"input {input_id}: string widget requires a concrete core.string input")
    if isinstance(widget, ColorWidget) and input_type != TypeExpr.concrete("core.string"):
        raise ValueError(f"input {input_id}: color widget requires a concrete core.string input")
    if isinstance(widget, CurveWidget) and input_type != TypeExpr.concrete("dinkster.curve"):
        raise ValueError(f"input {input_id}: curve widget requires a concrete dinkster.curve input")
    if isinstance(widget, CompositorWidget) and input_type != TypeExpr.concrete(
        "dinkster.compositor"
    ):
        raise ValueError(
            f"input {input_id}: compositor widget requires a concrete dinkster.compositor input"
        )
    if isinstance(widget, BooleanWidget) and input_type != TypeExpr.concrete("core.boolean"):
        raise ValueError(f"input {input_id}: boolean widget requires a concrete core.boolean input")
    if isinstance(widget, ComboWidget) and input_type != TypeExpr.concrete("core.combo"):
        raise ValueError(f"input {input_id}: combo widget requires a concrete core.combo input")
    if isinstance(widget, MultiComboWidget) and input_type != TypeExpr.list_of(
        TypeExpr.concrete("core.combo")
    ):
        raise ValueError(f"input {input_id}: multi combo widget requires a list<core.combo> input")
    if isinstance(widget, AssetWidget) and not (
        input_type == TypeExpr.concrete("dinkster.asset")
        or (input_type.kind == "asset" and input_type.runtime_type_id() is not None)
        or input_type.kind == "concrete"
        or (
            widget.allow_upload
            and input_type == TypeExpr.list_of(TypeExpr.concrete("dinkster.asset"))
        )
    ):
        # Asset destinations receive one ref directly. Concrete scalar
        # destinations receive an asset-stamped literal that the worker
        # decodes (or a stamped list that it terminally merges). The exact
        # upload-enabled list<dinkster.asset> spelling supports ordered source
        # uploads; other lists and unresolved asset variables cannot determine
        # a contract.
        raise ValueError(
            f"input {input_id}: asset widget requires a concrete "
            "scalar, bare dinkster.asset, upload-enabled list<dinkster.asset>, or "
            "fully-concrete asset<...> input; other list inputs and "
            "asset<variable> are not supported"
        )
    if isinstance(widget, SaveTargetWidget) and input_type != TypeExpr.concrete(
        "dinkster.save_target"
    ):
        raise ValueError(
            f"input {input_id}: save target widget requires a concrete dinkster.save_target input"
        )


@dataclass(frozen=True)
class InputSpec:
    id: str
    type: TypeExpr
    required: bool = True
    default: object = None
    doc: str = ""
    on_absent: AbsentPolicy | None = None
    widget: Widget | None = None
    display_name: str = ""
    force_input: bool = False
    advanced: bool = False
    hidden: bool = False
    lazy: bool = False
    source_filename: SourceFilenameSpec | None = None
    accepts_storage: bool = False
    accepts_stream: bool = False
    alpha_policy: Literal["preserve", "require", "create_if_missing", "drop"] = "preserve"
    mask_polarity: Literal["coverage", "transparency"] | None = None
    mask_semantic: Literal["alpha", "selection", "other"] | None = None
    """Presentation-only label for this input; empty means "derive from the
    id", exactly as before the field existed. Like doc and widget it never
    joins schema signatures or execution identity (hazard H15: presentation
    is pixels, never identity).

    ``lazy`` marks an input whose linked producer is deferred until the node's
    worker-bound hook requests it. Unlike the nearby presentation fields, it
    changes scheduling and therefore joins schema signatures.

    ``accepts_storage`` declares that the consumer accepts compact storage
    dtypes without input conversion. It joins execution identity only when true.

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
        if type(self.hidden) is not bool:
            raise ValueError(f"input {self.id}: hidden must be a bool")
        if type(self.lazy) is not bool:
            raise ValueError(f"input {self.id}: lazy must be a bool")
        if type(self.accepts_storage) is not bool:
            raise ValueError(f"input {self.id}: accepts_storage must be a bool")
        if type(self.accepts_stream) is not bool:
            raise ValueError(f"input {self.id}: accepts_stream must be a bool")
        _validate_media_policy(self.alpha_policy, self.mask_polarity, self.mask_semantic)
        descriptors = (
            tuple(item.widget for item in self.widget.representations)
            if isinstance(self.widget, WidgetRepresentations)
            else (self.widget,)
        )
        if self.source_filename is not None:
            source_type = TypeExpr.concrete("dinkster.asset")
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


@dataclass(frozen=True)
class OutputKnownValue:
    """An output equal to one typed primitive input before execution."""

    input: str

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.input), str) or not self.input:
            raise ValueError("output known value input must be a non-empty string")


def _validate_media_policy(alpha: str, polarity: str | None, semantic: str | None) -> None:
    if alpha not in {"preserve", "require", "create_if_missing", "drop"}:
        raise ValueError(f"unknown alpha policy: {alpha!r}")
    if polarity is not None and polarity not in {"coverage", "transparency"}:
        raise ValueError(f"unknown mask polarity: {polarity!r}")
    if semantic is not None and semantic not in {"alpha", "selection", "other"}:
        raise ValueError(f"unknown mask semantic: {semantic!r}")


@dataclass(frozen=True)
class OutputSpec:
    """Outputs have real ids, never positions (no RETURN_TYPES tuples).

    ``preview`` is narrow final-output discovery intent. It helps clients
    choose an already-produced output to inspect after a run; it does not
    request execution, alter scheduling/cache identity, select a renderer,
    or describe the separate ephemeral ``report_preview`` channel.
    """

    id: str
    type: TypeExpr
    doc: str = ""
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
    display_name: str = ""
    alpha_policy: Literal["preserve", "require", "create_if_missing", "drop"] = "preserve"
    mask_polarity: Literal["coverage", "transparency"] | None = None
    mask_semantic: Literal["alpha", "selection", "other"] | None = None

    def __post_init__(self) -> None:
        _validate_media_policy(self.alpha_policy, self.mask_polarity, self.mask_semantic)
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


RESERVED_INPUT_IDS = frozenset({"output_spec"})
"""Framework-reserved execute() parameter names. ``output_spec`` carries the
elaborated output interface (an OutputInterface) to nodes that declare output
families; it is never an input id or input family id."""

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


MEMBER_SEP = "."
"""Separator between a family id and a member suffix in member input ids.

A member of family ``operands`` stored as ``operands.x`` keeps the id
``operands.x`` forever - member identity is the stored id, never an ordinal
(hazard H10). The engine never renumbers members.
"""


@dataclass(frozen=True, init=False)
class InputFamilySpec:
    """An autogrow input family (dynamic inputs, hazard H10).

    The schema declares the family; the document's stored keys/links determine
    which members exist. Elaboration (see elaborate.py) turns members into
    ordinary InputSpecs on an effective schema, so nothing downstream branches
    on "is this dynamic".
    """

    id: str
    template: tuple[DynamicEntry, ...]
    min_members: int = 0
    max_members: int | None = None
    doc: str = ""
    display_name: str = ""
    required: bool = True
    member_prefix: str | None = None
    member_names: tuple[str, ...] | None = None

    def __init__(
        self,
        id: str,
        template: tuple[DynamicEntry, ...] | TypeExpr,
        min_members: int = 0,
        max_members: int | None = None,
        doc: str = "",
        display_name: str = "",
        required: bool = True,
        member_prefix: str | None = None,
        member_names: tuple[str, ...] | None = None,
    ) -> None:
        if member_names is not None and max_members is not None:
            raise ValueError(f"family {id}: max_members is derived from member_names")
        object.__setattr__(self, "id", id)
        object.__setattr__(
            self,
            "template",
            (InputSpec("value", template),) if isinstance(template, TypeExpr) else template,
        )
        object.__setattr__(self, "min_members", min_members)
        object.__setattr__(
            self,
            "max_members",
            len(member_names) if member_names is not None else max_members,
        )
        object.__setattr__(self, "doc", doc)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "member_prefix", member_prefix)
        object.__setattr__(self, "member_names", member_names)
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_structural_id(self.id, "input family id")
        if not self.template:
            raise ValueError(f"family {self.id}: template must not be empty")
        _validate_dynamic_entries(self.template, f"family '{self.id}' template")
        if self.min_members < 0:
            raise ValueError(f"family {self.id}: min_members must be >= 0")
        if self.member_prefix is not None and self.member_names is not None:
            raise ValueError(
                f"family {self.id}: member_prefix and member_names are mutually exclusive"
            )
        if self.member_prefix is not None:
            _validate_structural_id(self.member_prefix, "input family member_prefix")
            if self.max_members is not None and not 1 <= self.max_members <= 1000:
                raise ValueError(f"family {self.id}: prefix max_members must be in [1, 1000]")
        if self.member_names is not None:
            if len(set(self.member_names)) != len(self.member_names):
                raise ValueError(f"family {self.id}: duplicate member_names")
            for name in self.member_names:
                _validate_structural_id(name, "input family member name")
            if self.min_members > len(self.member_names):
                raise ValueError(f"family {self.id}: min_members exceeds member_names capacity")
        elif self.max_members is not None and self.max_members < self.min_members:
            raise ValueError(f"family {self.id}: max_members < min_members")

    @property
    def type(self) -> TypeExpr:
        """Compatibility shorthand for the normalized one-input template."""
        if len(self.template) == 1 and isinstance(self.template[0], InputSpec):
            return self.template[0].type
        raise AttributeError("a grouped input family has no single type")

    def member_id(self, suffix: str) -> str:
        return f"{self.id}{MEMBER_SEP}{suffix}"

    def member_suffix(self, input_id: str) -> str | None:
        """The member suffix if input_id belongs to this family, else None."""
        prefix = self.id + MEMBER_SEP
        if input_id.startswith(prefix) and len(input_id) > len(prefix):
            return input_id[len(prefix) :]
        return None


@dataclass(frozen=True)
class OutputCountSpec:
    """Bind a family's members to a stored integer input.

    ``index`` yields the canonical suffixes ``0`` through ``count - 1``.
    """

    input: str
    suffix: Literal["index"] = "index"

    def __post_init__(self) -> None:
        _validate_structural_id(self.input, "output family count input")
        if self.suffix != "index":
            raise ValueError("output family count suffix must be 'index'")


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


@dataclass(frozen=True)
class OutputDescriptorsSpec:
    """An ordered output interface declared by stored JSON entries, never execution."""

    input: str
    choices: tuple[OutputSpec, ...]
    max_entries: int
    min_entries: int = 0
    fixed_ids: bool = False
    probe: OutputProbeSpec | None = None

    def __post_init__(self) -> None:
        _validate_structural_id(self.input, "output descriptors input")
        if type(self.min_entries) is not int or type(self.max_entries) is not int:
            raise ValueError("output descriptor bounds must be integers")
        if not 0 <= self.min_entries <= self.max_entries <= 512:
            raise ValueError("output descriptor bounds must satisfy 0 <= min <= max <= 512")
        if type(self.fixed_ids) is not bool:
            raise ValueError("output descriptors fixed_ids must be a bool")
        if self.probe is not None and not self.fixed_ids:
            raise ValueError("probed output descriptors require fixed semantic ids")
        if not isinstance(cast("object", self.choices), tuple) or not 1 <= len(self.choices) <= 512:
            raise ValueError(
                "output descriptor choices must be an immutable tuple of 1-512 choices"
            )
        ids: set[str] = set()
        for choice in self.choices:
            _validate_structural_id(choice.id, "output descriptor choice id")
            if choice.id in ids:
                raise ValueError("output descriptor choices must have unique ids")
            ids.add(choice.id)
            if choice.type.kind != "concrete":
                raise ValueError("output descriptor choices must have concrete runtime types")
            if choice.represents is not None or choice.known_value is not None:
                raise ValueError("output descriptor choices cannot declare input representations")


@dataclass(frozen=True)
class OutputFamilySpec:
    """A dynamic output family (hazard H10), symmetric with InputFamilySpec.

    Membership is **document-determined**: the graph node stores an ordered
    member-suffix list per family (GraphNode.output_members), and elaboration
    turns it into ordinary OutputSpecs before validation and planning. A node
    can never change its interface by executing - data-dependent cardinality
    belongs in a collection-typed value on one static output. ``preview`` is
    inherited by every elaborated member as final-output discovery intent; it
    never changes execution, scheduling, cache identity, rendering, retention,
    or the separate ephemeral ``report_preview`` channel. ``count`` constrains
    membership to canonical zero-based suffixes derived from a stored integer
    input.
    """

    id: str
    type: TypeExpr
    min_members: int = 0
    max_members: int | None = None
    doc: str = ""
    preview: bool = False
    """Generic final-output intent/discovery for every elaborated member."""
    count: OutputCountSpec | None = None

    def __post_init__(self) -> None:
        if type(self.preview) is not bool:
            raise ValueError(f"output family {self.id}: preview must be a bool")
        if MEMBER_SEP in self.id:
            raise ValueError(f"family id may not contain '{MEMBER_SEP}': {self.id}")
        if self.min_members < 0:
            raise ValueError(f"family {self.id}: min_members must be >= 0")
        if self.max_members is not None and self.max_members < self.min_members:
            raise ValueError(f"family {self.id}: max_members < min_members")

    def member_id(self, suffix: str) -> str:
        return f"{self.id}{MEMBER_SEP}{suffix}"

    def member_suffix(self, output_id: str) -> str | None:
        """The member suffix if output_id belongs to this family, else None."""
        prefix = self.id + MEMBER_SEP
        if output_id.startswith(prefix) and len(output_id) > len(prefix):
            return output_id[len(prefix) :]
        return None


STRUCTURAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
VARIANT_KEY_PATTERN = STRUCTURAL_ID_PATTERN
"""Structural dynamic identifiers are schema-authored, stable, and wire-safe."""
OPTION_KEY_PATTERN = re.compile(r"[!-~]+( [!-~]+)*")
"""Dynamic combo option keys only; widened a second time on 2026-07-29 by
joint backend/frontend decision to admit printable non-space ASCII tokens
separated by single interior spaces (upstream BFL keys "Flux.2 [pro]" and
"Flux.2 [max]"); no leading, trailing, or consecutive spaces; keys are
exact-string identity - never trimmed, case-folded, or normalized."""


def _validate_structural_id(value: str, field_name: str) -> None:
    if not STRUCTURAL_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must match [A-Za-z0-9_-]+: {value!r}")


def _validate_dynamic_entries(entries: tuple[DynamicEntry, ...], owner: str) -> None:
    allowed = (InputSpec, InputFamilySpec, DynamicComboSpec, DynamicSlotSpec)
    for raw_entry in cast("tuple[object, ...]", entries):
        if not isinstance(raw_entry, allowed):
            raise ValueError(f"{owner}: only input dynamic entries are legal")
    ids = [entry.id for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{owner}: duplicate dynamic entry ids")
    for entry in entries:
        _validate_structural_id(entry.id, f"{owner} entry id")


@dataclass(frozen=True)
class DynamicComboOption:
    key: str
    inputs: tuple[DynamicEntry, ...] = ()

    def __post_init__(self) -> None:
        if not OPTION_KEY_PATTERN.fullmatch(self.key):
            raise ValueError(f"dynamic combo option key must match [!-~]+( [!-~]+)*: {self.key!r}")
        _validate_dynamic_entries(self.inputs, f"dynamic combo option '{self.key}'")


@dataclass(frozen=True)
class DynamicComboSpec:
    id: str
    options: tuple[DynamicComboOption, ...]
    default: str | None = None
    required: bool = True
    doc: str = ""
    display_name: str = ""

    def __post_init__(self) -> None:
        _validate_structural_id(self.id, "dynamic combo id")
        keys = [option.key for option in self.options]
        if len(set(keys)) != len(keys):
            raise ValueError(f"dynamic combo {self.id}: duplicate option keys")
        for index, left in enumerate(keys):
            for right in keys[index + 1 :]:
                # The frontend persists `${construct}.[${key}]` paths and
                # matches branches with startsWith. If one sibling starts
                # with another plus "]", their composed prefixes alias.
                if right.startswith(left + "]") or left.startswith(right + "]"):
                    raise ValueError(
                        f"dynamic combo {self.id}: option keys {left!r} and {right!r} "
                        "alias frontend branch paths"
                    )
        if self.default is not None and self.default not in keys:
            raise ValueError(f"dynamic combo {self.id}: default must name a declared option key")

    def option(self, key: str) -> DynamicComboOption | None:
        for option in self.options:
            if option.key == key:
                return option
        return None


@dataclass(frozen=True)
class SlotVariant:
    """One per-type specialization of a dynamic slot (DESIGN, hazard H10).

    ``type`` is the slot socket's type while this variant is active -
    recursively concrete (concrete or list-of-concrete), because a variant
    IS a type choice; "any of several" is spelled as several variants.
    ``inputs`` are the variant's dependent inputs under construct-LOCAL
    names; elaboration materializes them as ``<slot id>.<local>`` members
    of the effective interface. Two variants of one slot may reuse a local
    name (they are mutually exclusive by construction) - the same
    conceptual knob typed differently per variant."""

    key: str
    type: TypeExpr
    inputs: tuple[DynamicEntry, ...] = ()
    doc: str = ""

    def __post_init__(self) -> None:
        if not VARIANT_KEY_PATTERN.fullmatch(self.key):
            raise ValueError(f"slot variant key must match [A-Za-z0-9_-]+: {self.key!r}")
        if self.type.runtime_type_id() is None:
            raise ValueError(
                f"slot variant '{self.key}': type must be recursively concrete "
                "(a variant is a type choice; use several variants for several types)"
            )
        if len({entry.id for entry in self.inputs}) != len(self.inputs):
            raise ValueError(f"slot variant '{self.key}': duplicate dependent input ids")
        for entry in self.inputs:
            if MEMBER_SEP in entry.id:
                raise ValueError(
                    f"slot variant '{self.key}': dependent input id may not contain "
                    f"'{MEMBER_SEP}': {entry.id}"
                )
        _validate_dynamic_entries(self.inputs, f"slot variant '{self.key}'")


@dataclass(frozen=True)
class DynamicSlotSpec:
    """A type-specializing input slot (DESIGN, hazard H10).

    The schema declares the variants; the DOCUMENT stores which one is
    active (GraphNode.slot_variants: slot id -> variant key - the frontend
    materializes the choice on connect). Elaboration turns the active
    variant into ordinary InputSpecs on the effective schema - the slot
    itself under the slot id with the variant's type, dependents as
    ``<slot id>.<local>`` - so nothing downstream branches on "is this a
    slot". The choice is recorded on the effective schema (slot_choices)
    and therefore joins the schema signature: switching variants is a new
    computation even when two variants share an identical interface shape.

    ``required=False`` lets a document omit the choice entirely (nothing
    materializes); a required slot with no stored choice refuses at
    elaboration - loudly, never a guessed default."""

    id: str
    variants: tuple[SlotVariant, ...] | None = None
    required: bool | None = None
    doc: str = ""
    display_name: str = ""
    inputs: tuple[DynamicEntry, ...] = ()
    slot_type: TypeExpr | None = None
    force_input: bool = False
    type_template_id: str = ""
    """Node-level type variable bound to the selected closed-slot variant."""

    def __post_init__(self) -> None:
        _validate_structural_id(self.id, "dynamic slot id")
        if type(self.type_template_id) is not str:
            raise ValueError(f"slot {self.id}: type_template_id must be a string")
        has_variants = self.variants is not None
        has_slot_type = self.slot_type is not None
        if has_variants == has_slot_type:
            raise ValueError(f"slot {self.id}: exactly one of variants or slot_type is required")
        object.__setattr__(
            self,
            "required",
            has_variants if self.required is None else self.required,
        )
        _validate_dynamic_entries(self.inputs, f"slot '{self.id}' shared inputs")
        if has_variants:
            assert self.variants is not None
            if not self.variants:
                raise ValueError(f"slot {self.id}: at least one variant is required")
            keys = [variant.key for variant in self.variants]
            if len(set(keys)) != len(keys):
                raise ValueError(f"slot {self.id}: duplicate variant keys")
            if self.force_input:
                raise ValueError(f"slot {self.id}: force_input is only legal on open slots")
            if self.type_template_id and not self.required:
                raise ValueError(f"slot {self.id}: type_template_id requires a required slot")
        else:
            assert self.slot_type is not None
            if self.type_template_id:
                raise ValueError(f"slot {self.id}: type_template_id is only legal on closed slots")
            if self.required:
                raise ValueError(f"slot {self.id}: open slots must be optional")
            if self.slot_type.kind == "variable":
                raise ValueError(f"slot {self.id}: slot_type must not be a variable")

    def variant(self, key: str) -> SlotVariant | None:
        for candidate in self.variants or ():
            if candidate.key == key:
                return candidate
        return None

    def member_id(self, local: str) -> str:
        return f"{self.id}{MEMBER_SEP}{local}"

    def member_local(self, input_id: str) -> str | None:
        """The local dependent name if input_id belongs to this slot, else None."""
        prefix = self.id + MEMBER_SEP
        if input_id.startswith(prefix) and len(input_id) > len(prefix):
            return input_id[len(prefix) :]
        return None


DynamicEntry = InputSpec | InputFamilySpec | DynamicComboSpec | DynamicSlotSpec


def _input_family_option_sources(widget: Widget | None) -> tuple[InputFamilyOptionSource, ...]:
    if isinstance(widget, ComboWidget):
        return (widget.option_source,) if widget.option_source is not None else ()
    if isinstance(widget, WidgetRepresentations):
        return tuple(
            source
            for representation in widget.representations
            for source in _input_family_option_sources(representation.widget)
        )
    return ()


def _dynamic_input_specs(entries: Sequence[DynamicEntry]) -> tuple[InputSpec, ...]:
    found: list[InputSpec] = []
    for entry in entries:
        if isinstance(entry, InputSpec):
            found.append(entry)
        elif isinstance(entry, InputFamilySpec):
            found.extend(_dynamic_input_specs(entry.template))
        elif isinstance(entry, DynamicComboSpec):
            for option in entry.options:
                found.extend(_dynamic_input_specs(option.inputs))
        else:
            found.extend(_dynamic_input_specs(entry.inputs))
            for variant in entry.variants or ():
                found.extend(_dynamic_input_specs(variant.inputs))
    return tuple(found)


def _dynamic_slots(
    entries: tuple[DynamicEntry, ...], depth: int = 0
) -> Iterable[tuple[DynamicSlotSpec, int]]:
    for entry in entries:
        if isinstance(entry, InputFamilySpec):
            yield from _dynamic_slots(entry.template, depth + 1)
        elif isinstance(entry, DynamicComboSpec):
            for option in entry.options:
                yield from _dynamic_slots(option.inputs, depth + 1)
        elif isinstance(entry, DynamicSlotSpec):
            yield entry, depth
            yield from _dynamic_slots(entry.inputs, depth + 1)
            for variant in entry.variants or ():
                yield from _dynamic_slots(variant.inputs, depth + 1)


def _type_variables(expr: TypeExpr) -> Iterable[TypeExpr]:
    if expr.kind == "variable":
        yield expr
    elif expr.kind in ("list", "asset", "stream"):
        assert expr.element is not None
        yield from _type_variables(expr.element)


@dataclass(frozen=True)
class SlotValue:
    """What execute() receives for a dynamic slot: the connected value, the
    active variant key (explicit dispatch - node code never sniffs runtime
    types), and the variant's dependent inputs under their LOCAL names."""

    variant: str
    value: object
    options: Mapping[str, object] = field(default_factory=dict[str, object])


@dataclass(frozen=True)
class OutputInterface:
    """The elaborated output membership a node execution must produce.

    Passed to execute() as the reserved ``output_spec`` parameter - only when
    the node declares output families, so static nodes never see it. Narrow
    and immutable by design: node authors read which members exist and return
    ``{family_id: {suffix: value}}``; they never construct interfaces.
    """

    members: tuple[tuple[str, tuple[str, ...]], ...]
    """Ordered (family_id, (suffix, ...)) pairs - the document's membership."""
    outputs: tuple[OutputSpec, ...] = ()
    """Ordered effective descriptors, including stable IDs, names and selected types."""

    def family(self, family_id: str) -> tuple[str, ...]:
        """The ordered member suffixes of one output family."""
        for fid, suffixes in self.members:
            if fid == family_id:
                return suffixes
        return ()


@dataclass(frozen=True)
class SelectorSpec:
    input: str
    branches: Mapping[str, str]

    def __post_init__(self) -> None:
        def nonempty_string(value: object) -> bool:
            return isinstance(value, str) and bool(value)

        branches = dict(cast("Mapping[object, object]", self.branches))
        if set(branches) != {"false", "true"}:
            raise ValueError("selector branches must be keyed by exactly 'false' and 'true'")
        if not nonempty_string(self.input) or not all(
            nonempty_string(value) for value in branches.values()
        ):
            raise ValueError("selector input and branch ids must be non-empty strings")
        if branches["false"] == branches["true"]:
            raise ValueError("selector branch input ids must be distinct")
        object.__setattr__(self, "branches", MappingProxyType(cast("dict[str, str]", branches)))


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

    ``applies`` scopes the mirror to a subset of the node's combo values:
    each key names a declared combo of the schema and maps to the option
    keys the mirror covers. A client renders an estimate only when every
    named combo's stored value is in its covered set, and renders none
    otherwise - a mirror has no abstain channel of its own, so the
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
    combos: tuple[DynamicComboSpec, ...],
) -> None:
    combos_by_id = {combo.id: combo for combo in combos}
    for applies_id, covered in applies.items():
        combo_spec = combos_by_id.get(applies_id)
        if combo_spec is None:
            raise ValueError(
                f"{node_type}: {subject} applies key {applies_id!r} "
                "does not name a declared dynamic combo"
            )
        if not combo_spec.required:
            raise ValueError(
                f"{node_type}: {subject} applies key {applies_id!r} must name a required combo"
            )
        declared_keys = {option.key for option in combo_spec.options}
        undeclared = sorted(set(covered) - declared_keys)
        if undeclared:
            raise ValueError(
                f"{node_type}: {subject} applies[{applies_id!r}] covers "
                f"undeclared option(s): {', '.join(undeclared)}"
            )


JsonScalar = str | int | float | bool | None


def _json_scalar_key(value: JsonScalar) -> tuple[str, object]:
    if type(value) in (int, float):
        return ("number", value)
    return (type(value).__name__, value)


@dataclass(frozen=True)
class ConditionalWidgetCondition:
    input: str
    values: tuple[JsonScalar, ...]

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.input), str) or not self.input:
            raise ValueError("conditional widget input must be a non-empty string")
        if not isinstance(cast("object", self.values), tuple):
            raise ValueError("conditional widget values must be an immutable tuple")
        if not self.values:
            raise ValueError("conditional widget values must be non-empty")
        for value in self.values:
            if not isinstance(value, (str, int, float, bool)) and value is not None:
                raise ValueError("conditional widget values must be JSON scalars")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("conditional widget values must be finite")
            if type(value) is float and value.is_integer() and abs(value) > 2**53 - 1:
                raise ValueError("conditional widget integer values must be JSON-safe")
            if type(value) is int and abs(value) > 2**53 - 1:
                raise ValueError("conditional widget integer values must be JSON-safe")
        if len({_json_scalar_key(value) for value in self.values}) != len(self.values):
            raise ValueError("conditional widget values must be unique")


@dataclass(frozen=True)
class ConditionalWidgetGroup(ConditionalWidgetCondition):
    members: tuple[str, ...]
    requires: tuple[ConditionalWidgetCondition, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(cast("object", self.members), tuple):
            raise ValueError("conditional widget group members must be an immutable tuple")
        if not self.members or any(
            not isinstance(cast("object", member), str) or not member for member in self.members
        ):
            raise ValueError("conditional widget group members must be non-empty strings")
        if len(set(self.members)) != len(self.members):
            raise ValueError("conditional widget group members must be unique")
        if not isinstance(cast("object", self.requires), tuple):
            raise ValueError("conditional widget group requires must be an immutable tuple")
        if any(
            not isinstance(cast("object", item), ConditionalWidgetCondition)
            for item in self.requires
        ):
            raise ValueError("conditional widget group requires must contain conditions")


@dataclass(frozen=True)
class NodeSchema:
    node_type: str
    version: int = 1
    display_name: str = ""
    category: str = ""
    description: str = ""
    editor_role: str | None = None
    """Frontend capability role. Presentation metadata only."""
    inputs: tuple[InputSpec, ...] = ()
    outputs: tuple[OutputSpec, ...] = ()
    input_families: tuple[InputFamilySpec, ...] = ()
    output_families: tuple[OutputFamilySpec, ...] = ()
    combos: tuple[DynamicComboSpec, ...] = ()
    slots: tuple[DynamicSlotSpec, ...] = ()
    slot_choices: tuple[tuple[str, str], ...] = ()
    """The active variant key per slot, set ONLY on elaborated schemas
    (base schemas declare slots; elaboration replaces them with concrete
    inputs plus this record). Part of the wire form and therefore of the
    schema signature - the variant choice is computation identity even
    when two variants share an interface shape (hazard H4)."""
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
    search terms never resolve anything (that is ``aliases``), never
    join the schema signature, and collisions across schemas are
    meaningless - many nodes may legitimately answer to "blur"."""
    aliases: tuple[str, ...] = ()
    """Alternate type names accepted by submission-format adapters when
    resolving this node (e.g. a legacy ComfyUI v1 class_type, or a former
    node_type after a rename). Resolution metadata only: aliases never
    appear in stored documents (documents always carry node_type), never
    join the schema signature, and collisions across schemas are legal at
    registration - an adapter must refuse an ambiguous alias at use, never
    pick silently. Distinct from ``search_terms``: an alias must stay
    unambiguous at use, a search term is free-form discovery data."""
    output_node: bool = False
    """This node acts as a sink/output of a workflow (v1 OUTPUT_NODE, a
    save/preview node). A hint for submission formats that carry no explicit
    targets (they target every output_node in the document) and for
    frontend default-target affordances. Never scheduling, never validity,
    never part of the schema signature."""
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
    """This node's execution may return a runtime graph expansion payload
    that Dinkster cannot execute. Set only where that is true: compat
    translation marks every schema it serves from translated ComfyUI
    sources, because a v1 ``{"expand": ...}`` result or a V3
    ``NodeOutput.expand`` can be produced from runtime data, while native
    execution has no expansion concept and ordinary schemas stay unflagged.
    Importers use the flag to refuse unsupported structures (such as a
    Generic Loop region containing such a node) before submission; execution
    still refuses any payload that slips through, loud. Capability metadata
    only: never validity and never part of the schema signature."""
    mirror: MirrorSpec | None = None
    """Optional frontend-renderable mirror of this node's transform (see
    MirrorSpec). Presentation metadata only: the engine never reads it,
    mirrored estimates never become outputs, and ``execute()`` stays the
    sole authoritative implementation. Never part of the schema signature -
    declaring, changing, or removing a mirror must not invalidate caches."""
    selector: SelectorSpec | None = None
    widget_groups: tuple[ConditionalWidgetGroup, ...] = ()
    """Conditional visibility for top-level widgets. Presentation metadata only."""
    output_descriptors: OutputDescriptorsSpec | None = None
    chunk_safe: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    """Media ranges; ordinary inputs promise independent, cardinality-preserving execution."""
    chunk_safe_applies: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if not self.node_type:
            raise ValueError("node_type is required")
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
            if not set(inputs) <= {spec.id for spec in (*self.inputs, *self.slots)}:
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
                self.node_type, "chunk_safe", self.chunk_safe_applies, self.combos
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
                self.combos,
            )
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError(f"{self.node_type}: duplicate aliases")
        for alias in self.aliases:
            if not alias:
                raise ValueError(f"{self.node_type}: aliases must be non-empty strings")
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
        if not isinstance(cast("object", self.widget_groups), tuple):
            raise ValueError(f"{self.node_type}: widget_groups must be an immutable tuple")
        widget_ids = {spec.id for spec in self.inputs if spec.widget is not None}
        for index, group in enumerate(self.widget_groups):
            if not isinstance(cast("object", group), ConditionalWidgetGroup):
                raise ValueError(
                    f"{self.node_type}: widget_groups[{index}] must be a ConditionalWidgetGroup"
                )
            conditions = (group, *group.requires)
            drivers = tuple(condition.input for condition in conditions)
            if len(set(drivers)) != len(drivers):
                raise ValueError(f"{self.node_type}: widget group condition inputs must be unique")
            if any(driver not in widget_ids for driver in drivers):
                raise ValueError(
                    f"{self.node_type}: widget group conditions must name top-level widget inputs"
                )
            if any(member not in widget_ids for member in group.members):
                raise ValueError(
                    f"{self.node_type}: widget group members must name top-level widget inputs"
                )
            if set(drivers) & set(group.members):
                raise ValueError(
                    f"{self.node_type}: widget group conditions cannot control themselves"
                )
        if self.selector is not None:
            selector = self.selector
            inputs_by_id = {spec.id: spec for spec in self.inputs}
            if (
                len(self.outputs) != 1
                or self.output_families
                or self.output_descriptors is not None
            ):
                raise ValueError(f"{self.node_type}: selector schemas require exactly one output")
            selector_input = inputs_by_id.get(selector.input)
            if selector_input is None or selector_input.type != TypeExpr.concrete("core.boolean"):
                raise ValueError(
                    f"{self.node_type}: selector input must name a stored core.boolean input"
                )
            branch_ids = (selector.branches["false"], selector.branches["true"])
            if selector.input in branch_ids:
                raise ValueError(
                    f"{self.node_type}: selector input must be distinct from branch inputs"
                )
            for branch_id in branch_ids:
                branch = inputs_by_id.get(branch_id)
                if branch is None:
                    raise ValueError(
                        f"{self.node_type}: selector branch {branch_id!r} is not an input"
                    )
                if branch.type != self.outputs[0].type:
                    raise ValueError(
                        f"{self.node_type}: selector branches must share the output type"
                    )
        in_family_ids = [fam.id for fam in self.input_families]
        out_family_ids = [fam.id for fam in self.output_families]
        combo_ids = [combo.id for combo in self.combos]
        slot_ids = [slot.id for slot in self.slots]
        reserved = RESERVED_INPUT_IDS & (
            set(input_ids) | set(in_family_ids) | set(combo_ids) | set(slot_ids)
        )
        if reserved:
            raise ValueError(
                f"{self.node_type}: reserved input id(s): {', '.join(sorted(reserved))}"
            )
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError(f"{self.node_type}: duplicate slot ids")
        binding_ids: set[str] = set()
        output_variables = tuple(
            variable
            for output in (*self.outputs, *self.output_families)
            for variable in _type_variables(output.type)
        )
        for slot, depth in _dynamic_slots((*self.input_families, *self.combos, *self.slots)):
            template_id = slot.type_template_id
            if not template_id:
                continue
            if depth:
                raise ValueError(
                    f"{self.node_type}: slot '{slot.id}' type_template_id is only legal "
                    "on top-level slots"
                )
            if template_id in binding_ids:
                raise ValueError(
                    f"{self.node_type}: type variable '{template_id}' is bound by multiple slots"
                )
            binding_ids.add(template_id)
            matching_outputs = tuple(
                variable for variable in output_variables if variable.template_id == template_id
            )
            if not matching_outputs:
                raise ValueError(
                    f"{self.node_type}: slot '{slot.id}' binds type variable "
                    f"'{template_id}', but no output uses it"
                )
            for variant in slot.variants or ():
                type_id = variant.type.runtime_type_id()
                assert type_id is not None
                if any(not variable.accepts_concrete(type_id) for variable in matching_outputs):
                    raise ValueError(
                        f"{self.node_type}: slot '{slot.id}' variant '{variant.key}' type "
                        f"{type_id!r} is outside output variable '{template_id}' allowlist"
                    )
        dynamic_ids = in_family_ids + combo_ids + slot_ids
        if len(set(dynamic_ids)) != len(dynamic_ids):
            raise ValueError(f"{self.node_type}: duplicate or colliding dynamic construct ids")
        if set(dynamic_ids) & set(input_ids):
            raise ValueError(f"{self.node_type}: dynamic construct id collides with an input id")
        for construct in (*self.input_families, *self.combos, *self.slots):
            for input_id in input_ids:
                if input_id.startswith(construct.id + MEMBER_SEP):
                    kind = (
                        "family"
                        if isinstance(construct, InputFamilySpec)
                        else "slot"
                        if isinstance(construct, DynamicSlotSpec)
                        else "dynamic combo"
                    )
                    raise ValueError(
                        f"{self.node_type}: input '{input_id}' shadows {kind} '{construct.id}'"
                    )
        choice_ids = [slot_id for slot_id, _ in self.slot_choices]
        if len(set(choice_ids)) != len(choice_ids):
            raise ValueError(f"{self.node_type}: duplicate slot choice ids")
        if self.slot_choices and (self.input_families or self.combos or self.slots):
            raise ValueError(
                f"{self.node_type}: slot_choices belong to elaborated schemas; "
                "base schemas declare dynamic constructs"
            )
        if len(set(in_family_ids)) != len(in_family_ids):
            raise ValueError(f"{self.node_type}: duplicate input family ids")
        if len(set(out_family_ids)) != len(out_family_ids):
            raise ValueError(f"{self.node_type}: duplicate output family ids")
        for spec in self.inputs:
            for source in _input_family_option_sources(spec.widget):
                if source.input_family not in in_family_ids:
                    raise ValueError(
                        f"{self.node_type}: combo option source names unknown input family "
                        f"'{source.input_family}'"
                    )
        dynamic_specs = _dynamic_input_specs((*self.input_families, *self.combos, *self.slots))
        if any(_input_family_option_sources(spec.widget) for spec in dynamic_specs):
            raise ValueError(
                f"{self.node_type}: input-family combo option sources are only legal on "
                "top-level inputs"
            )
        inputs_by_id = {spec.id: spec for spec in self.inputs}
        if self.output_descriptors is not None:
            descriptor_input = inputs_by_id.get(self.output_descriptors.input)
            if (
                descriptor_input is None
                or not descriptor_input.required
                or descriptor_input.type != TypeExpr.concrete("core.string")
            ):
                raise ValueError(
                    f"{self.node_type}: output descriptors require a required top-level "
                    "core.string input"
                )
            probe = self.output_descriptors.probe
            if probe is not None:
                asset_input = inputs_by_id.get(probe.input)
                if (
                    asset_input is None
                    or not asset_input.required
                    or asset_input.type != TypeExpr.concrete("dinkster.asset")
                ):
                    raise ValueError(
                        "output probe requires a required top-level dinkster.asset input"
                    )
        for output in self.outputs:
            known_value = output.known_value
            if known_value is not None:
                known_input = inputs_by_id.get(known_value.input)
                if known_input is None:
                    raise ValueError(
                        f"{self.node_type}: output {output.id!r} known value input "
                        f"{known_value.input!r} is not a declared top-level input"
                    )
                primitive_types = {
                    TypeExpr.concrete("core.int"),
                    TypeExpr.concrete("core.float"),
                    TypeExpr.concrete("core.string"),
                    TypeExpr.concrete("core.boolean"),
                }
                if known_input.type != output.type or output.type not in primitive_types:
                    raise ValueError(
                        f"{self.node_type}: output {output.id!r} known value must reference "
                        "an input with the same concrete primitive type"
                    )
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
                    self.combos,
                )
        for out_fam in self.output_families:
            if out_fam.count is None:
                continue
            count_input = inputs_by_id.get(out_fam.count.input)
            if (
                count_input is None
                or not count_input.required
                or count_input.type != TypeExpr.concrete("core.int")
            ):
                raise ValueError(
                    f"{self.node_type}: output family count must reference a required "
                    "top-level core.int input"
                )
        if set(in_family_ids) & set(input_ids):
            raise ValueError(f"{self.node_type}: input family id collides with input id")
        if set(out_family_ids) & set(output_ids):
            raise ValueError(f"{self.node_type}: output family id collides with output id")
        # Static ids must not be parseable as members of a declared family -
        # member ownership has to be unambiguous.
        for fam in self.input_families:
            for input_id in input_ids:
                if fam.member_suffix(input_id) is not None:
                    raise ValueError(
                        f"{self.node_type}: input '{input_id}' shadows family '{fam.id}'"
                    )
        for out_fam in self.output_families:
            for output_id in output_ids:
                if out_fam.member_suffix(output_id) is not None:
                    raise ValueError(
                        f"{self.node_type}: output '{output_id}' shadows family '{out_fam.id}'"
                    )

    @property
    def is_static(self) -> bool:
        return (
            not self.input_families
            and not self.output_families
            and self.output_descriptors is None
            and not self.combos
            and not self.slots
        )

    def slot(self, slot_id: str) -> DynamicSlotSpec | None:
        for spec in self.slots:
            if spec.id == slot_id:
                return spec
        return None

    def slot_of_input(self, input_id: str) -> tuple[DynamicSlotSpec, str | None] | None:
        """The (slot, local dependent name) owning input_id, if any: the
        slot input itself yields (slot, None), a dependent yields
        (slot, local). Mirrors family_of_input for the slot construct."""
        for spec in self.slots:
            if input_id == spec.id:
                return (spec, None)
            local = spec.member_local(input_id)
            if local is not None:
                return (spec, local)
        return None

    def input(self, input_id: str) -> InputSpec | None:
        for spec in self.inputs:
            if spec.id == input_id:
                return spec
        return None

    def input_family(self, family_id: str) -> InputFamilySpec | None:
        for fam in self.input_families:
            if fam.id == family_id:
                return fam
        return None

    def family_of_input(self, input_id: str) -> InputFamilySpec | None:
        """The input family that owns input_id as a member, if any."""
        for fam in self.input_families:
            if fam.member_suffix(input_id) is not None:
                return fam
        return None

    def output_family(self, family_id: str) -> OutputFamilySpec | None:
        for fam in self.output_families:
            if fam.id == family_id:
                return fam
        return None

    def output(self, output_id: str) -> OutputSpec | None:
        for spec in self.outputs:
            if spec.id == output_id:
                return spec
        return None


def _remote_choice_ids(schema: NodeSchema) -> tuple[str, ...]:
    """Collect remote COMBO ids from every recursive schema input position."""
    found: list[str] = []

    def visit_widget(widget: Widget | None) -> None:
        if isinstance(widget, ComboWidget) and widget.remote_route:
            found.append(_remote_choice_id(widget.remote_route))
        elif isinstance(widget, WidgetRepresentations):
            for representation in widget.representations:
                visit_widget(representation.widget)

    def visit_entries(entries: Sequence[DynamicEntry]) -> None:
        for entry in entries:
            if isinstance(entry, InputSpec):
                visit_widget(entry.widget)
            elif isinstance(entry, InputFamilySpec):
                visit_entries(entry.template)
            elif isinstance(entry, DynamicComboSpec):
                for option in entry.options:
                    visit_entries(option.inputs)
            else:
                visit_entries(entry.inputs)
                for variant in entry.variants or ():
                    visit_entries(variant.inputs)

    visit_entries(schema.inputs)
    for family in schema.input_families:
        visit_entries(family.template)
    for combo in schema.combos:
        for option in combo.options:
            visit_entries(option.inputs)
    for slot in schema.slots:
        visit_entries(slot.inputs)
        for variant in slot.variants or ():
            visit_entries(variant.inputs)
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
