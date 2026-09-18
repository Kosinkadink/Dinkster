"""Declarative node-replacement rules: pack-declared migration data.

Field-for-field mirror of Dinkster-Frontend's ``replace/model.ts`` (the
frontend's replacement engine is the executor of these rules, so its model is
the reference shape - we mirror, never dialect). A rule is DATA: a closed
predicate and transform vocabulary, never code, so rules ship inside schemas,
are testable against golden corpora, and are safe on a wire.

Shape: one rule migrates one source node type (``ReplacementRule.from_type``,
``"from"`` on the wire). Its cases are evaluated top-down against the source
node's document state; the FIRST matching case wins and the last case must be
unconditional (the required fallback). Each case names its own target ``to``,
so one predecessor can fan out to different successors under guards - a rule
carried on a successor's schema is deliberately NOT constrained to target
that schema.

Ordinary mappings reference schema interface ids, including paths materialized
by literal target ``slotVariants`` selections. The additive ``inputFamilies`` and
``outputFamilies`` vocabularies map top-level families without spelling member
paths; input families may address helpers, and the planner preserves source
suffixes and authored order. A same-type ``migration`` rule declares source input
or dynamic-choice paths retired from the current schema so stored nodes can
migrate once into a new shape. Cross-schema reference
checks belong to tooling that can see both schemas (``dinkster doctor``); this
module enforces the closed vocabulary itself.

The vocabulary grows additively (merge/split transforms are later additions,
never reinterpretations); unknown kinds reject at construction and at wire
decode, exactly like the frontend's ``isReplacementRule`` validator.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from .model import NodeSchema

PredicateKind = Literal[
    "always", "inputConnected", "valuePresent", "valueEquals", "not", "all", "any"
]
TransformKind = Literal["enumRename", "scale"]
MappingSourceKind = Literal["copy", "value", "link", "constant"]
InputFamilyMappingKind = Literal["copy", "members"]
OutputFamilyMappingKind = Literal["copy", "members"]

_INPUT_PREDICATES = frozenset({"inputConnected", "valuePresent", "valueEquals"})
_COMPOSITE_PREDICATES = frozenset({"not", "all", "any"})
_STRUCTURAL_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _require_json(value: object, where: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError(f"{where} must be JSON-serializable data") from None


@dataclass(frozen=True)
class ReplacementPredicate:
    """Closed guard vocabulary, evaluated against the source node's document
    state (connections + stored values). ``of`` holds one child for ``not``
    and any number for ``all``/``any`` (encoded to the wire as a single
    object or an array respectively, matching the TS union)."""

    kind: PredicateKind
    input: str = ""
    value: object = None  # valueEquals only; arbitrary JSON (null included)
    of: tuple[ReplacementPredicate, ...] = ()

    def __post_init__(self) -> None:
        if self.kind in _INPUT_PREDICATES:
            if not self.input:
                raise ValueError(f"{self.kind} predicate requires an input id")
            if self.of:
                raise ValueError(f"{self.kind} predicate carries no children")
        elif self.kind in _COMPOSITE_PREDICATES:
            if self.input:
                raise ValueError(f"{self.kind} predicate carries no input id")
            if self.kind == "not" and len(self.of) != 1:
                raise ValueError("not predicate requires exactly one child")
        elif self.kind == "always":
            if self.input or self.of:
                raise ValueError("always predicate carries no fields")
        else:
            raise ValueError(f"unknown predicate kind: {self.kind!r}")
        if self.kind == "valueEquals":
            _require_json(self.value, "valueEquals.value")
        elif self.value is not None:
            raise ValueError(f"{self.kind} predicate carries no value")

    @classmethod
    def always(cls) -> ReplacementPredicate:
        return cls(kind="always")

    @classmethod
    def input_connected(cls, input_id: str) -> ReplacementPredicate:
        return cls(kind="inputConnected", input=input_id)

    @classmethod
    def value_present(cls, input_id: str) -> ReplacementPredicate:
        return cls(kind="valuePresent", input=input_id)

    @classmethod
    def value_equals(cls, input_id: str, value: object) -> ReplacementPredicate:
        return cls(kind="valueEquals", input=input_id, value=value)

    @classmethod
    def not_(cls, child: ReplacementPredicate) -> ReplacementPredicate:
        return cls(kind="not", of=(child,))

    @classmethod
    def all_of(cls, *children: ReplacementPredicate) -> ReplacementPredicate:
        return cls(kind="all", of=children)

    @classmethod
    def any_of(cls, *children: ReplacementPredicate) -> ReplacementPredicate:
        return cls(kind="any", of=children)


@dataclass(frozen=True)
class ValueTransform:
    """Closed value-transform vocabulary. ``enumRename``: a source value
    missing from the map is an ERROR (the frontend's semantics - never a
    silent pass-through). ``scale``: ``value * factor + offset``; a
    non-number source is an ERROR."""

    kind: TransformKind
    map: tuple[tuple[str, str], ...] = ()
    factor: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.kind == "enumRename":
            if self.factor != 1.0 or self.offset != 0.0:
                raise ValueError("enumRename carries no factor/offset")
            for source, target in self.map:
                # Author data may arrive untyped despite the annotations.
                if not isinstance(source, str) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                    target, str
                ):
                    raise ValueError("enumRename map entries must be strings")
        elif self.kind == "scale":
            if self.map:
                raise ValueError("scale carries no map")
            try:
                finite = (
                    type(self.factor) in (int, float)
                    and type(self.offset) in (int, float)
                    and math.isfinite(self.factor)
                    and math.isfinite(self.offset)
                )
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("scale factor and offset must be finite numbers")
        else:
            raise ValueError(f"unknown transform kind: {self.kind!r}")

    @classmethod
    def enum_rename(cls, mapping: Mapping[str, str]) -> ValueTransform:
        return cls(kind="enumRename", map=tuple(mapping.items()))

    @classmethod
    def scale(cls, factor: float, offset: float = 0.0) -> ValueTransform:
        return cls(kind="scale", factor=factor, offset=offset)


@dataclass(frozen=True)
class MappingSource:
    """What feeds one TARGET input. ``copy`` moves the incoming connection
    AND copies the stored value (the workhorse); ``value`` copies only the
    stored value (optionally transformed, never moves links); ``link`` moves
    only the connection; ``constant`` writes a literal."""

    kind: MappingSourceKind
    input: str = ""
    transform: ValueTransform | None = None
    value: object = None  # constant only; arbitrary JSON (null included)

    def __post_init__(self) -> None:
        if self.kind in ("copy", "value", "link"):
            if not self.input:
                raise ValueError(f"{self.kind} mapping requires a source input id")
            if self.value is not None:
                raise ValueError(f"{self.kind} mapping carries no constant value")
            if self.transform is not None and self.kind != "value":
                raise ValueError(f"{self.kind} mapping carries no transform")
        elif self.kind == "constant":
            if self.input or self.transform is not None:
                raise ValueError("constant mapping carries only a value")
            _require_json(self.value, "constant.value")
        else:
            raise ValueError(f"unknown mapping source kind: {self.kind!r}")

    @classmethod
    def copy(cls, input_id: str) -> MappingSource:
        return cls(kind="copy", input=input_id)

    @classmethod
    def from_value(cls, input_id: str, transform: ValueTransform | None = None) -> MappingSource:
        return cls(kind="value", input=input_id, transform=transform)

    @classmethod
    def link(cls, input_id: str) -> MappingSource:
        return cls(kind="link", input=input_id)

    @classmethod
    def constant(cls, value: object) -> MappingSource:
        return cls(kind="constant", value=value)


@dataclass(frozen=True)
class InputFamilyMember:
    """One explicit member created in a dynamic target input family."""

    suffix: str
    inputs: tuple[tuple[str, MappingSource], ...]

    def __post_init__(self) -> None:
        if _STRUCTURAL_ID.fullmatch(self.suffix) is None:
            raise ValueError(f"invalid input family member suffix: {self.suffix!r}")
        input_ids = [input_id for input_id, _ in self.inputs]
        if not input_ids:
            raise ValueError(f"input family member {self.suffix!r} requires input mappings")
        if len(set(input_ids)) != len(input_ids):
            raise ValueError(f"input family member {self.suffix!r} has duplicate input mappings")
        for input_id in input_ids:
            if _STRUCTURAL_ID.fullmatch(input_id) is None:
                raise ValueError(f"invalid input family template input id: {input_id!r}")

    @classmethod
    def build(cls, suffix: str, *, inputs: Mapping[str, MappingSource]) -> InputFamilyMember:
        return cls(suffix=suffix, inputs=tuple(inputs.items()))


@dataclass(frozen=True)
class InputFamilyMapping:
    """A top-level dynamic target-family mapping.

    ``copy`` preserves every source-family suffix and member order while
    mapping template-local inputs. ``members`` creates an explicit ordered
    target member list from ordinary static source inputs.
    """

    kind: InputFamilyMappingKind
    source_family: str = ""
    inputs: tuple[tuple[str, MappingSource], ...] = ()
    members: tuple[InputFamilyMember, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "copy":
            if _STRUCTURAL_ID.fullmatch(self.source_family) is None:
                raise ValueError("copy input family mapping requires a source family id")
            if not self.inputs:
                raise ValueError("copy input family mapping requires template input mappings")
            if self.members:
                raise ValueError("copy input family mapping carries no explicit members")
        elif self.kind == "members":
            if self.source_family or self.inputs:
                raise ValueError("members input family mapping carries only explicit members")
            if not self.members:
                raise ValueError("members input family mapping requires explicit members")
            suffixes = [member.suffix for member in self.members]
            if len(set(suffixes)) != len(suffixes):
                raise ValueError("members input family mapping has duplicate suffixes")
        else:
            raise ValueError(f"unknown input family mapping kind: {self.kind!r}")
        input_ids = [input_id for input_id, _ in self.inputs]
        if len(set(input_ids)) != len(input_ids):
            raise ValueError("copy input family mapping has duplicate input mappings")
        for input_id in input_ids:
            if _STRUCTURAL_ID.fullmatch(input_id) is None:
                raise ValueError(f"invalid input family template input id: {input_id!r}")

    @classmethod
    def copy(cls, source_family: str, *, inputs: Mapping[str, MappingSource]) -> InputFamilyMapping:
        return cls(kind="copy", source_family=source_family, inputs=tuple(inputs.items()))

    @classmethod
    def from_members(cls, *members: InputFamilyMember) -> InputFamilyMapping:
        return cls(kind="members", members=members)


@dataclass(frozen=True)
class OutputFamilyMember:
    """One explicit member created in a dynamic target output family."""

    suffix: str
    output: str

    def __post_init__(self) -> None:
        if (
            not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                self.suffix, str
            )
            or _STRUCTURAL_ID.fullmatch(self.suffix) is None
        ):
            raise ValueError(f"invalid output family member suffix: {self.suffix!r}")
        if (
            not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                self.output, str
            )
            or _STRUCTURAL_ID.fullmatch(self.output) is None
        ):
            raise ValueError(f"invalid source output id: {self.output!r}")


@dataclass(frozen=True)
class OutputFamilyMapping:
    """A top-level dynamic target-output-family mapping.

    ``copy`` preserves every source-family suffix and member order.
    ``members`` creates an explicit ordered target member list from ordinary
    static source outputs.
    """

    kind: OutputFamilyMappingKind
    source_family: str = ""
    members: tuple[OutputFamilyMember, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "copy":
            if _STRUCTURAL_ID.fullmatch(self.source_family) is None:
                raise ValueError("copy output family mapping requires a source family id")
            if self.members:
                raise ValueError("copy output family mapping carries no explicit members")
        elif self.kind == "members":
            if self.source_family:
                raise ValueError("members output family mapping carries only explicit members")
            if not self.members:
                raise ValueError("members output family mapping requires explicit members")
            suffixes = [member.suffix for member in self.members]
            if len(set(suffixes)) != len(suffixes):
                raise ValueError("members output family mapping has duplicate suffixes")
        else:
            raise ValueError(f"unknown output family mapping kind: {self.kind!r}")

    @classmethod
    def copy(cls, source_family: str) -> OutputFamilyMapping:
        return cls(kind="copy", source_family=source_family)

    @classmethod
    def from_members(cls, *members: OutputFamilyMember) -> OutputFamilyMapping:
        return cls(kind="members", members=members)


@dataclass(frozen=True)
class ReplacementNode:
    """One helper node created by a replacement case."""

    type: str
    values: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                self.type, str
            )
            or not self.type
        ):
            raise ValueError("replacement helper requires a node type")
        keys = [key for key, _ in self.values]
        if len(set(keys)) != len(keys):
            raise ValueError("replacement helper has duplicate value ids")
        for key, value in self.values:
            if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                key, str
            ):
                raise ValueError("replacement helper value ids must be strings")
            if any(_STRUCTURAL_ID.fullmatch(segment) is None for segment in key.split(".")):
                raise ValueError("replacement helper has an invalid value id")
            _require_json(value, f"replacement helper value {key!r}")

    @classmethod
    def build(cls, type: str, *, values: Mapping[str, object] | None = None) -> ReplacementNode:
        return cls(type=type, values=tuple((values or {}).items()))


@dataclass(frozen=True)
class ReplacementLink:
    """One internal connection from an output address to an input address."""

    from_address: str  # "from" on the wire (Python keyword)
    to: str


def _address_local_id(address: str, local_ids: frozenset[str], helpers_enabled: bool) -> str | None:
    if (
        not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            address, str
        )
        or not address
    ):
        raise ValueError(f"invalid replacement address: {address!r}")
    if not helpers_enabled:
        return None
    parts = address.split(":")
    if len(parts) == 1:
        return None
    if (
        len(parts) != 2
        or not parts[0]
        or not parts[1]
        or _STRUCTURAL_ID.fullmatch(parts[0]) is None
        or parts[0] not in local_ids
    ):
        raise ValueError(f"invalid replacement address: {address!r}")
    return parts[0]


@dataclass(frozen=True)
class ReplacementCase:
    """One guarded migration target. The primary ``to`` node keeps the source
    document id; optional ``nodes`` are helpers addressed as ``localId:port``.
    ``slot_variants`` selects dynamic target constructs by materialized path,
    ``inputs`` maps target input addresses to mapping sources,
    ``input_families`` maps top-level primary or helper families while
    ``output_families`` maps primary-node families, ``links`` wire target nodes
    internally, and ``outputs`` maps
    target output addresses to SOURCE output ids. Unmapped primary/helper
    inputs get declared defaults written explicitly by the frontend planner.
    A source output referenced by two entries is an error here (fan-in would
    duplicate links); a source output whose live links no entry consumes is
    the frontend's review-forcing warning, not ours to detect (it needs
    document state)."""

    to: str
    when: ReplacementPredicate | None = None
    nodes: tuple[tuple[str, ReplacementNode], ...] | None = None
    inputs: tuple[tuple[str, MappingSource], ...] = ()
    input_families: tuple[tuple[str, InputFamilyMapping], ...] = ()
    output_families: tuple[tuple[str, OutputFamilyMapping], ...] = ()
    links: tuple[ReplacementLink, ...] = ()
    outputs: tuple[tuple[str, str], ...] = ()
    slot_variants: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.to:
            raise ValueError("replacement case requires a target node type")
        local_id_list = [local_id for local_id, _ in self.nodes or ()]
        if len(set(local_id_list)) != len(local_id_list):
            raise ValueError(f"case -> {self.to}: duplicate helper local id")
        if any(_STRUCTURAL_ID.fullmatch(local_id) is None for local_id in local_id_list):
            raise ValueError(f"case -> {self.to}: invalid helper local id")
        local_ids = frozenset(local_id_list)
        helpers_enabled = self.nodes is not None
        slot_variant_addresses = [address for address, _ in self.slot_variants]
        if len(set(slot_variant_addresses)) != len(slot_variant_addresses):
            raise ValueError(f"case -> {self.to}: duplicate slot variant address")
        for address, choice in self.slot_variants:
            local_id = _address_local_id(address, local_ids, helpers_enabled)
            path = address.split(":", 1)[1] if local_id is not None else address
            if any(_STRUCTURAL_ID.fullmatch(segment) is None for segment in path.split(".")):
                raise ValueError(f"case -> {self.to}: invalid slot variant path")
            choice_value = cast(object, choice)
            if not isinstance(choice_value, str) or not choice_value:
                raise ValueError(f"case -> {self.to}: slot variant must be a non-empty string")
        target_inputs = [target for target, _ in self.inputs]
        if len(set(target_inputs)) != len(target_inputs):
            raise ValueError(f"case -> {self.to}: duplicate target input mapping")
        for target in target_inputs:
            _address_local_id(target, local_ids, helpers_enabled)
        target_families = [target for target, _ in self.input_families]
        if len(set(target_families)) != len(target_families):
            raise ValueError(f"case -> {self.to}: duplicate target input family mapping")
        for target in target_families:
            local_id = _address_local_id(target, local_ids, helpers_enabled)
            family_id = target.split(":", 1)[1] if local_id is not None else target
            if _STRUCTURAL_ID.fullmatch(family_id) is None:
                raise ValueError(f"case -> {self.to}: invalid target input family id")
        target_output_families = [target for target, _ in self.output_families]
        if len(set(target_output_families)) != len(target_output_families):
            raise ValueError(f"case -> {self.to}: duplicate target output family mapping")
        for target in target_output_families:
            if _STRUCTURAL_ID.fullmatch(target) is None:
                raise ValueError(f"case -> {self.to}: invalid target output family id")
        for link in self.links:
            _address_local_id(link.from_address, local_ids, helpers_enabled)
            _address_local_id(link.to, local_ids, helpers_enabled)
        fed = set(target_inputs)
        for link in self.links:
            if link.to in fed:
                raise ValueError(f"case -> {self.to}: target input {link.to!r} has two feeders")
            fed.add(link.to)
        target_outputs = [target for target, _ in self.outputs]
        if len(set(target_outputs)) != len(target_outputs):
            raise ValueError(f"case -> {self.to}: duplicate target output mapping")
        for target in target_outputs:
            _address_local_id(target, local_ids, helpers_enabled)
        source_outputs = [source for _, source in self.outputs]
        source_outputs.extend(
            member.output
            for _, mapping in self.output_families
            if mapping.kind == "members"
            for member in mapping.members
        )
        if len(set(source_outputs)) != len(source_outputs):
            raise ValueError(
                f"case -> {self.to}: a source output feeds two target outputs "
                "(fan-in would duplicate links)"
            )
        source_output_families = [
            mapping.source_family for _, mapping in self.output_families if mapping.kind == "copy"
        ]
        if len(set(source_output_families)) != len(source_output_families):
            raise ValueError(
                f"case -> {self.to}: a source output family feeds two target families "
                "(fan-in would duplicate links)"
            )
        edges: dict[str, set[str]] = {}
        for link in self.links:
            source = _address_local_id(link.from_address, local_ids, helpers_enabled) or ""
            target = _address_local_id(link.to, local_ids, helpers_enabled) or ""
            edges.setdefault(source, set()).add(target)
        visiting: set[str] = set()
        visited: set[str] = set()

        def cyclic(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(cyclic(target) for target in edges.get(node, ())):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        if any(cyclic(node) for node in ("", *local_ids)):
            raise ValueError(f"case -> {self.to}: internal links must be acyclic")

    @classmethod
    def build(
        cls,
        to: str,
        *,
        when: ReplacementPredicate | None = None,
        nodes: Mapping[str, ReplacementNode] | None = None,
        slot_variants: Mapping[str, str] | None = None,
        inputs: Mapping[str, MappingSource] | None = None,
        input_families: Mapping[str, InputFamilyMapping] | None = None,
        output_families: Mapping[str, OutputFamilyMapping] | None = None,
        links: Sequence[ReplacementLink] = (),
        outputs: Mapping[str, str] | None = None,
    ) -> ReplacementCase:
        """Ergonomic constructor: plain dicts in, canonical tuples stored."""
        return cls(
            to=to,
            when=when,
            nodes=None if nodes is None else tuple(nodes.items()),
            slot_variants=tuple((slot_variants or {}).items()),
            inputs=tuple((inputs or {}).items()),
            input_families=tuple((input_families or {}).items()),
            output_families=tuple((output_families or {}).items()),
            links=tuple(links),
            outputs=tuple((outputs or {}).items()),
        )

    @property
    def unconditional(self) -> bool:
        return self.when is None or self.when.kind == "always"


@dataclass(frozen=True)
class ReplacementMigration:
    """Source input or dynamic-choice paths from a retired node shape."""

    historical_inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                self.historical_inputs, tuple
            )
            or not self.historical_inputs
        ):
            raise ValueError("replacement migration requires historical input ids")
        if len(set(self.historical_inputs)) != len(self.historical_inputs):
            raise ValueError("replacement migration has duplicate historical input ids")
        for input_id in self.historical_inputs:
            if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                input_id, str
            ) or any(_STRUCTURAL_ID.fullmatch(segment) is None for segment in input_id.split(".")):
                raise ValueError(f"invalid historical input id: {input_id!r}")


@dataclass(frozen=True)
class ReplacementRule:
    """One rule migrates one source node type. Cases evaluate top-down,
    first match wins, and the last case must be unconditional so every node
    receives a deterministic migration target."""

    from_type: str  # "from" on the wire (Python keyword)
    cases: tuple[ReplacementCase, ...] = ()
    note: str = ""  # human note surfaced in diagnostics/review
    migration: ReplacementMigration | None = None

    def __post_init__(self) -> None:
        if not self.from_type:
            raise ValueError("replacement rule requires a source node type")
        if not self.cases:
            raise ValueError(f"rule for {self.from_type}: at least one case required")
        if self.migration is not None and not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.migration, ReplacementMigration
        ):
            raise ValueError("replacement rule migration must be ReplacementMigration")
        if not self.cases[-1].unconditional:
            raise ValueError(
                f"rule for {self.from_type}: the last case must be unconditional "
                "(the required fallback)"
            )
        if self.migration is not None and any(case.to != self.from_type for case in self.cases):
            raise ValueError(
                f"migration rule for {self.from_type}: every case must preserve the node type"
            )


# -- wire encoding (exact field names of replace/model.ts) --------------------


def predicate_to_wire(predicate: ReplacementPredicate) -> dict[str, object]:
    wire: dict[str, object] = {"kind": predicate.kind}
    if predicate.kind in _INPUT_PREDICATES:
        wire["input"] = predicate.input
    if predicate.kind == "valueEquals":
        wire["value"] = predicate.value
    if predicate.kind == "not":
        wire["of"] = predicate_to_wire(predicate.of[0])
    elif predicate.kind in ("all", "any"):
        wire["of"] = [predicate_to_wire(child) for child in predicate.of]
    return wire


def transform_to_wire(transform: ValueTransform) -> dict[str, object]:
    if transform.kind == "enumRename":
        return {"kind": "enumRename", "map": dict(transform.map)}
    wire: dict[str, object] = {"kind": "scale", "factor": transform.factor}
    if transform.offset != 0.0:
        wire["offset"] = transform.offset
    return wire


def mapping_source_to_wire(source: MappingSource) -> dict[str, object]:
    wire: dict[str, object] = {"kind": source.kind}
    if source.kind == "constant":
        wire["value"] = source.value
    else:
        wire["input"] = source.input
    if source.transform is not None:
        wire["transform"] = transform_to_wire(source.transform)
    return wire


def input_family_member_to_wire(member: InputFamilyMember) -> dict[str, object]:
    return {
        "suffix": member.suffix,
        "inputs": {target: mapping_source_to_wire(source) for target, source in member.inputs},
    }


def input_family_mapping_to_wire(mapping: InputFamilyMapping) -> dict[str, object]:
    if mapping.kind == "copy":
        return {
            "kind": "copy",
            "sourceFamily": mapping.source_family,
            "inputs": {target: mapping_source_to_wire(source) for target, source in mapping.inputs},
        }
    return {
        "kind": "members",
        "members": [input_family_member_to_wire(member) for member in mapping.members],
    }


def output_family_mapping_to_wire(mapping: OutputFamilyMapping) -> dict[str, object]:
    if mapping.kind == "copy":
        return {"kind": "copy", "sourceFamily": mapping.source_family}
    return {
        "kind": "members",
        "members": [
            {"suffix": member.suffix, "output": member.output} for member in mapping.members
        ],
    }


def node_to_wire(node: ReplacementNode) -> dict[str, object]:
    wire: dict[str, object] = {"type": node.type}
    if node.values:
        wire["values"] = dict(node.values)
    return wire


def link_to_wire(link: ReplacementLink) -> dict[str, object]:
    return {"from": link.from_address, "to": link.to}


def case_to_wire(case: ReplacementCase) -> dict[str, object]:
    wire: dict[str, object] = {"to": case.to}
    if case.when is not None:
        wire["when"] = predicate_to_wire(case.when)
    if case.nodes is not None:
        wire["nodes"] = {local_id: node_to_wire(node) for local_id, node in case.nodes}
    if case.slot_variants:
        wire["slotVariants"] = dict(case.slot_variants)
    if case.inputs:
        wire["inputs"] = {target: mapping_source_to_wire(source) for target, source in case.inputs}
    if case.input_families:
        wire["inputFamilies"] = {
            target: input_family_mapping_to_wire(mapping) for target, mapping in case.input_families
        }
    if case.output_families:
        wire["outputFamilies"] = {
            target: output_family_mapping_to_wire(mapping)
            for target, mapping in case.output_families
        }
    if case.links:
        wire["links"] = [link_to_wire(link) for link in case.links]
    if case.outputs:
        wire["outputs"] = dict(case.outputs)
    return wire


def rule_to_wire(rule: ReplacementRule) -> dict[str, object]:
    wire: dict[str, object] = {"from": rule.from_type}
    if rule.note:
        wire["note"] = rule.note
    if rule.migration is not None:
        wire["migration"] = {"historicalInputs": list(rule.migration.historical_inputs)}
    wire["cases"] = [case_to_wire(case) for case in rule.cases]
    return wire


def predicate_from_wire(wire: Any) -> ReplacementPredicate:
    kind = str(wire["kind"])
    if kind == "not":
        return ReplacementPredicate(kind="not", of=(predicate_from_wire(wire["of"]),))
    if kind in ("all", "any"):
        children = tuple(predicate_from_wire(child) for child in wire["of"])
        return ReplacementPredicate(kind=kind, of=children)  # pyright: ignore[reportArgumentType]
    if kind in _INPUT_PREDICATES:
        return ReplacementPredicate(
            kind=kind,  # pyright: ignore[reportArgumentType]
            input=str(wire["input"]),
            value=wire["value"] if kind == "valueEquals" else None,
        )
    # Constructor validation rejects unknown kinds.
    return ReplacementPredicate(kind=kind)  # pyright: ignore[reportArgumentType]


def transform_from_wire(wire: Any) -> ValueTransform:
    kind = str(wire["kind"])
    if kind == "enumRename":
        return ValueTransform.enum_rename({str(k): str(v) for k, v in wire["map"].items()})
    if kind == "scale":
        return ValueTransform(kind="scale", factor=wire["factor"], offset=wire.get("offset", 0.0))
    raise ValueError(f"unknown transform kind: {kind!r}")


def mapping_source_from_wire(wire: Any) -> MappingSource:
    if not isinstance(wire, Mapping):
        raise ValueError("mapping source must be an object")
    data = cast("Mapping[str, Any]", wire)
    kind = data.get("kind")
    if not isinstance(kind, str):
        raise ValueError("mapping source kind must be a string")
    if kind not in ("constant", "copy", "value", "link"):
        raise ValueError(f"unknown mapping source kind: {kind!r}")
    expected = {"kind", "value"} if kind == "constant" else {"kind", "input"}
    if kind == "value" and "transform" in data:
        expected.add("transform")
    if set(data) != expected:
        raise ValueError(f"{kind} mapping source contains unexpected fields")
    if kind != "constant" and not isinstance(data["input"], str):
        raise ValueError(f"{kind} mapping source input must be a string")
    transform_wire = data.get("transform")
    return MappingSource(
        kind=kind,  # pyright: ignore[reportArgumentType]  # constructor validates
        input=str(data.get("input", "")),
        transform=None if transform_wire is None else transform_from_wire(transform_wire),
        value=data.get("value") if kind == "constant" else None,
    )


def input_family_member_from_wire(wire: Any) -> InputFamilyMember:
    return InputFamilyMember(
        suffix=str(wire["suffix"]),
        inputs=tuple(
            (str(target), mapping_source_from_wire(source))
            for target, source in wire["inputs"].items()
        ),
    )


def input_family_mapping_from_wire(wire: Any) -> InputFamilyMapping:
    kind = str(wire["kind"])
    if kind == "copy":
        return InputFamilyMapping(
            kind="copy",
            source_family=str(wire["sourceFamily"]),
            inputs=tuple(
                (str(target), mapping_source_from_wire(source))
                for target, source in wire["inputs"].items()
            ),
        )
    if kind == "members":
        return InputFamilyMapping(
            kind="members",
            members=tuple(input_family_member_from_wire(member) for member in wire["members"]),
        )
    raise ValueError(f"unknown input family mapping kind: {kind!r}")


def output_family_mapping_from_wire(wire: Any) -> OutputFamilyMapping:
    if not isinstance(wire, Mapping):
        raise ValueError("output family mapping must be an object")
    data = cast("Mapping[str, Any]", wire)
    kind = data.get("kind")
    if not isinstance(kind, str):
        raise ValueError("output family mapping kind must be a string")
    if kind == "copy":
        if set(data) != {"kind", "sourceFamily"}:
            raise ValueError(
                "copy output family mapping must contain exactly kind and sourceFamily"
            )
        source_family = data["sourceFamily"]
        if not isinstance(source_family, str):
            raise ValueError("copy output family sourceFamily must be a string")
        return OutputFamilyMapping(kind="copy", source_family=source_family)
    if kind == "members":
        if set(data) != {"kind", "members"}:
            raise ValueError("members output family mapping must contain exactly kind and members")
        members = data["members"]
        if not isinstance(members, list):
            raise ValueError("output family members must be an array")
        decoded: list[OutputFamilyMember] = []
        for raw_member in cast("list[Any]", members):
            if not isinstance(raw_member, Mapping):
                raise ValueError("output family member must contain exactly suffix and output")
            member = cast("Mapping[str, Any]", raw_member)
            if set(member) != {"suffix", "output"}:
                raise ValueError("output family member must contain exactly suffix and output")
            decoded.append(OutputFamilyMember(suffix=member["suffix"], output=member["output"]))
        return OutputFamilyMapping(
            kind="members",
            members=tuple(decoded),
        )
    raise ValueError(f"unknown output family mapping kind: {kind!r}")


def node_from_wire(wire: Any) -> ReplacementNode:
    return ReplacementNode(
        type=wire["type"],
        values=tuple((key, value) for key, value in wire.get("values", {}).items()),
    )


def link_from_wire(wire: Any) -> ReplacementLink:
    return ReplacementLink(from_address=wire["from"], to=wire["to"])


def case_from_wire(wire: Any) -> ReplacementCase:
    when_wire = wire.get("when")
    expected = {
        "to",
        "when",
        "nodes",
        "slotVariants",
        "inputs",
        "inputFamilies",
        "outputFamilies",
        "links",
        "outputs",
    }
    unknown = set(wire) - expected
    if unknown:
        raise ValueError(
            f"replacement case contains unexpected fields: {', '.join(sorted(unknown))}"
        )
    slot_variants_wire = wire.get("slotVariants", {})
    if not isinstance(slot_variants_wire, Mapping):
        raise ValueError("replacement case slotVariants must be an object")
    slot_variants_data = cast("Mapping[object, object]", slot_variants_wire)
    if not all(
        isinstance(address, str) and isinstance(choice, str)
        for address, choice in slot_variants_data.items()
    ):
        raise ValueError(
            "replacement case slotVariants must map string addresses to string choices"
        )
    return ReplacementCase(
        to=str(wire["to"]),
        when=None if when_wire is None else predicate_from_wire(when_wire),
        nodes=(
            tuple((str(local_id), node_from_wire(node)) for local_id, node in wire["nodes"].items())
            if "nodes" in wire
            else None
        ),
        slot_variants=tuple(
            (str(target), str(choice)) for target, choice in slot_variants_data.items()
        ),
        inputs=tuple(
            (str(target), mapping_source_from_wire(source))
            for target, source in wire.get("inputs", {}).items()
        ),
        input_families=tuple(
            (str(target), input_family_mapping_from_wire(mapping))
            for target, mapping in wire.get("inputFamilies", {}).items()
        ),
        output_families=tuple(
            (str(target), output_family_mapping_from_wire(mapping))
            for target, mapping in wire.get("outputFamilies", {}).items()
        ),
        links=tuple(link_from_wire(link) for link in wire.get("links", [])),
        outputs=tuple(
            (str(target), str(source)) for target, source in wire.get("outputs", {}).items()
        ),
    )


def rule_from_wire(wire: Any) -> ReplacementRule:
    migration = None
    if "migration" in wire:
        migration_wire = wire["migration"]
        if not isinstance(migration_wire, Mapping):
            raise ValueError("replacement migration must be an object")
        migration_data = cast("Mapping[str, Any]", migration_wire)
        if set(migration_data) != {"historicalInputs"}:
            raise ValueError("replacement migration must contain exactly historicalInputs")
        historical_inputs = migration_data["historicalInputs"]
        if not isinstance(historical_inputs, list) or not all(
            isinstance(input_id, str) for input_id in cast("list[Any]", historical_inputs)
        ):
            raise ValueError("replacement migration historicalInputs must be an array of strings")
        migration = ReplacementMigration(tuple(cast("list[str]", historical_inputs)))
    return ReplacementRule(
        from_type=str(wire["from"]),
        note=str(wire.get("note", "")),
        cases=tuple(case_from_wire(case) for case in wire["cases"]),
        migration=migration,
    )


# -- cross-schema reference checks (tooling: dinkster doctor) ---------------------


@dataclass(frozen=True)
class ReplacementProblem:
    """One invalid static-id reference in a carried replacement rule.

    Machine anchors first (the frontend keys badges/affordances on them),
    human prose alongside: ``carrier`` ships the rule, ``from_type`` is the
    rule's predecessor, ``ref`` is the offending id, and ``target`` is the
    schema the ref was checked against (the predecessor for source-side
    refs, the case's ``to`` for target-side refs)."""

    carrier: str
    from_type: str
    case_index: int
    ref: str
    ref_kind: Literal["input", "output"]
    target: str
    message: str


def problem_to_wire(problem: ReplacementProblem) -> dict[str, object]:
    """The /api/diagnostics entry shape (agreed with Dinkster-Frontend):
    structured anchors plus the human string; growth is additive."""
    return {
        "carrier": problem.carrier,
        "from": problem.from_type,
        "caseIndex": problem.case_index,
        "ref": problem.ref,
        "refKind": problem.ref_kind,
        "target": problem.target,
        "message": problem.message,
    }


def _predicate_input_refs(predicate: ReplacementPredicate) -> tuple[str, ...]:
    if predicate.kind in _INPUT_PREDICATES:
        return (predicate.input,)
    refs: list[str] = []
    for child in predicate.of:
        refs.extend(_predicate_input_refs(child))
    return tuple(refs)


def _check_static_id(
    ref: str,
    static_ids: frozenset[str],
    family_ids: frozenset[str],
    schema_type: str,
    role: str,
) -> str | None:
    if ref in static_ids:
        return None
    if ref in family_ids or any(ref.startswith(fam + ".") for fam in family_ids):
        return (
            f"'{ref}' is a dynamic family {role} of {schema_type}; replacement "
            "rules reference static interface ids only"
        )
    return f"'{ref}' is not a static {role} id of {schema_type}"


def _static_family_template_ids(family: object) -> frozenset[str] | None:
    from .model import InputFamilySpec, InputSpec

    assert isinstance(family, InputFamilySpec)
    if not all(isinstance(entry, InputSpec) for entry in family.template):
        return None
    return frozenset(entry.id for entry in family.template)


def _has_dynamic_constructs(schema: NodeSchema) -> bool:
    from .model import DynamicComboSpec, DynamicEntry, DynamicSlotSpec, InputFamilySpec

    def contains(entries: Sequence[DynamicEntry]) -> bool:
        return any(
            isinstance(entry, (DynamicComboSpec, DynamicSlotSpec))
            or (isinstance(entry, InputFamilySpec) and contains(entry.template))
            for entry in entries
        )

    return contains((*schema.input_families, *schema.combos, *schema.slots))


def _dynamic_construct_info(schema: NodeSchema, path: str) -> tuple[frozenset[str], frozenset[str]]:
    from .model import DynamicComboSpec, DynamicEntry, DynamicSlotSpec

    kinds: set[str] = set()
    choices: set[str] = set()

    def walk(entries: Sequence[DynamicEntry], parts: Sequence[str]) -> None:
        if not parts:
            return
        for entry in entries:
            if entry.id != parts[0]:
                continue
            if isinstance(entry, DynamicComboSpec):
                if len(parts) == 1:
                    kinds.add("combo")
                    choices.update(option.key for option in entry.options)
                else:
                    for option in entry.options:
                        walk(option.inputs, parts[1:])
            elif isinstance(entry, DynamicSlotSpec):
                if len(parts) == 1:
                    if entry.variants is not None:
                        kinds.add("slot")
                        for variant in entry.variants:
                            choices.add(variant.key)
                else:
                    walk(entry.inputs, parts[1:])
                    for variant in entry.variants or ():
                        walk(variant.inputs, parts[1:])

    walk((*schema.input_families, *schema.combos, *schema.slots), path.split("."))
    return frozenset(kinds), frozenset(choices)


def _is_materializable_input_path(schema: NodeSchema, path: str) -> bool:
    from .model import DynamicComboSpec, DynamicEntry, DynamicSlotSpec, InputSpec

    if any(spec.id == path for spec in schema.inputs):
        return True

    def matches(entries: Sequence[DynamicEntry], parts: Sequence[str]) -> bool:
        if not parts:
            return False
        for entry in entries:
            if entry.id != parts[0]:
                continue
            if isinstance(entry, InputSpec):
                return len(parts) == 1
            if isinstance(entry, DynamicComboSpec):
                if any(matches(option.inputs, parts[1:]) for option in entry.options):
                    return True
            elif isinstance(entry, DynamicSlotSpec):
                if len(parts) == 1:
                    return True
                if matches(entry.inputs, parts[1:]) or any(
                    matches(variant.inputs, parts[1:]) for variant in entry.variants or ()
                ):
                    return True
        return False

    return matches(
        (*schema.input_families, *schema.combos, *schema.slots),
        path.split("."),
    )


def validate_replacement_references(
    schemas: Mapping[str, NodeSchema],
) -> tuple[ReplacementProblem, ...]:
    """Cross-schema reference checks for every replacement rule carried by
    ``schemas``. Pure and tolerant of absence: a rule's predecessor or target
    that is not in the mapping is simply unchecked (rules legitimately
    migrate nodes from never-installed packs). Source input references name
    static ids, materializable paths or combo selectors declared in any branch,
    plus a same-type migration's declared historical inputs. Active source
    branches and values are the importer's responsibility. Source output
    references name static ids.
    ``inputFamilies`` additionally checks top-level family ids and static
    template-local input ids. ``outputFamilies`` checks top-level family ids,
    source outputs, and target member bounds. A case with ``slotVariants`` is
    elaborated before its target input paths are checked. Returns structured
    problems (``problem_to_wire`` gives the /api/diagnostics shape); empty
    means clean. Doctor surfaces these as errors (code
    ``schema.replacement-invalid``)."""
    from .elaborate import MAX_FAMILY_MEMBERS, ElaborationError, elaborate

    problems: list[ReplacementProblem] = []
    for carrier_type, carrier in schemas.items():
        for rule in carrier.replacements:
            source: NodeSchema | None = schemas.get(rule.from_type)
            for index, case in enumerate(rule.cases):
                where = f"{carrier_type}: rule from {rule.from_type}, case {index} -> {case.to}"

                def add(
                    ref: str,
                    ref_kind: Literal["input", "output"],
                    checked_against: str,
                    detail: str,
                    *,
                    _where: str = where,
                    _index: int = index,
                    _from: str = rule.from_type,
                    _carrier: str = carrier_type,
                ) -> None:
                    problems.append(
                        ReplacementProblem(
                            carrier=_carrier,
                            from_type=_from,
                            case_index=_index,
                            ref=ref,
                            ref_kind=ref_kind,
                            target=checked_against,
                            message=f"{_where}: {detail}",
                        )
                    )

                connection_consumers: dict[str, list[str]] = {}
                for target, mapping in case.inputs:
                    if mapping.kind in ("copy", "link"):
                        connection_consumers.setdefault(mapping.input, []).append(target)
                family_connection_consumers: dict[tuple[str, str], list[str]] = {}
                for target_family, family_mapping in case.input_families:
                    if family_mapping.kind == "members":
                        for member in family_mapping.members:
                            for target, mapping in member.inputs:
                                if mapping.kind in ("copy", "link"):
                                    connection_consumers.setdefault(mapping.input, []).append(
                                        f"{target_family}.{member.suffix}.{target}"
                                    )
                    else:
                        for target, mapping in family_mapping.inputs:
                            if mapping.kind in ("copy", "link"):
                                family_connection_consumers.setdefault(
                                    (family_mapping.source_family, mapping.input), []
                                ).append(f"{target_family}.*.{target}")
                for source_input, targets in connection_consumers.items():
                    if len(targets) > 1:
                        add(
                            source_input,
                            "input",
                            rule.from_type,
                            f"source input '{source_input}' connection is consumed by multiple "
                            f"target mappings: {', '.join(targets)}",
                        )
                for (source_family, source_input), targets in family_connection_consumers.items():
                    if len(targets) > 1:
                        add(
                            source_input,
                            "input",
                            rule.from_type,
                            "source family input "
                            f"'{source_family}.*.{source_input}' connection "
                            f"is consumed by multiple target mappings: {', '.join(targets)}",
                        )

                if source is not None:
                    src_inputs = frozenset(spec.id for spec in source.inputs)
                    if rule.migration is not None:
                        src_inputs = src_inputs.union(rule.migration.historical_inputs)
                    src_outputs = frozenset(spec.id for spec in source.outputs)
                    src_in_fams = frozenset(f.id for f in source.input_families)
                    src_out_fams = frozenset(f.id for f in source.output_families)
                    refs: list[str] = []
                    if case.when is not None:
                        refs.extend(_predicate_input_refs(case.when))
                    for _, mapping in case.inputs:
                        if mapping.kind != "constant":
                            refs.append(mapping.input)
                    for _, family_mapping in case.input_families:
                        if family_mapping.kind != "members":
                            continue
                        for member in family_mapping.members:
                            for _, mapping in member.inputs:
                                if mapping.kind != "constant":
                                    refs.append(mapping.input)
                    for ref in refs:
                        if _is_materializable_input_path(source, ref) or (
                            "combo" in _dynamic_construct_info(source, ref)[0]
                        ):
                            continue
                        detail = _check_static_id(
                            ref, src_inputs, src_in_fams, rule.from_type, "input"
                        )
                        if detail is not None:
                            add(ref, "input", rule.from_type, detail)
                    source_families = {family.id: family for family in source.input_families}
                    for _, family_mapping in case.input_families:
                        if family_mapping.kind != "copy":
                            continue
                        family = source_families.get(family_mapping.source_family)
                        if family is None:
                            add(
                                family_mapping.source_family,
                                "input",
                                rule.from_type,
                                f"'{family_mapping.source_family}' is not an input family id "
                                f"of {rule.from_type}",
                            )
                            continue
                        template_ids = _static_family_template_ids(family)
                        if template_ids is None:
                            add(
                                family_mapping.source_family,
                                "input",
                                rule.from_type,
                                f"input family '{family_mapping.source_family}' of "
                                f"{rule.from_type} has nested dynamic entries",
                            )
                            continue
                        for _, mapping in family_mapping.inputs:
                            if mapping.kind == "constant":
                                continue
                            if mapping.input not in template_ids:
                                add(
                                    mapping.input,
                                    "input",
                                    rule.from_type,
                                    f"'{mapping.input}' is not a template input id of family "
                                    f"'{family_mapping.source_family}' in {rule.from_type}",
                                )
                    for _, source_output in case.outputs:
                        detail = _check_static_id(
                            source_output,
                            src_outputs,
                            src_out_fams,
                            rule.from_type,
                            "output",
                        )
                        if detail is not None:
                            add(source_output, "output", rule.from_type, detail)
                    source_output_families = {
                        family.id: family for family in source.output_families
                    }
                    for _, family_mapping in case.output_families:
                        if family_mapping.kind == "copy":
                            if family_mapping.source_family not in source_output_families:
                                add(
                                    family_mapping.source_family,
                                    "output",
                                    rule.from_type,
                                    f"'{family_mapping.source_family}' is not an output family "
                                    f"id of {rule.from_type}",
                                )
                            continue
                        for member in family_mapping.members:
                            detail = _check_static_id(
                                member.output,
                                src_outputs,
                                src_out_fams,
                                rule.from_type,
                                "output",
                            )
                            if detail is not None:
                                add(member.output, "output", rule.from_type, detail)
                helper_types = {local_id: node.type for local_id, node in case.nodes or ()}
                effective_input_ids: dict[str, frozenset[str] | None] = {}

                def split_target_address(
                    address: str, *, _helpers_enabled: bool = case.nodes is not None
                ) -> tuple[str, str]:
                    if _helpers_enabled and ":" in address:
                        local_id, ref = address.split(":")
                        return local_id, ref
                    return "", address

                schema_types_by_local = {"": case.to, **helper_types}
                choices_by_local: dict[str, dict[str, str]] = {}
                invalid_choice_locals: set[str] = set()
                dynamic_locals = {
                    local_id
                    for local_id, schema_type in schema_types_by_local.items()
                    if (schema := schemas.get(schema_type)) is not None
                    and _has_dynamic_constructs(schema)
                }
                dynamic_locals.update(
                    split_target_address(address)[0] for address, _ in case.input_families
                )
                family_owned_inputs: dict[str, set[str]] = {}
                for address, choice in case.slot_variants:
                    local_id, ref = split_target_address(address)
                    dynamic_locals.add(local_id)
                    schema_type = schema_types_by_local[local_id]
                    schema = schemas.get(schema_type)
                    if schema is None:
                        continue
                    kinds, valid_choices = _dynamic_construct_info(schema, ref)
                    if not kinds:
                        add(
                            ref,
                            "input",
                            schema_type,
                            f"'{ref}' is not a dynamic construct path of {schema_type}",
                        )
                        invalid_choice_locals.add(local_id)
                    elif choice not in valid_choices:
                        add(
                            ref,
                            "input",
                            schema_type,
                            f"slot variant '{ref}' names unknown choice {choice!r}",
                        )
                        invalid_choice_locals.add(local_id)
                    else:
                        choices_by_local.setdefault(local_id, {})[ref] = choice

                if dynamic_locals:
                    evidence_by_local: dict[str, list[str]] = {}

                    for address, _mapping in case.inputs:
                        local_id, ref = split_target_address(address)
                        evidence_by_local.setdefault(local_id, []).append(ref)
                    for link in case.links:
                        local_id, ref = split_target_address(link.to)
                        evidence_by_local.setdefault(local_id, []).append(ref)
                    for local_id, node in case.nodes or ():
                        evidence_by_local.setdefault(local_id, []).extend(
                            ref for ref, _value in node.values
                        )
                    source_families = (
                        {}
                        if source is None
                        else {family.id: family for family in source.input_families}
                    )
                    for target_family_address, family_mapping in case.input_families:
                        local_id, target_family = split_target_address(target_family_address)
                        target_schema = schemas.get(schema_types_by_local[local_id])
                        family = (
                            None
                            if target_schema is None
                            else next(
                                (
                                    candidate
                                    for candidate in target_schema.input_families
                                    if candidate.id == target_family
                                ),
                                None,
                            )
                        )
                        if family is None:
                            continue
                        if family_mapping.kind == "members":
                            suffixes = tuple(member.suffix for member in family_mapping.members)
                        else:
                            source_family = source_families.get(family_mapping.source_family)
                            member_count = (
                                family.min_members
                                if source_family is None
                                else source_family.min_members
                            )
                            suffixes = (
                                ()
                                if source_family is None or source_family.member_names is None
                                else source_family.member_names[:member_count]
                            )
                            if len(suffixes) < member_count:
                                suffixes = (
                                    family.member_names[:member_count]
                                    if family.member_names is not None
                                    else tuple(
                                        str(member_index) for member_index in range(member_count)
                                    )
                                )
                        evidence_by_local.setdefault(local_id, []).extend(
                            f"{target_family}.{suffix}" for suffix in suffixes
                        )
                        template_ids = _static_family_template_ids(family)
                        if template_ids is not None:
                            owned = family_owned_inputs.setdefault(local_id, set())
                            if len(template_ids) == 1:
                                owned.update(f"{target_family}.{suffix}" for suffix in suffixes)
                            else:
                                owned.update(
                                    f"{target_family}.{suffix}.{input_id}"
                                    for suffix in suffixes
                                    for input_id in template_ids
                                )

                    for local_id, schema_type in schema_types_by_local.items():
                        if local_id not in dynamic_locals:
                            continue
                        schema = schemas.get(schema_type)
                        if schema is None:
                            continue
                        if local_id in invalid_choice_locals:
                            effective_input_ids[local_id] = None
                            continue
                        choices = choices_by_local.get(local_id, {})
                        evidence = list(evidence_by_local.get(local_id, ()))
                        try:
                            effective = elaborate(
                                replace(schema, output_families=()),
                                evidence,
                                slot_variants=choices,
                            )
                        except ElaborationError as error:
                            effective_input_ids[local_id] = None
                            detail = str(error)
                            problem_ref = max(
                                (ref for ref in choices if ref in detail),
                                key=len,
                                default=next(iter(choices), schema_type),
                            )
                            add(problem_ref, "input", schema_type, detail)
                        else:
                            effective_input_ids[local_id] = frozenset(
                                spec.id for spec in effective.inputs
                            )

                def check_target_ref(
                    address: str,
                    ref_kind: Literal["input", "output"],
                    *,
                    allow_family_member: bool = False,
                    _helpers_enabled: bool = case.nodes is not None,
                    _helper_types: Mapping[str, str] = helper_types,
                    _effective_input_ids: Mapping[str, frozenset[str] | None] = effective_input_ids,
                    _family_owned_inputs: Mapping[str, set[str]] = family_owned_inputs,
                    _primary_type: str = case.to,
                ) -> None:
                    if _helpers_enabled and ":" in address:
                        local_id, ref = address.split(":")
                        schema_type = _helper_types[local_id]
                    else:
                        local_id = ""
                        ref = address
                        schema_type = _primary_type
                    schema = schemas.get(schema_type)
                    if schema is None:
                        return
                    if ref_kind == "input":
                        if local_id in _effective_input_ids:
                            selected_inputs = _effective_input_ids[local_id]
                            if selected_inputs is None:
                                if not _is_materializable_input_path(schema, ref):
                                    add(
                                        ref,
                                        "input",
                                        schema_type,
                                        f"'{ref}' is not a materializable input path "
                                        f"of {schema_type}",
                                    )
                                return
                            if ref not in selected_inputs:
                                add(
                                    ref,
                                    "input",
                                    schema_type,
                                    f"'{ref}' is not an input id of the selected interface "
                                    f"for {schema_type}",
                                )
                            elif not allow_family_member and ref in _family_owned_inputs.get(
                                local_id, set()
                            ):
                                add(
                                    ref,
                                    "input",
                                    schema_type,
                                    f"'{ref}' is owned by an input family mapping for "
                                    f"{schema_type}",
                                )
                            return
                        static_ids = frozenset(spec.id for spec in schema.inputs)
                        family_ids = frozenset(f.id for f in schema.input_families)
                        role = "input"
                    else:
                        static_ids = frozenset(spec.id for spec in schema.outputs)
                        family_ids = frozenset(f.id for f in schema.output_families)
                        role = "output"
                    detail = _check_static_id(ref, static_ids, family_ids, schema_type, role)
                    if detail is not None:
                        add(ref, ref_kind, schema_type, detail)

                for target_input, _ in case.inputs:
                    check_target_ref(target_input, "input")
                target_schema = schemas.get(case.to)
                if target_schema is not None:
                    for target_family_address, family_mapping in case.input_families:
                        local_id, target_family = split_target_address(target_family_address)
                        family_schema_type = schema_types_by_local[local_id]
                        family_schema = schemas.get(family_schema_type)
                        if family_schema is None:
                            continue
                        target_families = {
                            family.id: family for family in family_schema.input_families
                        }
                        family = target_families.get(target_family)
                        if family is None:
                            add(
                                target_family_address,
                                "input",
                                family_schema_type,
                                f"'{target_family}' is not an input family id of "
                                f"{family_schema_type}",
                            )
                            continue
                        template_ids = _static_family_template_ids(family)
                        if template_ids is None:
                            add(
                                target_family_address,
                                "input",
                                family_schema_type,
                                f"input family '{target_family}' of "
                                f"{family_schema_type} has nested "
                                "dynamic entries",
                            )
                            continue
                        mappings = (
                            (family_mapping.inputs,)
                            if family_mapping.kind == "copy"
                            else tuple(member.inputs for member in family_mapping.members)
                        )
                        for inputs in mappings:
                            for target_input, _ in inputs:
                                if target_input not in template_ids:
                                    add(
                                        target_input,
                                        "input",
                                        family_schema_type,
                                        f"'{target_input}' is not a template input id of "
                                        f"family '{target_family}' in {family_schema_type}",
                                    )
                        if family_mapping.kind == "copy" and source is not None:
                            source_family = next(
                                (
                                    candidate
                                    for candidate in source.input_families
                                    if candidate.id == family_mapping.source_family
                                ),
                                None,
                            )
                            if source_family is not None:
                                if family.min_members > source_family.min_members:
                                    add(
                                        target_family_address,
                                        "input",
                                        family_schema_type,
                                        f"family '{target_family}' in "
                                        f"{family_schema_type} requires more "
                                        "members than the copied source family guarantees",
                                    )
                                if family.max_members is not None and (
                                    source_family.max_members is None
                                    or source_family.max_members > family.max_members
                                ):
                                    add(
                                        target_family_address,
                                        "input",
                                        family_schema_type,
                                        f"family '{target_family}' in "
                                        f"{family_schema_type} cannot admit "
                                        "every copied source-family member count",
                                    )
                                if family.member_names is not None and (
                                    source_family.member_names is None
                                    or not set(source_family.member_names).issubset(
                                        family.member_names
                                    )
                                ):
                                    add(
                                        target_family_address,
                                        "input",
                                        family_schema_type,
                                        f"family '{target_family}' in "
                                        f"{family_schema_type} cannot admit "
                                        "every copied source-family suffix",
                                    )
                        if family_mapping.kind == "members" and family.member_names is not None:
                            for member in family_mapping.members:
                                if member.suffix not in family.member_names:
                                    add(
                                        member.suffix,
                                        "input",
                                        family_schema_type,
                                        f"'{member.suffix}' is not an admitted member name of "
                                        f"family '{target_family}' in {family_schema_type}",
                                    )
                        if family_mapping.kind == "members":
                            member_count = len(family_mapping.members)
                            if member_count < family.min_members or (
                                family.max_members is not None and member_count > family.max_members
                            ):
                                add(
                                    target_family_address,
                                    "input",
                                    family_schema_type,
                                    f"family '{target_family}' in "
                                    f"{family_schema_type} does not admit "
                                    f"{member_count} explicit members",
                                )
                    target_output_families = {
                        family.id: family for family in target_schema.output_families
                    }
                    mapped_target_inputs = dict(case.inputs)
                    for target_family, family_mapping in case.output_families:
                        family = target_output_families.get(target_family)
                        if family is None:
                            add(
                                target_family,
                                "output",
                                case.to,
                                f"'{target_family}' is not an output family id of {case.to}",
                            )
                            continue
                        if family_mapping.kind == "copy" and source is not None:
                            source_family = next(
                                (
                                    candidate
                                    for candidate in source.output_families
                                    if candidate.id == family_mapping.source_family
                                ),
                                None,
                            )
                            if source_family is not None:
                                if family.min_members > source_family.min_members:
                                    add(
                                        target_family,
                                        "output",
                                        case.to,
                                        f"family '{target_family}' in {case.to} requires more "
                                        "members than the copied source family guarantees",
                                    )
                                if family.max_members is not None and (
                                    source_family.max_members is None
                                    or source_family.max_members > family.max_members
                                ):
                                    add(
                                        target_family,
                                        "output",
                                        case.to,
                                        f"family '{target_family}' in {case.to} cannot admit "
                                        "every copied source-family member count",
                                    )
                                if family.count is not None and source_family.count is None:
                                    add(
                                        target_family,
                                        "output",
                                        case.to,
                                        f"count-bound family '{target_family}' in {case.to} "
                                        "cannot copy an unbound source family",
                                    )
                                if family.count is not None and source_family.count is not None:
                                    count_mapping = mapped_target_inputs.get(family.count.input)
                                    if not (
                                        count_mapping is not None
                                        and count_mapping.kind in ("copy", "value")
                                        and count_mapping.input == source_family.count.input
                                        and count_mapping.transform is None
                                    ):
                                        add(
                                            family.count.input,
                                            "input",
                                            case.to,
                                            f"count input '{family.count.input}' for output family "
                                            f"'{target_family}' in {case.to} must preserve source "
                                            f"count input '{source_family.count.input}'",
                                        )
                        if family_mapping.kind == "members":
                            member_count = len(family_mapping.members)
                            if member_count > MAX_FAMILY_MEMBERS:
                                add(
                                    target_family,
                                    "output",
                                    case.to,
                                    f"family '{target_family}' in {case.to} exceeds the "
                                    f"{MAX_FAMILY_MEMBERS} member budget",
                                )
                            if family.count is not None:
                                count_mapping = mapped_target_inputs.get(family.count.input)
                                if not (
                                    count_mapping is not None
                                    and count_mapping.kind == "constant"
                                    and type(count_mapping.value) is int
                                    and count_mapping.value == member_count
                                ):
                                    add(
                                        family.count.input,
                                        "input",
                                        case.to,
                                        f"count input '{family.count.input}' for output family "
                                        f"'{target_family}' in {case.to} must be the constant "
                                        f"member count {member_count}",
                                    )
                            if member_count < family.min_members or (
                                family.max_members is not None and member_count > family.max_members
                            ):
                                add(
                                    target_family,
                                    "output",
                                    case.to,
                                    f"family '{target_family}' in {case.to} does not admit "
                                    f"{member_count} explicit members",
                                )
                            suffixes = tuple(member.suffix for member in family_mapping.members)
                            if family.count is not None and suffixes != tuple(
                                str(member_index) for member_index in range(member_count)
                            ):
                                add(
                                    target_family,
                                    "output",
                                    case.to,
                                    f"count-bound family '{target_family}' in {case.to} requires "
                                    "canonical zero-based explicit member suffixes",
                                )
                for target_output, _ in case.outputs:
                    check_target_ref(target_output, "output")
                for link in case.links:
                    check_target_ref(link.from_address, "output")
                    check_target_ref(link.to, "input", allow_family_member=True)
                for local_id, node in case.nodes or ():
                    for ref, _ in node.values:
                        check_target_ref(f"{local_id}:{ref}", "input")
    return tuple(problems)
