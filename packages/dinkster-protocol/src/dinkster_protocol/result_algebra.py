"""Immutable declarations for the dormant invocation result algebra."""
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations
from dinkster_values import MEBIBYTE

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from dinkster_values import Value

from . import ErrorHint, NodeError

RESULT_ALGEBRA_CAPABILITY = "dinkster.invocation-result-algebra.v1"
RESULT_ALGEBRA_VERSION = 1
RESULT_ALGEBRA_MAX_COUNT = 4096
RESULT_ALGEBRA_MAX_LITERAL_DEPTH = 32
RESULT_ALGEBRA_MAX_LITERAL_ITEMS = 16384
RESULT_ALGEBRA_MAX_DOCUMENT_BYTES = MEBIBYTE
RESULT_ALGEBRA_MAX_NESTING = 64

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _identifier(value: object, name: str) -> None:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"{name} is not a valid identifier")


def _pairs(value: object, name: str) -> tuple[tuple[str, object], ...]:
    if type(value) is not tuple or len(value) > RESULT_ALGEBRA_MAX_COUNT:
        raise ValueError(f"{name} must be a bounded tuple")
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError(f"{name} entries must be pairs")
        _identifier(item[0], f"{name} key")
    result = value
    keys = tuple(item[0] for item in result)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{name} must be uniquely sorted")
    return result


@dataclass(frozen=True)
class JsonObject:
    fields: tuple[tuple[str, JsonLiteral], ...]

    def __post_init__(self) -> None:
        pairs = _pairs(self.fields, "literal object fields")
        for _, value in pairs:
            _literal(value)


JsonLiteral: TypeAlias = None | bool | int | float | str | tuple["JsonLiteral", ...] | JsonObject


def _text(value: object, name: str) -> None:
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} contains an invalid Unicode scalar") from exc


def _literal(value: object, depth: int = 0, budget: list[int] | None = None) -> JsonLiteral:
    if budget is None:
        budget = [RESULT_ALGEBRA_MAX_LITERAL_ITEMS]
    budget[0] -= 1
    if budget[0] < 0 or depth > RESULT_ALGEBRA_MAX_LITERAL_DEPTH:
        raise ValueError("literal exceeds cumulative item or depth limit")
    if value is None or type(value) is bool:
        return value  # type: ignore[return-value]
    if type(value) is str:
        _text(value, "literal string")
        return value
    if type(value) is int:
        if not -(2**63) <= value < 2**63:
            raise ValueError("literal integer is outside signed 64-bit range")
        return value
    if type(value) is float:
        if not math.isfinite(value) or (value == 0 and math.copysign(1, value) < 0):
            raise ValueError("literal float must be finite and not negative zero")
        return value
    if isinstance(value, JsonObject):
        fields = _pairs(value.fields, "literal object fields")
        return JsonObject(tuple((key, _literal(item, depth + 1, budget)) for key, item in fields))
    if type(value) is tuple:
        return tuple(_literal(item, depth + 1, budget) for item in value)
    raise ValueError("literal must be immutable RPC-clean JSON")


@dataclass(frozen=True)
class PresentOutput:
    value: Value

    def __post_init__(self) -> None:
        if not isinstance(self.value, Value):
            raise ValueError("present output must contain a Value")


@dataclass(frozen=True)
class BlockedOutput:
    message: str | None = None

    def __post_init__(self) -> None:
        if self.message is not None and type(self.message) is not str:
            raise ValueError("blocked message must be a string or None")
        if self.message is not None:
            _text(self.message, "blocked message")


@dataclass(frozen=True)
class CurrentOutputRef:
    output_id: str

    def __post_init__(self) -> None:
        _identifier(self.output_id, "output_id")


@dataclass(frozen=True)
class LocalOutputRef:
    local_node_id: str
    output_id: str

    def __post_init__(self) -> None:
        _identifier(self.local_node_id, "local_node_id")
        _identifier(self.output_id, "output_id")


@dataclass(frozen=True)
class CurrentNodeRef:
    pass


@dataclass(frozen=True)
class LocalNodeRef:
    local_node_id: str

    def __post_init__(self) -> None:
        _identifier(self.local_node_id, "local_node_id")


NodeMetadataRef: TypeAlias = CurrentNodeRef | LocalNodeRef


@dataclass(frozen=True)
class LiteralInput:
    value: JsonLiteral

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _literal(self.value))


InputBinding: TypeAlias = LiteralInput | CurrentOutputRef | LocalOutputRef
OutputBinding: TypeAlias = PresentOutput | BlockedOutput | CurrentOutputRef | LocalOutputRef


@dataclass(frozen=True)
class LocalNode:
    local_id: str
    node_type: str
    inputs: tuple[tuple[str, InputBinding], ...] = ()
    parent: NodeMetadataRef | None = None
    display: NodeMetadataRef | None = None

    def __post_init__(self) -> None:
        _identifier(self.local_id, "local_id")
        _identifier(self.node_type, "node_type")
        pairs = _pairs(self.inputs, "inputs")
        if not all(
            isinstance(item[1], (LiteralInput, CurrentOutputRef, LocalOutputRef)) for item in pairs
        ):
            raise ValueError("invalid input binding")
        if self.parent is not None and not isinstance(self.parent, (CurrentNodeRef, LocalNodeRef)):
            raise ValueError("invalid parent metadata reference")
        if self.display is not None and not isinstance(
            self.display, (CurrentNodeRef, LocalNodeRef)
        ):
            raise ValueError("invalid display metadata reference")


def _local_refs(value: object) -> Sequence[str]:
    if isinstance(value, (LocalOutputRef, LocalNodeRef)):
        return (value.local_node_id,)
    return ()


@dataclass(frozen=True)
class LocalExpansion:
    nodes: tuple[LocalNode, ...]

    def __post_init__(self) -> None:
        if (
            type(self.nodes) is not tuple
            or not self.nodes
            or len(self.nodes) > RESULT_ALGEBRA_MAX_COUNT
        ):
            raise ValueError("expansion nodes must be a nonempty bounded tuple")
        if not all(isinstance(node, LocalNode) for node in self.nodes):
            raise ValueError("expansion contains an invalid node")
        ids = tuple(node.local_id for node in self.nodes)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("local nodes must be uniquely sorted")
        declared = set(ids)
        edges: dict[str, set[str]] = {item: set() for item in ids}
        for node in self.nodes:
            refs = [*(_local_refs(node.parent)), *(_local_refs(node.display))]
            refs.extend(ref for _, binding in node.inputs for ref in _local_refs(binding))
            if any(ref not in declared for ref in refs):
                raise ValueError("local reference escapes expansion")
            edges[node.local_id].update(refs)
        state: dict[str, int] = {}
        for root in ids:
            if state.get(root) == 2:
                continue
            pending: list[tuple[str, bool]] = [(root, False)]
            while pending:
                node_id, leaving = pending.pop()
                if leaving:
                    state[node_id] = 2
                    continue
                if state.get(node_id) == 1:
                    raise ValueError("local expansion contains a cycle")
                if state.get(node_id) == 2:
                    continue
                state[node_id] = 1
                pending.append((node_id, True))
                pending.extend((target, False) for target in edges[node_id])


def _validate_bindings(bindings: object, *, local_ids: set[str]) -> None:
    pairs = _pairs(bindings, "bindings")
    if not all(
        isinstance(item[1], (PresentOutput, BlockedOutput, CurrentOutputRef, LocalOutputRef))
        for item in pairs
    ):
        raise ValueError("invalid output binding")
    present = {key for key, value in pairs if isinstance(value, PresentOutput)}
    for _, value in pairs:
        if isinstance(value, CurrentOutputRef) and value.output_id not in present:
            raise ValueError("current reference must resolve directly to same-unit present output")
        if isinstance(value, LocalOutputRef) and value.local_node_id not in local_ids:
            raise ValueError("local return reference escapes expansion")


@dataclass(frozen=True)
class DirectReturn:
    bindings: tuple[tuple[str, OutputBinding], ...]

    def __post_init__(self) -> None:
        _validate_bindings(self.bindings, local_ids=set())


@dataclass(frozen=True)
class ExpandedReturn:
    expansion: LocalExpansion
    bindings: tuple[tuple[str, OutputBinding], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.expansion, LocalExpansion):
            raise ValueError("invalid expansion")
        _validate_bindings(
            self.bindings, local_ids={node.local_id for node in self.expansion.nodes}
        )
        present = {
            output_id for output_id, binding in self.bindings if isinstance(binding, PresentOutput)
        }
        for node in self.expansion.nodes:
            for _, binding in node.inputs:
                if isinstance(binding, CurrentOutputRef) and binding.output_id not in present:
                    raise ValueError(
                        "current expansion input must resolve to same-unit present output"
                    )


ReturnUnit: TypeAlias = DirectReturn | ExpandedReturn


@dataclass(frozen=True)
class ReturnBatch:
    mode: Literal["scalar", "mapped"]
    units: tuple[ReturnUnit, ...]

    def __post_init__(self) -> None:
        if self.mode not in ("scalar", "mapped"):
            raise ValueError("mode must be scalar or mapped")
        if type(self.units) is not tuple or len(self.units) > RESULT_ALGEBRA_MAX_COUNT:
            raise ValueError("units must be a bounded tuple")
        if not all(isinstance(unit, (DirectReturn, ExpandedReturn)) for unit in self.units):
            raise ValueError("invalid return unit")
        if self.mode == "scalar" and len(self.units) != 1:
            raise ValueError("scalar batch must contain exactly one unit")


@dataclass(frozen=True)
class InvocationFailure:
    error: NodeError

    def __post_init__(self) -> None:
        if type(self.error) is not NodeError:
            raise ValueError("failure must contain a NodeError")
        for name, value in (
            ("node_id", self.error.node_id),
            ("node_type", self.error.node_type),
            ("message", self.error.message),
            ("traceback", self.error.traceback),
        ):
            _text(value, f"error {name}")
        if type(self.error.hints) is not tuple or len(self.error.hints) > RESULT_ALGEBRA_MAX_COUNT:
            raise ValueError("error hints must be a bounded immutable tuple")
        for hint in self.error.hints:
            if type(hint) is not ErrorHint:
                raise ValueError("error hints must contain exact ErrorHint values")
            _text(hint.code, "error hint code")
            _text(hint.message, "error hint message")
            if hint.suggestion is not None:
                _text(hint.suggestion, "error hint suggestion")


@dataclass(frozen=True)
class InvocationReturn:
    batch: ReturnBatch

    def __post_init__(self) -> None:
        if not isinstance(self.batch, ReturnBatch):
            raise ValueError("return must contain a ReturnBatch")


InvocationOutcome: TypeAlias = InvocationFailure | InvocationReturn


def negotiate_result_capabilities(requested: object) -> frozenset[str]:
    """Strictly parse an optional hello list and return the supported intersection."""
    if requested is None:
        return frozenset()
    if not isinstance(requested, (list, tuple, set, frozenset)):
        raise ValueError("result capabilities must be an array or set")
    items = tuple(requested)
    if not all(type(item) is str for item in items) or len(items) != len(set(items)):
        raise ValueError("result capabilities must be unique strings")
    return frozenset({RESULT_ALGEBRA_CAPABILITY}.intersection(items))
