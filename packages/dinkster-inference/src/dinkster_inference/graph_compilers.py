"""Worker-local deterministic graph compiler execution."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import NoReturn, TypeAlias, cast

from dinkster_protocol import (
    GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
    GRAPH_COMPILE_ERROR_GENERATED_LIMIT,
    GRAPH_COMPILE_ERROR_ID_COLLISION,
    GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
    GRAPH_COMPILE_ERROR_NODE_LIMIT,
    GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
    GRAPH_COMPILE_ERROR_PASS_LIMIT,
    GRAPH_COMPILE_MAX_GENERATED_PER_PASS,
    GRAPH_COMPILE_MAX_NODES,
    GRAPH_COMPILE_MAX_PASSES,
    BehaviorValue,
    canonical_compile_reply_bytes,
    generated_node_id,
)

CompilerConfig: TypeAlias = tuple[tuple[str, BehaviorValue], ...]


class InferenceGraphCompileError(Exception):
    """A worker refusal carrying the transport's nonempty error name."""

    def __init__(self, error_name: str, message: str) -> None:
        if not error_name:
            raise ValueError("error_name must be non-empty")
        self.error_name = error_name
        super().__init__(message)


class _HostCompileRefusal(InferenceGraphCompileError):
    pass


def _refuse(error_name: str, message: str) -> NoReturn:
    raise _HostCompileRefusal(error_name, message)


def _frozen_json(value: object) -> object:
    if isinstance(value, Mapping):
        raw = cast("Mapping[object, object]", value)
        if not all(isinstance(key, str) for key in raw):
            _refuse(GRAPH_COMPILE_ERROR_MALFORMED_REPLY, "graph values require string keys")
        return MappingProxyType({cast(str, key): _frozen_json(item) for key, item in raw.items()})
    if isinstance(value, list | tuple):
        return tuple(_frozen_json(item) for item in cast("Sequence[object]", value))
    return value


def _json_copy(value: object, what: str) -> object:
    try:
        encoded = canonical_compile_reply_bytes({"value": value})
        return cast("dict[str, object]", json.loads(encoded))["value"]
    except (RecursionError, TypeError, ValueError) as exc:
        _refuse(GRAPH_COMPILE_ERROR_MALFORMED_REPLY, f"{what} is not canonical JSON: {exc}")


def _mutable_json(value: object) -> object:
    if isinstance(value, Mapping):
        mutable: dict[str, object] = {}
        raw = cast("Mapping[object, object]", value)
        for key in raw:
            assert isinstance(key, str)
            mutable[key] = _mutable_json(raw[key])
        return mutable
    if isinstance(value, tuple):
        return [_mutable_json(item) for item in cast("tuple[object, ...]", value)]
    return value


def _valid_path(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and all(
            segment and not any(char in segment for char in "/[]") for segment in value.split("/")
        )
    )


def _validate_input_wire(inputs: Mapping[str, object]) -> None:
    for input_id, value in inputs.items():
        if not isinstance(value, Mapping):
            continue
        marker = cast("Mapping[object, object]", value)
        reserved = {"$link", "$typed"} & set(marker)
        if not reserved:
            continue
        if len(marker) != 1:
            _refuse(
                GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
                f"generated input {input_id!r} has a malformed marker",
            )
        if "$link" in marker:
            link = marker["$link"]
            if (
                not isinstance(link, Mapping)
                or set(cast("Mapping[object, object]", link)) != {"node", "output"}
                or not all(
                    isinstance(cast("Mapping[object, object]", link)[key], str)
                    and cast("Mapping[object, object]", link)[key]
                    for key in ("node", "output")
                )
            ):
                _refuse(
                    GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
                    f"generated input {input_id!r} has a malformed link",
                )
        else:
            typed = marker["$typed"]
            if (
                not isinstance(typed, Mapping)
                or set(cast("Mapping[object, object]", typed)) != {"type", "value"}
                or not isinstance(cast("Mapping[object, object]", typed)["type"], str)
                or not cast("Mapping[object, object]", typed)["type"]
            ):
                _refuse(
                    GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
                    f"generated input {input_id!r} has a malformed typed literal",
                )


def _graph_nodes(graph: object) -> dict[str, object]:
    if not isinstance(graph, Mapping):
        _refuse(GRAPH_COMPILE_ERROR_MALFORMED_REPLY, "graph must be an object")
    raw = cast("Mapping[object, object]", graph)
    if set(raw) != {"nodes"} or not isinstance(raw.get("nodes"), Mapping):
        _refuse(GRAPH_COMPILE_ERROR_MALFORMED_REPLY, "graph must contain only a nodes object")
    nodes = cast("Mapping[object, object]", raw["nodes"])
    if not all(isinstance(key, str) for key in nodes):
        _refuse(GRAPH_COMPILE_ERROR_MALFORMED_REPLY, "graph node ids must be strings")
    return cast("dict[str, object]", nodes)


def _paths(graph: object) -> set[str]:
    paths: set[str] = set()
    pending: list[tuple[object, str]] = [(graph, "")]
    while pending:
        current, prefix = pending.pop()
        nodes = _graph_nodes(current)
        for node_id, raw_entry in nodes.items():
            if not _valid_path(node_id):
                _refuse(GRAPH_COMPILE_ERROR_MALFORMED_REPLY, f"invalid node id {node_id!r}")
            path = f"{prefix}/{node_id}" if prefix else node_id
            paths.add(path)
            if not isinstance(raw_entry, Mapping):
                _refuse(GRAPH_COMPILE_ERROR_MALFORMED_REPLY, f"node {path!r} must be an object")
            entry = cast("Mapping[object, object]", raw_entry)
            region = entry.get("region")
            if region is not None:
                if set(entry) != {"region"} or not isinstance(region, Mapping):
                    _refuse(GRAPH_COMPILE_ERROR_MALFORMED_REPLY, f"region {path!r} is malformed")
                body = cast("Mapping[object, object]", region).get("body")
                pending.append((body, path))
            elif not isinstance(entry.get("nodeType"), str) or not entry.get("nodeType"):
                _refuse(GRAPH_COMPILE_ERROR_MALFORMED_REPLY, f"node {path!r} has no nodeType")
    return paths


@dataclass(frozen=True)
class GeneratedNodeSpec:
    """One root node addition with deterministic provenance."""

    local_key: str
    sources: tuple[str, ...]
    node_type: str
    inputs: Mapping[str, object]

    def __post_init__(self) -> None:
        local_key = cast("object", self.local_key)
        sources = cast("object", self.sources)
        node_type = cast("object", self.node_type)
        inputs = cast("object", self.inputs)
        if not isinstance(local_key, str) or not local_key:
            raise ValueError("local_key must be non-empty")
        if (
            not isinstance(sources, tuple)
            or not sources
            or not all(_valid_path(source) for source in cast("tuple[object, ...]", sources))
        ):
            raise ValueError("sources must be an ordered non-empty tuple of full paths")
        if not isinstance(node_type, str) or not node_type:
            raise ValueError("node_type must be non-empty")
        if not isinstance(inputs, Mapping) or not all(
            isinstance(key, str) and key for key in cast("Mapping[object, object]", inputs)
        ):
            raise ValueError("inputs must be a string-keyed mapping")
        copied = _json_copy(dict(self.inputs), "generated inputs")
        copied_inputs = cast("dict[str, object]", copied)
        _validate_input_wire(copied_inputs)
        object.__setattr__(
            self, "inputs", cast("Mapping[str, object]", _frozen_json(copied_inputs))
        )


@dataclass(frozen=True)
class InputRewrite:
    """Rewire one existing plain root input to a canonical graph link."""

    node_id: str
    input_id: str
    source_node: str
    source_output: str

    def __post_init__(self) -> None:
        values = cast(
            "tuple[object, ...]",
            (self.node_id, self.input_id, self.source_node, self.source_output),
        )
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError("rewrite fields must be non-empty strings")
        if not _valid_path(self.node_id) or "/" in self.node_id:
            raise ValueError("rewrite node_id must name a plain root node")
        if not _valid_path(self.source_node):
            raise ValueError("rewrite source_node must be a full node path")


@dataclass(frozen=True)
class CompilerEmission:
    """One additive and input-rewire compiler result."""

    generated: tuple[GeneratedNodeSpec, ...] = ()
    rewrites: tuple[InputRewrite, ...] = ()

    def __post_init__(self) -> None:
        generated = cast("object", self.generated)
        rewrites = cast("object", self.rewrites)
        if not isinstance(generated, tuple) or not all(
            isinstance(item, GeneratedNodeSpec) for item in cast("tuple[object, ...]", generated)
        ):
            raise TypeError("generated must contain GeneratedNodeSpec values")
        if not isinstance(rewrites, tuple) or not all(
            isinstance(item, InputRewrite) for item in cast("tuple[object, ...]", rewrites)
        ):
            raise TypeError("rewrites must contain InputRewrite values")


@dataclass
class _AttemptState:
    count: int = 0
    refusal: _HostCompileRefusal | None = None


@dataclass(frozen=True)
class GraphCompileView:
    """Read-only root graph view supplied to one compiler callback."""

    graph: Mapping[str, object]
    targets: tuple[str, ...]
    pass_index: int
    compiler_id: str
    _cancelled: Callable[[], bool] = field(repr=False, compare=False)
    _attempts: _AttemptState = field(repr=False, compare=False)
    _issued: list[GeneratedNodeSpec] = field(repr=False, compare=False)

    def generated_id(self, sources: Sequence[str], local_key: str) -> str:
        return generated_node_id(self.compiler_id, sources, local_key)

    def check_cancelled(self) -> None:
        """Cooperatively stop long callback preprocessing."""
        _poll(self._cancelled)

    def attempt_generated_node(
        self,
        local_key: str,
        sources: tuple[str, ...],
        node_type: str,
        inputs: Mapping[str, object],
    ) -> GeneratedNodeSpec:
        """Count and validate one generated-node attempt before construction."""
        self.check_cancelled()
        if self._attempts.refusal is not None:
            raise self._attempts.refusal
        if self._attempts.count >= GRAPH_COMPILE_MAX_GENERATED_PER_PASS:
            refusal = _HostCompileRefusal(
                GRAPH_COMPILE_ERROR_GENERATED_LIMIT,
                "generated-node attempt limit exceeded",
            )
            self._attempts.refusal = refusal
            raise refusal
        self._attempts.count += 1
        try:
            spec = GeneratedNodeSpec(
                local_key,
                sources,
                node_type,
                inputs,
            )
        except _HostCompileRefusal as exc:
            self._attempts.refusal = exc
            raise
        self._issued.append(spec)
        return spec


GraphCompilerCallback: TypeAlias = Callable[[GraphCompileView], CompilerEmission]


@dataclass(frozen=True)
class GraphCompilerDescriptor:
    """One worker-local graph compiler and its RPC-clean identity facts."""

    id: str
    order: int
    compile: GraphCompilerCallback
    config: CompilerConfig = ()

    def __post_init__(self) -> None:
        compiler_id = cast("object", self.id)
        order = cast("object", self.order)
        config = cast("object", self.config)
        if (
            not isinstance(compiler_id, str)
            or "." not in compiler_id
            or compiler_id.startswith(".")
            or compiler_id.endswith(".")
        ):
            raise ValueError("graph compiler id must be namespace-qualified")
        if type(order) is not int or not -(2**31) <= order < 2**31:
            raise ValueError("graph compiler order must be a signed 32-bit integer")
        if not callable(self.compile):
            raise TypeError("graph compiler compile must be callable")
        if not isinstance(config, tuple):
            raise ValueError("config must be sorted unique RPC-clean key/value pairs")
        normalized: list[tuple[str, BehaviorValue]] = []
        for raw_item in cast("tuple[object, ...]", config):
            if not isinstance(raw_item, tuple):
                raise ValueError("config must be sorted unique RPC-clean key/value pairs")
            parts = cast("tuple[object, ...]", raw_item)
            if len(parts) != 2:
                raise ValueError("config must be sorted unique RPC-clean key/value pairs")
            key, value = parts
            if (
                not isinstance(key, str)
                or not key
                or key.startswith("config.")
                or value is not None
                and type(value) not in (str, int, bool)
            ):
                raise ValueError("config must be sorted unique RPC-clean key/value pairs")
            normalized.append((key, cast("BehaviorValue", value)))
        if tuple(sorted(normalized)) != tuple(normalized) or len(
            {key for key, _ in normalized}
        ) != len(normalized):
            raise ValueError("config must be sorted unique RPC-clean key/value pairs")


def graph_compiler_declaration_metadata(
    descriptor: GraphCompilerDescriptor,
) -> tuple[tuple[str, BehaviorValue], ...]:
    return tuple(
        sorted(
            (
                ("contractVersion", 1),
                ("order", descriptor.order),
                *((f"config.{key}", value) for key, value in descriptor.config),
            )
        )
    )


def _poll(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise asyncio.CancelledError


def _callback(
    descriptor: GraphCompilerDescriptor,
    view: GraphCompileView,
) -> CompilerEmission:
    try:
        emission = cast("object", descriptor.compile(view))
    except _HostCompileRefusal:
        raise
    except asyncio.CancelledError as exc:
        if view._cancelled():  # pyright: ignore[reportPrivateUsage]
            raise
        raise InferenceGraphCompileError(
            GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
            f"graph compiler {descriptor.id!r} raised CancelledError without host cancellation",
        ) from exc
    except Exception as exc:
        raise InferenceGraphCompileError(
            GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
            f"graph compiler {descriptor.id!r} failed with {type(exc).__name__}: {exc}",
        ) from exc
    if view._attempts.refusal is not None:  # pyright: ignore[reportPrivateUsage]
        raise view._attempts.refusal  # pyright: ignore[reportPrivateUsage]
    if not isinstance(emission, CompilerEmission):
        _refuse(
            GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
            f"graph compiler {descriptor.id!r} returned a malformed emission",
        )
    if any(
        not any(
            spec is issued
            for issued in view._issued  # pyright: ignore[reportPrivateUsage]
        )
        for spec in emission.generated
    ):
        _refuse(
            GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
            f"graph compiler {descriptor.id!r} returned an unissued generated node",
        )
    return emission


def execute_graph_compilers(
    generation_key: str,
    graph_wire: Mapping[str, object],
    targets: Sequence[str],
    compilers: Sequence[GraphCompilerDescriptor],
    *,
    cancelled: Callable[[], bool],
) -> Mapping[str, object]:
    """Execute a captured immutable compiler tuple to an idempotent fixpoint."""
    ordered = tuple(sorted(compilers, key=lambda item: (item.order, item.id)))
    if not ordered:
        _poll(cancelled)
        return {
            "generationKey": generation_key,
            "graph": graph_wire,
            "targets": list(targets),
            "passCount": 0,
            "attemptedGeneratedCounts": [],
            "origins": [],
        }
    copied = _json_copy(graph_wire, "graph")
    graph = cast("dict[str, object]", copied)
    nodes = _graph_nodes(graph)
    paths = _paths(graph)
    if len(paths) > GRAPH_COMPILE_MAX_NODES:
        _refuse(GRAPH_COMPILE_ERROR_NODE_LIMIT, f"compiled graph has {len(paths)} nodes")
    target_values = tuple(targets)
    raw_targets = cast("tuple[object, ...]", target_values)
    if not all(isinstance(target, str) for target in raw_targets):
        _refuse(GRAPH_COMPILE_ERROR_MALFORMED_REPLY, "targets must contain strings")
    attempted_counts: list[int] = []
    origins: dict[str, dict[str, object]] = {}
    pass_index = 0
    while True:
        changed = False
        attempts = _AttemptState()
        for descriptor in ordered:
            _poll(cancelled)
            view = GraphCompileView(
                cast("Mapping[str, object]", _frozen_json(graph)),
                target_values,
                pass_index,
                descriptor.id,
                cancelled,
                attempts,
                [],
            )
            emission = _callback(descriptor, view)
            _poll(cancelled)
            emitted_ids: set[str] = set()
            for spec in emission.generated:
                _poll(cancelled)
                node_id = generated_node_id(descriptor.id, spec.sources, spec.local_key)
                if node_id in emitted_ids or node_id in nodes:
                    _refuse(GRAPH_COMPILE_ERROR_ID_COLLISION, f"generated id {node_id!r} collides")
                if node_id in spec.sources:
                    _refuse(
                        GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
                        f"generated node {node_id!r} names itself as a source",
                    )
                missing = tuple(source for source in spec.sources if source not in paths)
                if missing:
                    _refuse(
                        GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
                        f"generated node {node_id!r} names missing source {missing[0]!r}",
                    )
                emitted_ids.add(node_id)
                nodes[node_id] = {
                    "nodeType": spec.node_type,
                    "inputs": _mutable_json(spec.inputs),
                }
                paths.add(node_id)
                if len(paths) > GRAPH_COMPILE_MAX_NODES:
                    _refuse(GRAPH_COMPILE_ERROR_NODE_LIMIT, "compiled graph exceeds node limit")
                origins[node_id] = {
                    "nodeId": node_id,
                    "compilerId": descriptor.id,
                    "passIndex": pass_index,
                    "sources": list(spec.sources),
                    "localKey": spec.local_key,
                }
                changed = True
            rewritten: set[tuple[str, str]] = set()
            for rewrite in emission.rewrites:
                _poll(cancelled)
                destination = (rewrite.node_id, rewrite.input_id)
                if destination in rewritten:
                    _refuse(
                        GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
                        f"input {rewrite.node_id!r}.{rewrite.input_id!r} is rewritten twice",
                    )
                rewritten.add(destination)
                raw_node = nodes.get(rewrite.node_id)
                if not isinstance(raw_node, dict) or "region" in raw_node:
                    _refuse(
                        GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
                        f"rewrite destination {rewrite.node_id!r} is not a plain root node",
                    )
                plain_node = cast("dict[str, object]", raw_node)
                raw_inputs = plain_node.get("inputs")
                if not isinstance(raw_inputs, dict) or rewrite.input_id not in raw_inputs:
                    _refuse(
                        GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
                        f"rewrite input {rewrite.node_id!r}.{rewrite.input_id!r} does not exist",
                    )
                if rewrite.source_node not in paths:
                    _refuse(
                        GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
                        f"rewrite source {rewrite.source_node!r} does not exist",
                    )
                link = {"$link": {"node": rewrite.source_node, "output": rewrite.source_output}}
                if raw_inputs[rewrite.input_id] != link:
                    raw_inputs[rewrite.input_id] = link
                    changed = True
        if not changed:
            _poll(cancelled)
            reply: dict[str, object] = {
                "generationKey": generation_key,
                "graph": graph,
                "targets": list(target_values),
                "passCount": len(attempted_counts),
                "attemptedGeneratedCounts": attempted_counts,
                "origins": [origins[node_id] for node_id in sorted(origins)],
            }
            canonical_compile_reply_bytes(reply)
            _poll(cancelled)
            return reply
        if pass_index >= GRAPH_COMPILE_MAX_PASSES:
            _refuse(GRAPH_COMPILE_ERROR_PASS_LIMIT, "compile pass limit exceeded")
        _poll(cancelled)
        attempted_counts.append(attempts.count)
        pass_index += 1


__all__ = [
    "CompilerEmission",
    "GeneratedNodeSpec",
    "GraphCompileView",
    "GraphCompilerCallback",
    "GraphCompilerDescriptor",
    "InferenceGraphCompileError",
    "InputRewrite",
    "execute_graph_compilers",
    "graph_compiler_declaration_metadata",
]
