"""Parent-owned graph compilation transport and validation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Container, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import NoReturn, TypeGuard, cast

from dinkster_graph import (
    NODE_ID_FORBIDDEN_CHARS,
    Diagnostic,
    Graph,
    Link,
    RegionNode,
    elaborate_graph,
    graph_from_wire,
    graph_to_wire,
    migrate_pure_node_type_replacements,
    snapshot_graph,
    validate,
)
from dinkster_protocol import (
    GENERATED_NODE_ID_PREFIX,
    GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
    GRAPH_COMPILE_ERROR_DEPTH_LIMIT,
    GRAPH_COMPILE_ERROR_GENERATED_LIMIT,
    GRAPH_COMPILE_ERROR_GENERATION_MISMATCH,
    GRAPH_COMPILE_ERROR_ID_COLLISION,
    GRAPH_COMPILE_ERROR_ID_FORMAT,
    GRAPH_COMPILE_ERROR_LINK_LIMIT,
    GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
    GRAPH_COMPILE_ERROR_NODE_LIMIT,
    GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
    GRAPH_COMPILE_ERROR_PASS_LIMIT,
    GRAPH_COMPILE_ERROR_REPLY_OVERSIZE,
    GRAPH_COMPILE_ERROR_SELECTOR_EMITTED,
    GRAPH_COMPILE_ERROR_SELECTOR_INPUT,
    GRAPH_COMPILE_ERROR_TARGET_MISMATCH,
    GRAPH_COMPILE_ERROR_TIMEOUT,
    GRAPH_COMPILE_ERROR_UNKNOWN_GENERATION,
    GRAPH_COMPILE_MAX_DEPTH,
    GRAPH_COMPILE_MAX_GENERATED_PER_PASS,
    GRAPH_COMPILE_MAX_LINKS,
    GRAPH_COMPILE_MAX_NODES,
    GRAPH_COMPILE_MAX_PASSES,
    GRAPH_COMPILE_MAX_REPLY_BYTES,
    GRAPH_COMPILE_RESULT_TYPE,
    GRAPH_COMPILE_TIMEOUT_SECONDS,
    GraphCompilerRegistrySnapshot,
    canonical_compile_reply_bytes,
    generated_node_id,
    is_extension_snapshot_digest,
)
from dinkster_schema import NodeSchema

GraphCompileTransport = Callable[
    [str, Mapping[str, object], Sequence[str]], Awaitable[Mapping[str, object]]
]
_ORIGIN_FIELDS = ("nodeId", "compilerId", "passIndex", "sources", "localKey")


class GraphCompileError(Exception):
    """A deterministic parent-side graph compilation refusal."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ParentGraphBudget:
    """Caller-selected admission bounds, capped by the compiler hard limits."""

    max_nodes: int = GRAPH_COMPILE_MAX_NODES
    max_edges: int = GRAPH_COMPILE_MAX_LINKS
    max_depth: int = GRAPH_COMPILE_MAX_DEPTH
    max_header_bytes: int = GRAPH_COMPILE_MAX_REPLY_BYTES

    def __post_init__(self) -> None:
        limits = {
            "max_nodes": GRAPH_COMPILE_MAX_NODES,
            "max_edges": GRAPH_COMPILE_MAX_LINKS,
            "max_depth": GRAPH_COMPILE_MAX_DEPTH,
            "max_header_bytes": GRAPH_COMPILE_MAX_REPLY_BYTES,
        }
        for field, hard_limit in limits.items():
            value = getattr(self, field)
            if type(value) is not int or not 0 <= value <= hard_limit:
                raise ValueError(f"{field} must be an integer from 0 through {hard_limit}")


@dataclass(frozen=True)
class GraphAdmissionOrigin:
    """Parent-authored provenance for one generated full node path."""

    node_id: str
    sources: tuple[str, ...]
    local_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))


@dataclass(frozen=True)
class VirtualGraphJoin:
    """One explicit root-level edge from the candidate into a new node."""

    target_node_id: str
    input_id: str
    source_node_id: str
    output_id: str


class GraphAdmissionError(Exception):
    """A typed refusal of a parent-authored graph delta."""

    def __init__(
        self,
        code: str,
        message: str,
        diagnostics: Sequence[Diagnostic] = (),
    ) -> None:
        self.code = code
        self.diagnostics = tuple(diagnostics)
        super().__init__(message)


@dataclass(frozen=True, init=False)
class AdmittedGraph:
    """A complete immutable parent candidate and its generated-node origins."""

    _graph: Graph
    origins: Mapping[str, GraphAdmissionOrigin]

    def __init__(
        self,
        graph: Graph,
        origins: Mapping[str, GraphAdmissionOrigin],
    ) -> None:
        object.__setattr__(self, "_graph", snapshot_graph(graph))
        object.__setattr__(self, "origins", MappingProxyType(dict(origins)))

    @property
    def graph(self) -> Graph:
        """Return a defensive snapshot of the admitted candidate."""
        return snapshot_graph(self._graph)


@dataclass(frozen=True, init=False)
class CompiledGraph:
    _graph: Graph
    targets: tuple[str, ...]
    extension_snapshot_digest: str
    origins: Mapping[str, Mapping[str, object]]

    def __init__(
        self,
        graph: Graph,
        targets: Sequence[str],
        extension_snapshot_digest: str,
        origins: Mapping[str, Mapping[str, object]],
    ) -> None:
        if not is_extension_snapshot_digest(extension_snapshot_digest):
            raise ValueError("extension_snapshot_digest must be a sha256 digest")
        target_values = tuple(cast("Sequence[object]", targets))
        if not all(isinstance(target, str) for target in target_values):
            raise TypeError("targets must be a sequence of strings")
        targets_tuple = cast("tuple[str, ...]", target_values)
        object.__setattr__(self, "_graph", snapshot_graph(graph))
        object.__setattr__(self, "targets", targets_tuple)
        object.__setattr__(self, "extension_snapshot_digest", extension_snapshot_digest)
        frozen: dict[str, Mapping[str, object]] = {}
        for node_id, origin in origins.items():
            normalized = dict(origin)
            sources = normalized.get("sources")
            source_values = (
                cast("Sequence[object]", sources)
                if isinstance(sources, Sequence) and not isinstance(sources, str)
                else ()
            )
            if (
                set(normalized) != set(_ORIGIN_FIELDS)
                or normalized.get("nodeId") != node_id
                or not isinstance(normalized.get("compilerId"), str)
                or not normalized["compilerId"]
                or type(normalized.get("passIndex")) is not int
                or cast(int, normalized["passIndex"]) < 0
                or not isinstance(normalized.get("localKey"), str)
                or not normalized["localKey"]
                or not source_values
                or not all(isinstance(source, str) and source for source in source_values)
            ):
                raise ValueError("origins must contain canonical immutable origin records")
            normalized["sources"] = tuple(cast("Sequence[str]", source_values))
            frozen[node_id] = MappingProxyType(normalized)
        object.__setattr__(self, "origins", MappingProxyType(frozen))

    @property
    def graph(self) -> Graph:
        """Return a defensive snapshot so the artifact cannot be mutated."""
        return snapshot_graph(self._graph)


_KNOWN_REMOTE_ERRORS = frozenset(
    {
        GRAPH_COMPILE_ERROR_UNKNOWN_GENERATION,
        GRAPH_COMPILE_ERROR_TIMEOUT,
        GRAPH_COMPILE_ERROR_REPLY_OVERSIZE,
        GRAPH_COMPILE_ERROR_MALFORMED_REPLY,
        GRAPH_COMPILE_ERROR_NODE_LIMIT,
        GRAPH_COMPILE_ERROR_LINK_LIMIT,
        GRAPH_COMPILE_ERROR_PASS_LIMIT,
        GRAPH_COMPILE_ERROR_DEPTH_LIMIT,
        GRAPH_COMPILE_ERROR_GENERATED_LIMIT,
        GRAPH_COMPILE_ERROR_TARGET_MISMATCH,
        GRAPH_COMPILE_ERROR_SELECTOR_EMITTED,
        GRAPH_COMPILE_ERROR_SELECTOR_INPUT,
        GRAPH_COMPILE_ERROR_ID_COLLISION,
        GRAPH_COMPILE_ERROR_ID_FORMAT,
        GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
        GRAPH_COMPILE_ERROR_GENERATION_MISMATCH,
        GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
    }
)


def _refuse(code: str, message: str) -> NoReturn:
    raise GraphCompileError(code, message)


def _paths_and_counts(graph: Graph) -> tuple[set[str], int, int, int]:
    paths: set[str] = set()
    node_count = 0
    link_count = 0
    max_depth = 0
    pending = [(graph, "", 0)]
    while pending:
        current, prefix, depth = pending.pop()
        for node_id, node in current.nodes.items():
            path = f"{prefix}/{node_id}" if prefix else node_id
            paths.add(path)
            node_count += 1
            max_depth = max(max_depth, depth)
            link_count += sum(isinstance(value, Link) for value in node.inputs.values())
            if isinstance(node, RegionNode):
                link_count += len(node.outputs)
                link_count += node.continue_source is not None
                pending.append((node.body, path, depth + 1))
    return paths, node_count, link_count, max_depth


def _check_limits(graph: Graph) -> set[str]:
    paths, nodes, links, depth = _paths_and_counts(graph)
    if nodes > GRAPH_COMPILE_MAX_NODES:
        _refuse(GRAPH_COMPILE_ERROR_NODE_LIMIT, f"compiled graph has {nodes} nodes")
    if links > GRAPH_COMPILE_MAX_LINKS:
        _refuse(GRAPH_COMPILE_ERROR_LINK_LIMIT, f"compiled graph has {links} links")
    if depth > GRAPH_COMPILE_MAX_DEPTH:
        _refuse(GRAPH_COMPILE_ERROR_DEPTH_LIMIT, f"compiled graph reaches depth {depth}")
    return paths


def _graph_exceeds_depth(graph: Graph) -> bool:
    pending = [(graph, 0)]
    while pending:
        current, depth = pending.pop()
        for node in current.nodes.values():
            if not isinstance(node, RegionNode):
                continue
            child_depth = depth + 1
            if child_depth > GRAPH_COMPILE_MAX_DEPTH:
                return True
            pending.append((node.body, child_depth))
    return False


def _wire_exceeds_limit(wire: object, max_depth: int) -> bool:
    pending = [(wire, 0)]
    seen: set[int] = set()
    while pending:
        current, depth = pending.pop()
        if not isinstance(current, Mapping):
            continue
        identity = id(cast(object, current))
        if identity in seen:
            continue
        seen.add(identity)
        current_mapping = cast("Mapping[object, object]", current)
        nodes = current_mapping.get("nodes")
        if not isinstance(nodes, Mapping):
            continue
        for entry in cast("Mapping[object, object]", nodes).values():
            if not isinstance(entry, Mapping):
                continue
            region = cast("Mapping[object, object]", entry).get("region")
            if not isinstance(region, Mapping):
                continue
            child_depth = depth + 1
            if child_depth > max_depth:
                return True
            pending.append((cast("Mapping[object, object]", region).get("body"), child_depth))
    return False


def _admission_refuse(
    code: str,
    message: str,
    diagnostics: Sequence[Diagnostic] = (),
) -> NoReturn:
    raise GraphAdmissionError(code, message, diagnostics)


def _nonempty_string(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value)


def _admission_header_bytes(
    delta: object,
    namespace: str,
    origins: Sequence[GraphAdmissionOrigin],
    joins: Sequence[VirtualGraphJoin],
) -> bytes:
    header = {
        "delta": delta,
        "namespace": namespace,
        "origins": [
            {
                "nodeId": origin.node_id,
                "sources": list(origin.sources),
                "localKey": origin.local_key,
            }
            for origin in origins
        ],
        "virtualJoins": [
            {
                "targetNodeId": join.target_node_id,
                "inputId": join.input_id,
                "sourceNodeId": join.source_node_id,
                "outputId": join.output_id,
            }
            for join in joins
        ],
    }
    try:
        return json.dumps(
            header,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except RecursionError as exc:
        _admission_refuse("malformed-delta", f"admission header recursion is malformed: {exc}")
    except (TypeError, ValueError) as exc:
        _admission_refuse("malformed-delta", f"admission header is not canonical JSON: {exc}")


def _with_virtual_joins(
    delta: Graph,
    parent: Graph,
    joins: Sequence[VirtualGraphJoin],
) -> Graph:
    nodes = dict(delta.nodes)
    available_sources = set(parent.nodes) | set(nodes)
    joined_inputs: set[tuple[str, str]] = set()
    for join in joins:
        values = (
            join.target_node_id,
            join.input_id,
            join.source_node_id,
            join.output_id,
        )
        if not all(_nonempty_string(value) for value in values):
            _admission_refuse("virtual-join", "virtual join fields must be non-empty strings")
        target = nodes.get(join.target_node_id)
        if target is None:
            _admission_refuse(
                "virtual-join",
                f"virtual join target {join.target_node_id!r} is not a new root node",
            )
        if join.source_node_id not in available_sources:
            _admission_refuse(
                "virtual-join",
                f"virtual join source {join.source_node_id!r} is outside the candidate",
            )
        key = (join.target_node_id, join.input_id)
        if key in joined_inputs or join.input_id in target.inputs:
            _admission_refuse(
                "virtual-join",
                f"virtual join collides with input {join.target_node_id!r}/{join.input_id!r}",
            )
        joined_inputs.add(key)
        inputs = dict(target.inputs)
        inputs[join.input_id] = Link(join.source_node_id, join.output_id)
        nodes[join.target_node_id] = replace(target, inputs=inputs)
    return Graph(nodes)


def admit_parent_graph(
    parent: Graph,
    delta: Mapping[str, object],
    *,
    namespace: str,
    origins: Sequence[GraphAdmissionOrigin],
    virtual_joins: Sequence[VirtualGraphJoin],
    budget: ParentGraphBudget,
    schemas: Mapping[str, NodeSchema],
    known_types: Container[str] | None = None,
) -> AdmittedGraph:
    """Admit a bounded parent-authored wire delta without mutating its parent.

    Delta links are confined to the delta. Every edge crossing from the
    accepted parent into a new root node must be declared as a virtual join.
    The function accepts wire data only, constructs a complete candidate, and
    reuses graph elaboration and validation before returning an immutable
    artifact.
    """
    if (
        not _nonempty_string(namespace)
        or "." not in namespace
        or namespace.startswith(".")
        or namespace.endswith(".")
    ):
        _admission_refuse("id-format", "admission namespace must be namespace-qualified")
    parent_snapshot = snapshot_graph(parent)
    origin_values = tuple(origins)
    join_values = tuple(virtual_joins)
    try:
        header = _admission_header_bytes(delta, namespace, origin_values, join_values)
    except GraphAdmissionError as exc:
        if exc.code == "malformed-delta" and _wire_exceeds_limit(delta, budget.max_depth):
            _admission_refuse("recursion-budget", "delta exceeds its recursion budget")
        raise
    if len(header) > budget.max_header_bytes:
        _admission_refuse(
            "header-budget",
            f"admission header has {len(header)} bytes, budget is {budget.max_header_bytes}",
        )
    if _wire_exceeds_limit(delta, budget.max_depth):
        _admission_refuse("recursion-budget", "delta exceeds its recursion budget")
    try:
        parsed_delta = graph_from_wire(delta)
    except RecursionError as exc:
        _admission_refuse("recursion-budget", f"delta recursion is malformed: {exc}")
    except (KeyError, TypeError, ValueError) as exc:
        _admission_refuse("malformed-delta", f"delta graph wire is malformed: {exc}")

    parent_paths, _, _, _ = _paths_and_counts(parent_snapshot)
    delta_paths, _, _, _ = _paths_and_counts(parsed_delta)
    root_collisions = set(parent_snapshot.nodes) & set(parsed_delta.nodes)
    if root_collisions:
        _admission_refuse(
            "id-collision",
            f"delta collides with existing node {sorted(root_collisions)[0]!r}",
        )

    origin_map: dict[str, GraphAdmissionOrigin] = {}
    for origin in origin_values:
        if (
            not _valid_full_path(origin.node_id)
            or not _nonempty_string(origin.local_key)
            or not origin.sources
            or not all(_valid_full_path(source) for source in origin.sources)
        ):
            _admission_refuse("origin-coverage", "origin values are malformed")
        if origin.node_id in origin_map:
            _admission_refuse("id-collision", f"duplicate origin {origin.node_id!r}")
        origin_map[origin.node_id] = origin
    if set(origin_map) != delta_paths:
        _admission_refuse("origin-coverage", "origins do not exactly cover delta nodes")

    delta_root_ids = set(parsed_delta.nodes)
    for node_id, node in parsed_delta.nodes.items():
        for input_id, value in node.inputs.items():
            if isinstance(value, Link) and value.node_id not in delta_root_ids:
                _admission_refuse(
                    "cross-boundary-link",
                    f"delta input {node_id!r}/{input_id!r} crosses its boundary without a join",
                )

    candidate_paths = parent_paths | delta_paths
    generated_leaves: set[str] = set()
    for path in sorted(delta_paths):
        origin = origin_map[path]
        if path in origin.sources or any(
            source not in candidate_paths for source in origin.sources
        ):
            _admission_refuse(
                "origin-coverage",
                f"origin {path!r} names an unverifiable source",
            )
        leaf = path.rsplit("/", 1)[-1]
        try:
            expected = generated_node_id(namespace, origin.sources, origin.local_key)
        except (TypeError, ValueError) as exc:
            _admission_refuse("id-format", f"generated-node identity is malformed: {exc}")
        if leaf != expected:
            _admission_refuse("id-format", f"generated node id {leaf!r} is invalid")
        if leaf in generated_leaves:
            _admission_refuse("id-collision", f"generated leaf {leaf!r} is duplicated")
        generated_leaves.add(leaf)

    joined_delta = _with_virtual_joins(parsed_delta, parent_snapshot, join_values)
    try:
        candidate = snapshot_graph(Graph(nodes={**parent_snapshot.nodes, **joined_delta.nodes}))
    except RecursionError as exc:
        _admission_refuse("malformed-delta", f"delta literal recursion is malformed: {exc}")
    _, nodes, edges, depth = _paths_and_counts(candidate)
    if nodes > budget.max_nodes:
        _admission_refuse(
            "node-budget",
            f"candidate has {nodes} nodes, budget is {budget.max_nodes}",
        )
    if edges > budget.max_edges:
        _admission_refuse(
            "edge-budget",
            f"candidate has {edges} edges, budget is {budget.max_edges}",
        )
    if depth > budget.max_depth:
        _admission_refuse(
            "recursion-budget",
            f"candidate reaches depth {depth}, budget is {budget.max_depth}",
        )

    effective, elaboration_diagnostics = elaborate_graph(candidate, schemas)
    diagnostics = elaboration_diagnostics + validate(
        candidate,
        schemas,
        (),
        effective,
        known_types,
    )
    blocking = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.severity == "error"
        or (
            diagnostic.code == "type-mismatch"
            and diagnostic.node_id is not None
            and diagnostic.node_id in delta_paths
        )
    ]
    if blocking:
        _admission_refuse(
            "graph-validation",
            "admitted candidate failed graph validation",
            blocking,
        )
    return AdmittedGraph(candidate, origin_map)


def _check_selectors(graph: Graph, schemas: Mapping[str, NodeSchema], code: str) -> None:
    def visit(current: Graph) -> None:
        for node in current.nodes.values():
            if isinstance(node, RegionNode):
                visit(node.body)
            else:
                schema = schemas.get(node.node_type)
                if schema is not None and schema.selector is not None:
                    _refuse(code, f"selector node {node.node_type!r} is not executable")

    visit(graph)


def _valid_full_path(path: object) -> bool:
    return (
        isinstance(path, str)
        and bool(path)
        and all(
            segment and not any(char in segment for char in NODE_ID_FORBIDDEN_CHARS)
            for segment in path.split("/")
        )
    )


def _wire_exceeds_depth(wire: object) -> bool:
    """Recognize excessive region depth without recursively parsing the wire."""
    pending = [(wire, 0)]
    while pending:
        current, depth = pending.pop()
        if not isinstance(current, Mapping):
            continue
        current_mapping = cast("Mapping[object, object]", current)
        nodes = current_mapping.get("nodes")
        if not isinstance(nodes, Mapping):
            continue
        for entry in cast("Mapping[object, object]", nodes).values():
            if not isinstance(entry, Mapping):
                continue
            region = cast("Mapping[object, object]", entry).get("region")
            if not isinstance(region, Mapping):
                continue
            child_depth = depth + 1
            if child_depth > GRAPH_COMPILE_MAX_DEPTH:
                return True
            pending.append((cast("Mapping[object, object]", region).get("body"), child_depth))
    return False


def _malformed(message: str) -> NoReturn:
    _refuse(GRAPH_COMPILE_ERROR_MALFORMED_REPLY, message)


def _parse_reply(
    raw: object,
    *,
    generation_key: str,
    requested_graph: Graph,
    requested_targets: tuple[str, ...],
    submitted_paths: set[str],
    registry: GraphCompilerRegistrySnapshot,
    schemas: Mapping[str, NodeSchema],
) -> CompiledGraph:
    # A: JSON mapping and full-envelope canonical size.
    if not isinstance(raw, Mapping):
        _malformed("compile reply must be a mapping with string keys")
    raw_mapping = cast("Mapping[object, object]", raw)
    if not all(isinstance(key, str) for key in raw_mapping):
        _malformed("compile reply must be a mapping with string keys")
    reply = cast("Mapping[str, object]", raw_mapping)
    try:
        canonical = canonical_compile_reply_bytes(reply)
    except RecursionError as exc:
        if _wire_exceeds_depth(reply.get("graph")):
            _refuse(GRAPH_COMPILE_ERROR_DEPTH_LIMIT, "compiled graph exceeds depth limit")
        _malformed(f"compile reply is not canonical JSON: {exc}")
    except (KeyError, TypeError, ValueError) as exc:
        _malformed(f"compile reply is not canonical JSON: {exc}")
    if len(canonical) > GRAPH_COMPILE_MAX_REPLY_BYTES:
        _refuse(GRAPH_COMPILE_ERROR_REPLY_OVERSIZE, "compile reply exceeds 8 MiB")

    # B: framing.
    blobs = reply.get("blobs")
    if (
        reply.get("type") != GRAPH_COMPILE_RESULT_TYPE
        or not isinstance(reply.get("requestId"), str)
        or not reply.get("requestId")
        or blobs != []
        or not isinstance(blobs, list)
    ):
        _malformed("compile reply framing is malformed")

    framing = {"type", "requestId", "blobs"}
    # C: remote error envelope.
    if "errorName" in reply:
        allowed = framing | {"errorName", "error"}
        if set(reply) - allowed or not {"type", "requestId", "errorName", "error"} <= set(reply):
            _malformed("compile error envelope fields are malformed")
        error_name = reply["errorName"]
        remote_message = reply["error"]
        if not isinstance(error_name, str) or not error_name or not isinstance(remote_message, str):
            _malformed("compile error envelope values are malformed")
        if error_name in _KNOWN_REMOTE_ERRORS:
            _refuse(error_name, remote_message)
        _refuse(
            GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
            f"unknown remote compile error {error_name}: {remote_message}",
        )

    # D: exact success envelope.
    semantic = {
        "generationKey",
        "graph",
        "targets",
        "passCount",
        "attemptedGeneratedCounts",
        "origins",
    }
    required = {"type", "requestId"} | semantic
    if not required <= set(reply) or set(reply) - (required | {"blobs"}):
        _malformed("compile success envelope fields are malformed")

    # E-F: generation and exact target identity.
    if reply["generationKey"] != generation_key:
        _refuse(GRAPH_COMPILE_ERROR_GENERATION_MISMATCH, "compile generation changed")
    reply_targets = reply["targets"]
    if not isinstance(reply_targets, list):
        _refuse(GRAPH_COMPILE_ERROR_TARGET_MISMATCH, "compiled targets changed")
    targets_list = cast("list[object]", reply_targets)
    if not all(isinstance(target, str) for target in targets_list) or targets_list != list(
        requested_targets
    ):
        _refuse(GRAPH_COMPILE_ERROR_TARGET_MISMATCH, "compiled targets changed")

    # G: pass and attempted-generation accounting.
    pass_count = reply["passCount"]
    if type(pass_count) is not int or pass_count < 0:
        _malformed("passCount must be a non-negative integer")
    if pass_count > GRAPH_COMPILE_MAX_PASSES:
        _refuse(GRAPH_COMPILE_ERROR_PASS_LIMIT, "compile pass limit exceeded")
    attempted = reply["attemptedGeneratedCounts"]
    if not isinstance(attempted, list):
        _malformed("attemptedGeneratedCounts is malformed")
    attempted_values = cast("list[object]", attempted)
    if len(attempted_values) != pass_count or any(
        type(value) is not int or value < 0 for value in attempted_values
    ):
        _malformed("attemptedGeneratedCounts is malformed")
    attempted_counts = cast("list[int]", attempted_values)
    if any(value > GRAPH_COMPILE_MAX_GENERATED_PER_PASS for value in attempted_counts):
        _refuse(GRAPH_COMPILE_ERROR_GENERATED_LIMIT, "generated-node attempt limit exceeded")

    # H: origin shape, compiler identity, and duplicate ids.
    raw_origins = reply["origins"]
    if not isinstance(raw_origins, list):
        _malformed("origins must be a list")
    origin_values = cast("list[object]", raw_origins)
    compiler_ids = {contribution.id for contribution in registry.contributions}
    origins: dict[str, dict[str, object]] = {}
    for raw_origin in origin_values:
        if not isinstance(raw_origin, Mapping):
            _malformed("origin fields are malformed")
        raw_origin_mapping = cast("Mapping[object, object]", raw_origin)
        if set(raw_origin_mapping) != set(_ORIGIN_FIELDS):
            _malformed("origin fields are malformed")
        origin = cast("Mapping[str, object]", raw_origin_mapping)
        node_id = origin["nodeId"]
        compiler_id = origin["compilerId"]
        local_key = origin["localKey"]
        pass_index = origin["passIndex"]
        sources = origin["sources"]
        source_values = cast("list[object]", sources) if isinstance(sources, list) else None
        if (
            not isinstance(node_id, str)
            or not node_id
            or not _valid_full_path(node_id)
            or not isinstance(compiler_id, str)
            or not compiler_id
            or not isinstance(local_key, str)
            or not local_key
            or type(pass_index) is not int
            or not 0 <= pass_index < pass_count
            or source_values is None
            or not source_values
            or not all(_valid_full_path(source) for source in source_values)
        ):
            _malformed("origin values are malformed")
        sources = cast("list[str]", source_values)
        if compiler_id not in compiler_ids:
            _refuse(GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE, f"unknown compiler {compiler_id!r}")
        if node_id in origins:
            _refuse(GRAPH_COMPILE_ERROR_ID_COLLISION, f"duplicate origin {node_id!r}")
        origins[node_id] = {
            "nodeId": node_id,
            "compilerId": compiler_id,
            "passIndex": pass_index,
            "sources": tuple(sources),
            "localKey": local_key,
        }

    # I: graph parse and root targets.
    try:
        graph = graph_from_wire(reply["graph"])
    except RecursionError as exc:
        if _wire_exceeds_depth(reply["graph"]):
            _refuse(GRAPH_COMPILE_ERROR_DEPTH_LIMIT, "compiled graph exceeds depth limit")
        _malformed(f"compiled graph recursion is malformed: {exc}")
    except (KeyError, TypeError, ValueError) as exc:
        _malformed(f"compiled graph is malformed: {exc}")
    if any(target not in graph.nodes for target in requested_targets):
        _refuse(GRAPH_COMPILE_ERROR_TARGET_MISMATCH, "compiled graph removed a target")

    # J: output structural limits and selectors.
    compiled_paths = _check_limits(graph)
    _check_selectors(graph, schemas, GRAPH_COMPILE_ERROR_SELECTOR_EMITTED)

    # K: exact full-path origin coverage.
    new_paths = compiled_paths - submitted_paths
    origin_paths = set(origins)
    if origin_paths & submitted_paths:
        _refuse(GRAPH_COMPILE_ERROR_ID_COLLISION, "origin collides with a submitted node")
    if origin_paths != new_paths:
        _refuse(GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE, "origins do not exactly cover new nodes")
    verifiable_sources = submitted_paths | compiled_paths
    for path in sorted(origin_paths):
        origin = origins[path]
        sources = cast("tuple[str, ...]", origin["sources"])
        if path in sources or any(source not in verifiable_sources for source in sources):
            _refuse(
                GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE,
                f"origin {path!r} names an unverifiable source",
            )

    # L: strict generated identity and surviving leaf uniqueness.
    for path in sorted(new_paths):
        leaf = path.rsplit("/", 1)[-1]
        origin = origins[path]
        expected = generated_node_id(
            cast(str, origin["compilerId"]),
            cast("tuple[str, ...]", origin["sources"]),
            cast(str, origin["localKey"]),
        )
        if (
            not leaf.startswith(GENERATED_NODE_ID_PREFIX)
            or len(leaf) != len(GENERATED_NODE_ID_PREFIX) + 16
            or any(char not in "0123456789abcdef" for char in leaf[len(GENERATED_NODE_ID_PREFIX) :])
            or leaf != expected
        ):
            _refuse(GRAPH_COMPILE_ERROR_ID_FORMAT, f"generated node id {leaf!r} is invalid")
    generated_leaves: set[str] = set()
    for path in sorted(new_paths):
        leaf = path.rsplit("/", 1)[-1]
        if leaf in generated_leaves:
            _refuse(GRAPH_COMPILE_ERROR_ID_COLLISION, f"generated leaf {leaf!r} is duplicated")
        generated_leaves.add(leaf)

    # M: surviving nodes cannot exceed attempted nodes in their pass.
    surviving = [0] * pass_count
    for origin in origins.values():
        surviving[cast(int, origin["passIndex"])] += 1
    if any(count > attempted_counts[index] for index, count in enumerate(surviving)):
        _refuse(GRAPH_COMPILE_ERROR_GENERATED_LIMIT, "surviving nodes exceed attempted nodes")

    # N: a zero-pass compiler cannot mutate the graph.
    if pass_count == 0 and (origins or graph_to_wire(graph) != graph_to_wire(requested_graph)):
        _refuse(GRAPH_COMPILE_ERROR_ORIGIN_COVERAGE, "zero-pass compile changed the graph")

    return CompiledGraph(graph, requested_targets, generation_key, origins)


async def compile_graph(
    graph: Graph,
    targets: Sequence[str],
    *,
    generation_key: str,
    registry: GraphCompilerRegistrySnapshot,
    transport: GraphCompileTransport | None,
    schemas: Mapping[str, NodeSchema],
) -> CompiledGraph:
    """Snapshot, preflight, transport, and validate one graph compilation."""
    targets_tuple = tuple(targets)
    if registry.contributions and _graph_exceeds_depth(graph):
        _refuse(GRAPH_COMPILE_ERROR_DEPTH_LIMIT, "compiled graph exceeds depth limit")
    graph = migrate_pure_node_type_replacements(graph, schemas)
    graph = snapshot_graph(graph)
    if not registry.contributions:
        return CompiledGraph(graph, targets_tuple, generation_key, {})
    submitted_paths = _check_limits(graph)
    if any(
        path.rsplit("/", 1)[-1].startswith(GENERATED_NODE_ID_PREFIX) for path in submitted_paths
    ):
        _refuse(GRAPH_COMPILE_ERROR_ID_COLLISION, "submitted graph contains a generated id")
    _check_selectors(graph, schemas, GRAPH_COMPILE_ERROR_SELECTOR_INPUT)
    graph_wire = graph_to_wire(graph)
    assert transport is not None
    try:
        raw = await asyncio.wait_for(
            transport(generation_key, graph_wire, targets_tuple),
            GRAPH_COMPILE_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise GraphCompileError(GRAPH_COMPILE_ERROR_TIMEOUT, "graph compilation timed out") from exc
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise GraphCompileError(
            GRAPH_COMPILE_ERROR_COMPILER_FAILURE,
            f"graph compiler failed with {type(exc).__name__}: {exc}",
        ) from exc
    return _parse_reply(
        raw,
        generation_key=generation_key,
        requested_graph=graph,
        requested_targets=targets_tuple,
        submitted_paths=submitted_paths,
        registry=registry,
        schemas=schemas,
    )
