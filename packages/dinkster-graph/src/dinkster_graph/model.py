"""Graph model: pure data, no IO, no execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from dinkster_schema import TypeExpr


@dataclass(frozen=True)
class Link:
    """An input driven by another node's output (real output ids, not indexes)."""

    node_id: str
    output_id: str


@dataclass(frozen=True)
class TypedLiteral:
    """A literal input carrying its own runtime type id.

    Plain literals wrap with the DESTINATION input's declared type, so they
    are legal only on runtime-resolvable (concrete or list-of-concrete)
    inputs. A typed literal stamps the type explicitly, letting a frontend
    inline a value into a wildcard/union/variable input - the widget-tap
    lowering case (joint contract with the frontend, 2026-07-25; ROADMAP
    "Typed-literal graph wire form"). The emission rule lives frontend-side:
    typed literals are sent ONLY where the destination type does not resolve
    to a runtime type id; on concrete inputs plain literals stay canonical
    (validation warns on redundant stamps)."""

    type_id: str
    value: object


@dataclass(frozen=True)
class GraphNode:
    node_type: str
    inputs: Mapping[str, Link | object] = field(default_factory=dict[str, object])
    output_members: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict[str, tuple[str, ...]]
    )
    """Stored output membership per declared output family: family id ->
    ordered member suffixes. Document state, exactly like inputs - output
    membership is document-determined, never runtime-determined (hazard H10)."""
    slot_variants: Mapping[str, str] = field(default_factory=dict[str, str])
    """Stored dynamic-slot choices: slot id -> active variant key. Document
    state (the frontend materializes the choice on connect); elaboration
    turns the active variant into concrete inputs. Never derived from solved
    or runtime types (hazard H10)."""


PORTS_NODE_ID = "$region"
REGION_INDEX_PORT_ID = "index"
REGION_INDEX_PORT_TYPE = TypeExpr.concrete("core.int")
"""Inside a region body, ``$region`` reads declared ports and the implicit
zero-based ``index`` of the immediate region. A declared ``index`` port keeps
its existing meaning and shadows the implicit value for compatibility.
``$region`` is reserved and can never be a real node id."""

NODE_ID_FORBIDDEN_CHARS: frozenset[str] = frozenset({"/", "[", "]"})
"""Characters banned in document node ids at every nesting level.

This closes the runtime iteration-id / diagnostic-path grammar: paths are
``segment ("/" segment)*`` where a segment is ``nodeId`` or
``nodeId "[" decimal-index "]"`` (no whitespace). Because node ids can never
contain '/', '[' or ']', stripping occurrence suffixes and splitting nested
paths back into document node ids is mechanical and unambiguous - the same
closure move as banning '<'/'>' in runtime type names."""


def top_level_node_id(runtime_node_id: str) -> str:
    """Recover the document node id at the root of a runtime node path."""
    return runtime_node_id.partition("/")[0].partition("[")[0]


RegionKind = Literal["map", "fold", "while"]

BindingMode = Literal["zip", "cross", "broadcast"]


@dataclass(frozen=True)
class RegionOutput:
    """One exported region output.

    ``gather``: collect the body output across iterations into ``list<T>``
    (empty list for zero iterations). ``compact``: gather only present body
    outputs, omitting typed-absent iterations. ``flatten``: collect a list-typed
    body output by concatenating its lists in iteration order into one
    ``list<T>`` (empty list for zero iterations). ``state``: the final value of
    a state chain - the key of a state-mode output MUST be a declared state
    port, and its source is also what feeds the port on the next iteration.
    """

    source: Link
    """A body node's output (``Link(body_node_id, output_id)``)."""
    mode: Literal["gather", "compact", "state", "flatten"] = "gather"


@dataclass(frozen=True)
class RegionNode:
    """A repetition boundary in the document (DESIGN 3.13).

    One engine primitive under all kinds: expand ``body`` once per binding
    set, with an optional sequential state chain. The kinds are validation
    profiles over that primitive:

    - ``map``: >=1 element ports, no state ports. Iterations are independent
      and may run in parallel.
    - ``fold``: >=1 element ports, >=1 state ports. Iterations chain
      sequentially through the state ports.
    - ``while``: no element ports, >=1 state ports, a ``continue_source``
      (boolean body output checked after each iteration) and a mandatory
      ``max_iterations`` cap.

    Ports are the region's declared interface: every key of ``inputs`` must
    be declared in ``ports`` with the type the BODY sees. An element port
    declared ``T`` expects ``list<T>`` from the outer graph and binds one
    element per iteration (``zip`` binding: all element lists must have equal
    length; ``cross``: the cartesian product of elements; ``broadcast``:
    shorter lists repeat their final element to match the longest). Inputs
    that are neither element nor state ports broadcast unchanged into every
    iteration.
    State ports take their initial value from ``inputs`` and chain through
    the body output named by the state-mode entry in ``outputs``.
    """

    kind: RegionKind
    body: Graph
    ports: Mapping[str, TypeExpr] = field(default_factory=dict[str, TypeExpr])
    inputs: Mapping[str, Link | object] = field(default_factory=dict[str, object])
    element_ports: tuple[str, ...] = ()
    state_ports: tuple[str, ...] = ()
    outputs: Mapping[str, RegionOutput] = field(default_factory=dict[str, RegionOutput])
    binding: BindingMode = "zip"
    max_iterations: int | None = None
    """Iteration cap. Mandatory for ``while`` (engine-enforced - no infinite
    graphs; reaching the cap with ``continue`` still true is a loud error,
    "run exactly N times" is a fold over a range list). Optional elsewhere:
    a safety bound on binding counts (cross products can explode)."""
    continue_source: Link | None = None
    """while only: the body's boolean output deciding whether to iterate."""


AnyNode = GraphNode | RegionNode


@dataclass(frozen=True)
class Graph:
    nodes: Mapping[str, AnyNode]


def snapshot_graph(graph: Graph) -> Graph:
    """A deep, run-scoped, immutable copy of the document.

    GraphNode is frozen, but its inputs/output_members attributes are
    Mappings the caller may have handed us as live dicts - mutating one
    across an await would let a run's topology drift between validation and
    invocation (TOCTOU). The engine snapshots at run() entry and uses only
    the snapshot for elaboration, validation, planning, resolution, and
    invocation. Member sequences are normalized to tuples; every mapping is
    a fresh copy behind a read-only proxy.

    JSON-shaped literal values are recursively copied, including the value
    inside a TypedLiteral. This keeps execution and its fingerprint inputs
    tied to the same snapshot even if the caller mutates the document.
    """

    def copy_literal(value: Link | object) -> Link | object:
        if isinstance(value, Link):
            return value
        if isinstance(value, TypedLiteral):
            return TypedLiteral(value.type_id, copy_literal(value.value))
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, list):
            return [copy_literal(item) for item in cast("list[object]", value)]
        if isinstance(value, dict):
            object_value = cast("dict[object, object]", value)
            if not all(isinstance(key, str) for key in object_value):
                raise TypeError("graph literal object keys must be strings")
            return {cast(str, key): copy_literal(item) for key, item in object_value.items()}
        raise TypeError(f"graph literal values must be JSON-shaped, got {type(value).__name__}")

    def copy_node(node: AnyNode) -> AnyNode:
        if isinstance(node, RegionNode):
            return RegionNode(
                kind=node.kind,
                body=snapshot_graph(node.body),
                ports=MappingProxyType(dict(node.ports)),
                inputs=MappingProxyType(
                    {key: copy_literal(value) for key, value in node.inputs.items()}
                ),
                element_ports=tuple(node.element_ports),
                state_ports=tuple(node.state_ports),
                outputs=MappingProxyType(dict(node.outputs)),
                binding=node.binding,
                max_iterations=node.max_iterations,
                continue_source=node.continue_source,
            )
        return GraphNode(
            node_type=node.node_type,
            inputs=MappingProxyType(
                {key: copy_literal(value) for key, value in node.inputs.items()}
            ),
            output_members=MappingProxyType(
                {fam: tuple(members) for fam, members in node.output_members.items()}
            ),
            slot_variants=MappingProxyType(dict(node.slot_variants)),
        )

    return Graph(
        nodes=MappingProxyType({node_id: copy_node(node) for node_id, node in graph.nodes.items()})
    )
