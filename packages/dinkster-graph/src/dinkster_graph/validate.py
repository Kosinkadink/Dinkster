"""Structural validation and type diagnostics.

Shared principle with Dinkster-Frontend: a graph is rejected only on structural
grounds and the scoped core.combo boundary; other type mismatches are
advisory warnings.
"""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from dinkster_schema import (
    AssetCoercionPlan,
    AssetWidget,
    CoercionProviders,
    ElaborationError,
    NodeSchema,
    TypeEquivalenceProvider,
    TypeExpr,
    TypeSolveError,
    WidgetRepresentations,
    bind_type_variables,
    combo_type_mismatch_is_error,
    elaborate,
    plan_asset_coercion,
    plan_type_equivalence,
    resolved_type_expr,
)

from .model import (
    NODE_ID_FORBIDDEN_CHARS,
    PORTS_NODE_ID,
    REGION_INDEX_PORT_ID,
    REGION_INDEX_PORT_TYPE,
    Graph,
    GraphNode,
    Link,
    RegionNode,
    TypedLiteral,
)
from .plan import find_cycle


@dataclass(frozen=True)
class Diagnostic:
    severity: Literal["error", "warning"]
    code: str
    message: str
    node_id: str | None = None
    input_id: str | None = None
    """The consumer-side input (or region port) the diagnostic anchors to,
    when it anchors to one - so frontends map diagnostics onto per-port
    affordances structurally instead of parsing messages."""


def elaborate_graph(
    graph: Graph,
    schemas: Mapping[str, NodeSchema],
) -> tuple[dict[str, NodeSchema], list[Diagnostic]]:
    """One elaboration pass for the whole document (hazard H10: elaborate
    once, before all consumers). The returned map is THE effective interface
    for this document - validation resolves links against it, and the engine
    reuses the same map for cache keys and invocations instead of
    re-elaborating. Nodes with unknown types or invalid stored membership are
    absent from the map and reported as diagnostics.
    """
    effective: dict[str, NodeSchema] = {}
    diags: list[Diagnostic] = []
    for node_id, node in graph.nodes.items():
        if isinstance(node, RegionNode):
            continue  # regions have no schema; their interface is declared
        base_schema = schemas.get(node.node_type)
        if base_schema is None:
            diags.append(
                Diagnostic(
                    "error", "unknown-node-type", f"unknown node type: {node.node_type}", node_id
                )
            )
            continue
        try:
            effective[node_id] = elaborate(
                base_schema,
                node.inputs,
                node.output_members,
                node.slot_variants,
            )
        except ElaborationError as exc:
            diags.append(Diagnostic("error", "elaboration-failed", str(exc), node_id))
    return effective, diags


@dataclass(frozen=True)
class _RegionData:
    """Everything validation derives once per region: the body's elaborated
    interfaces, nested region data, and the region's own output types."""

    body_effective: Mapping[str, NodeSchema]
    body_regions: Mapping[str, _RegionData]
    body_output_types: Mapping[str, Mapping[str, TypeExpr]]
    interface: Mapping[str, TypeExpr | None]
    """Region output id -> its type as seen from outside (``list<T>`` for
    gather, the port type for state). None when unresolvable (a body error
    reported elsewhere)."""


def _region_port_type(ports: Mapping[str, TypeExpr], output_id: str) -> TypeExpr | None:
    declared = ports.get(output_id)
    if declared is not None:
        return declared
    return REGION_INDEX_PORT_TYPE if output_id == REGION_INDEX_PORT_ID else None


def _body_output_type(region: RegionNode, data: _RegionData, source: Link) -> TypeExpr | None:
    """The type of a body output referenced by a region declaration."""
    if source.node_id == PORTS_NODE_ID:
        return _region_port_type(region.ports, source.output_id)
    nested = data.body_regions.get(source.node_id)
    if nested is not None:
        return nested.interface.get(source.output_id)
    resolved = data.body_output_types.get(source.node_id)
    if resolved is not None and source.output_id in resolved:
        return resolved[source.output_id]
    schema = data.body_effective.get(source.node_id)
    if schema is None:
        return None
    spec = schema.output(source.output_id)
    return None if spec is None else spec.type


def _body_output_exists(region: RegionNode, data: _RegionData, source: Link) -> bool:
    if source.node_id == PORTS_NODE_ID:
        return _region_port_type(region.ports, source.output_id) is not None
    nested = data.body_regions.get(source.node_id)
    if nested is not None:
        return source.output_id in nested.interface
    schema = data.body_effective.get(source.node_id)
    if schema is None:
        # Node exists but its type/elaboration failed: reported there, do
        # not pile a second diagnostic on the region declaration.
        return source.node_id in region.body.nodes
    return schema.output(source.output_id) is not None


def _prefixed(diag: Diagnostic, prefix: str) -> Diagnostic:
    if not prefix or diag.node_id is None:
        return diag
    return replace(diag, node_id=prefix + diag.node_id)


def _interface_pass(
    graph: Graph,
    schemas: Mapping[str, NodeSchema],
    prefix: str,
    include_elab: bool,
) -> tuple[dict[str, NodeSchema], dict[str, _RegionData], list[Diagnostic]]:
    """Elaborate one graph level and derive every region's declared interface
    bottom-up, so sibling link checks can resolve region outputs. Body
    diagnostics carry hierarchical node ids (``region/bodyNode``)."""
    effective, elab_diags = elaborate_graph(graph, schemas)
    diags = [_prefixed(d, prefix) for d in elab_diags] if include_elab else []
    regions: dict[str, _RegionData] = {}
    for node_id, node in graph.nodes.items():
        if not isinstance(node, RegionNode):
            continue
        body_eff, body_regions, body_diags = _interface_pass(
            node.body, schemas, f"{prefix}{node_id}/", include_elab=True
        )
        diags.extend(body_diags)
        body_output_types = _resolve_linked_output_types(
            node.body, node.ports, body_eff, body_regions
        )
        data = _RegionData(body_eff, body_regions, body_output_types, {})
        interface: dict[str, TypeExpr | None] = {}
        for out_id, out in node.outputs.items():
            if out.mode == "state":
                interface[out_id] = node.ports.get(out_id)
            else:
                inner = _body_output_type(node, data, out.source)
                if inner is None:
                    interface[out_id] = None
                elif out.mode == "flatten":
                    interface[out_id] = inner
                else:
                    interface[out_id] = TypeExpr.list_of(inner)
        regions[node_id] = replace(data, interface=interface)
    return effective, regions, diags


def _resolve_linked_output_types(
    graph: Graph,
    ports: Mapping[str, TypeExpr],
    effective: Mapping[str, NodeSchema],
    regions: Mapping[str, _RegionData],
) -> dict[str, dict[str, TypeExpr]]:
    """Resolve generic node outputs from linked concrete input declarations."""
    resolved: dict[str, dict[str, TypeExpr]] = {}

    def contains_variable(expr: TypeExpr) -> bool:
        if expr.kind == "variable":
            return True
        return expr.element is not None and contains_variable(expr.element)

    def source_type(link: Link) -> TypeExpr | None:
        if link.node_id == PORTS_NODE_ID:
            return _region_port_type(ports, link.output_id)
        nested = regions.get(link.node_id)
        if nested is not None:
            return nested.interface.get(link.output_id)
        node_outputs = resolved.get(link.node_id)
        if node_outputs is not None and link.output_id in node_outputs:
            return node_outputs[link.output_id]
        schema = effective.get(link.node_id)
        spec = None if schema is None else schema.output(link.output_id)
        return None if spec is None else spec.type

    for _ in range(len(graph.nodes)):
        changed = False
        for node_id, node in graph.nodes.items():
            if not isinstance(node, GraphNode):
                continue
            schema = effective.get(node_id)
            if schema is None:
                continue
            input_types: dict[str, str] = {}
            all_generic_inputs_resolved = True
            for input_id, value in node.inputs.items():
                input_spec = schema.input(input_id)
                if input_spec is None or not contains_variable(input_spec.type):
                    continue
                if not isinstance(value, Link):
                    all_generic_inputs_resolved = False
                    break
                linked_type = source_type(value)
                runtime_type = None if linked_type is None else linked_type.runtime_type_id()
                if runtime_type is None:
                    all_generic_inputs_resolved = False
                    break
                input_types[input_id] = runtime_type
            if not all_generic_inputs_resolved:
                continue
            try:
                bindings = bind_type_variables(schema.inputs, input_types)
            except TypeSolveError:
                continue
            outputs = {
                output.id: concrete
                for output in schema.outputs
                if (concrete := resolved_type_expr(output.type, bindings)) is not None
            }
            if outputs and resolved.get(node_id) != outputs:
                resolved[node_id] = outputs
                changed = True
        if not changed:
            break
    return resolved


def region_interface(
    region: RegionNode, schemas: Mapping[str, NodeSchema]
) -> dict[str, TypeExpr | None]:
    """A region's declared output types as seen from the outer graph:
    ``list<T>`` for gather and compact outputs, the body ``list<T>`` for
    flatten outputs, and the port type for state outputs. None for outputs
    whose body-side type cannot be resolved (a validation error the document
    already carries). The engine uses this for empty collections and absent
    synthesis; validation derives the same map internally, so the two can
    never disagree."""
    _, regions, _ = _interface_pass(Graph(nodes={"$": region}), schemas, "", include_elab=False)
    return dict(regions["$"].interface)


@dataclass(frozen=True)
class _Level:
    """One graph level's resolution context: the top document, or a region
    body (where ``ports`` supplies the reserved ``$region`` producer)."""

    graph: Graph
    schemas: Mapping[str, NodeSchema]
    effective: Mapping[str, NodeSchema]
    regions: Mapping[str, _RegionData]
    ports: Mapping[str, TypeExpr] | None
    prefix: str
    known_types: Container[str] | None


def _check_input_value(
    level: _Level,
    node_id: str,
    input_id: str,
    value: object,
    expected: TypeExpr | None,
    absent_policy: str,
    diags: list[Diagnostic],
    *,
    asset_stamp_required: bool = False,
) -> None:
    """Every per-input check, shared by node inputs and region inputs:
    producer resolution, maybe-absent, cardinality (authoritative), advisory
    type compatibility, and literal shape rules."""
    where = level.prefix + node_id
    if isinstance(value, Link):
        out_type: TypeExpr | None
        out_optional = False
        producer_desc = f"'{value.node_id}'"
        if value.node_id == PORTS_NODE_ID and level.ports is not None:
            out_type = _region_port_type(level.ports, value.output_id)
            if out_type is None:
                diags.append(
                    Diagnostic(
                        "error",
                        "dangling-port",
                        f"input '{input_id}' reads undeclared region port '{value.output_id}'",
                        where,
                        input_id,
                    )
                )
                return
        else:
            producer = level.graph.nodes.get(value.node_id)
            if producer is None:
                diags.append(
                    Diagnostic(
                        "error",
                        "dangling-link",
                        f"input '{input_id}' links to missing node '{value.node_id}'",
                        where,
                        input_id,
                    )
                )
                return
            if isinstance(producer, RegionNode):
                interface = level.regions[value.node_id].interface
                if value.output_id not in interface:
                    diags.append(
                        Diagnostic(
                            "error",
                            "dangling-output",
                            f"input '{input_id}' links to missing output "
                            f"'{value.output_id}' of region '{value.node_id}'",
                            where,
                            input_id,
                        )
                    )
                    return
                out_type = interface[value.output_id]
            else:
                # Links resolve against the producer's ELABORATED outputs, so
                # dynamic members are ordinary link targets and a member the
                # document no longer stores is a document-time dangling-output
                # error - never a runtime surprise (hazard H10).
                producer_effective = level.effective.get(value.node_id)
                if producer_effective is None:
                    return  # producer unknown/failed; already reported there
                out = producer_effective.output(value.output_id)
                if out is None:
                    base_producer = level.schemas[producer.node_type]
                    owner = next(
                        (
                            fam
                            for fam in base_producer.output_families
                            if fam.member_suffix(value.output_id) is not None
                        ),
                        None,
                    )
                    hint = (
                        f" (family '{owner.id}' currently has members "
                        f"{list(producer.output_members.get(owner.id, ()))})"
                        if owner is not None
                        else ""
                    )
                    diags.append(
                        Diagnostic(
                            "error",
                            "dangling-output",
                            f"input '{input_id}' links to missing output "
                            f"'{value.output_id}' of {producer.node_type}{hint}",
                            where,
                            input_id,
                        )
                    )
                    return
                out_type = out.type
                out_optional = out.optional
                producer_desc = producer.node_type
        if expected is None:
            return  # consumer-side declaration problem, reported there
        # Maybe-absent (DESIGN 3.15): an optional output can carry no value
        # at runtime; feeding it into an input that declares on_absent='fail'
        # means this document can fail at runtime by design. Legal (the
        # consumer asked for loudness) but worth surfacing at document time.
        if out_optional and absent_policy == "fail":
            diags.append(
                Diagnostic(
                    "warning",
                    "maybe-absent",
                    f"input '{input_id}' declares on_absent='fail' but is "
                    f"driven by optional output '{value.output_id}' of "
                    f"{producer_desc}, which may produce no value at runtime",
                    where,
                    input_id,
                )
            )
        if out_type is None:
            return  # unresolvable producer type; its own error is reported
        # Asset coercion (typed assets, joint contract 2026-07-26): when the
        # producer's runtime type is asset-shaped and the destination's
        # declared type matches a registered coercion plan (decode / lift /
        # merge), the engine coerces at input resolution - so both the
        # cardinality check and the advisory type check judge the COERCED
        # result, which fits by construction. Missing providers are errors
        # when a registry is at hand (mirrors typed-literal unknown-type).
        out_runtime = out_type.runtime_type_id()
        if out_runtime is not None and not expected.accepts_concrete(out_runtime):
            equivalence = (
                plan_type_equivalence(out_runtime, expected, level.known_types)
                if isinstance(level.known_types, TypeEquivalenceProvider)
                else None
            )
            if equivalence is not None:
                return
            coercion = plan_asset_coercion(out_runtime, expected)
            if coercion is not None:
                _check_coercion_providers(level, where, input_id, out_runtime, coercion, diags)
                return
        # Cardinality is structural, so a shape mismatch is an error
        # (DESIGN 3.13): a list<T> must never silently drive a scalar T
        # socket - that was v1's implicit repeated execution. The
        # diagnostics name the explicit fix.
        out_card = out_type.cardinality()
        in_card = expected.cardinality()
        if out_card != "unknown" and in_card != "unknown" and out_card != in_card:
            if out_card == "list":
                diags.append(
                    Diagnostic(
                        "error",
                        "list-into-scalar",
                        f"input '{input_id}' expects one value but is "
                        f"driven by a list; apply the node per element "
                        "with a Map region, reduce the list with a "
                        "Gather/Fold, or pick one element explicitly",
                        where,
                        input_id,
                    )
                )
            else:
                diags.append(
                    Diagnostic(
                        "error",
                        "scalar-into-list",
                        f"input '{input_id}' expects a list but is "
                        "driven by one value; collect values into a "
                        "list explicitly (a MakeList/Collect node)",
                        where,
                        input_id,
                    )
                )
            return
        # Type check: most mismatches stay advisory. core.combo is the one
        # hard type boundary; coercible asset sources returned above, so
        # reaching here means no accepted/coercible path exists.
        if out_runtime is not None and not expected.accepts_concrete(out_runtime):
            diags.append(
                Diagnostic(
                    ("error" if combo_type_mismatch_is_error(out_runtime, expected) else "warning"),
                    "type-mismatch",
                    f"input '{input_id}' expects {expected.kind}"
                    f"{list(expected.types)} but is driven by {out_runtime}",
                    where,
                    input_id,
                )
            )
        return
    if isinstance(value, TypedLiteral):
        _check_typed_literal(level, node_id, input_id, value, expected, diags)
        return
    if expected is None:
        return
    if asset_stamp_required:
        diags.append(
            Diagnostic(
                "error",
                "unstamped-asset-widget-literal",
                f"input '{input_id}' has an asset widget on a scalar value "
                "type; its picked AssetRef literal must carry an "
                "asset<T> or list<asset<T>> type stamp",
                where,
                input_id,
            )
        )
        return
    # Current limitation: PLAIN literals are allowed only on inputs whose declared
    # type resolves to a runtime type id (concrete, or list of concrete),
    # because the engine wraps them with it. Literals into non-concrete
    # inputs carry their own stamp instead (TypedLiteral, above).
    if expected.runtime_type_id() is None:
        diags.append(
            Diagnostic(
                "error",
                "literal-on-nonconcrete",
                f"input '{input_id}' is {expected.kind}-typed; "
                "plain literals require a concrete-typed input "
                "(use a typed literal to stamp the type explicitly)",
                where,
                input_id,
            )
        )
    elif expected.cardinality() == "list" and not isinstance(value, list):
        diags.append(
            Diagnostic(
                "error",
                "literal-shape",
                f"input '{input_id}' expects a list literal, got {type(value).__name__}",
                where,
                input_id,
            )
        )


def _check_coercion_providers(
    level: _Level,
    where: str,
    input_id: str,
    source: str,
    coercion: AssetCoercionPlan,
    diags: list[Diagnostic],
) -> None:
    """Provider availability for an asset coercion that structurally fits.

    Errors only when ``known_types`` is a real registry that can answer the
    question (TypeRegistry satisfies CoercionProviders); pure document
    tooling without one validates the plan structurally and stays silent,
    exactly like typed-literal atom checks without a registry."""
    if not isinstance(level.known_types, CoercionProviders):
        return
    for need in coercion.missing_providers(level.known_types):
        diags.append(
            Diagnostic(
                "error",
                "asset-coercion-unavailable",
                f"input '{input_id}' accepts {source} only through an asset "
                f"coercion that needs an unregistered provider: {need}",
                where,
                input_id,
            )
        )


def _check_typed_literal(
    level: _Level,
    node_id: str,
    input_id: str,
    typed: TypedLiteral,
    expected: TypeExpr | None,
    diags: list[Diagnostic],
) -> None:
    """A typed literal stamps its own runtime type id, so it is checked like
    a link from a producer of that type (joint contract with the frontend,
    2026-07-25): the stamp's validity and cardinality are authoritative
    (errors), declared-type compatibility stays advisory (warning). A stamp
    on a runtime-resolvable input is redundant - the emission rule says
    frontends send plain literals there - and warns, keeping mixed-version
    behavior loud (a pre-form server treats the marker dict as a plain
    literal, which only passes silently on concrete inputs)."""
    where = level.prefix + node_id
    atom = TypeExpr.runtime_type_atom(typed.type_id)
    if atom is None:
        diags.append(
            Diagnostic(
                "error",
                "typed-literal-malformed",
                f"input '{input_id}' carries a typed literal with malformed "
                f"type id {typed.type_id!r} (expected an atom or "
                "list<...> of an atom)",
                where,
                input_id,
            )
        )
        return
    known = level.known_types is None or atom in level.known_types
    if not known:
        diags.append(
            Diagnostic(
                "error",
                "typed-literal-unknown-type",
                f"input '{input_id}' carries a typed literal stamped with "
                f"unregistered type '{atom}'",
                where,
                input_id,
            )
        )
    # The stamp gives the value a definite shape: a list<...> stamp on a
    # non-list JSON value can never wrap, so it is a document error even
    # before touching the destination type. An asset<...> stamp is one
    # AssetRef - scalar - whatever it decodes to.
    is_list_stamp = TypeExpr.runtime_cardinality(typed.type_id) == "list"
    if is_list_stamp and not isinstance(typed.value, list):
        diags.append(
            Diagnostic(
                "error",
                "literal-shape",
                f"input '{input_id}' carries a typed literal stamped "
                f"{typed.type_id} but its value is "
                f"{type(typed.value).__name__}, not a list",
                where,
                input_id,
            )
        )
    if expected is None:
        return  # consumer-side declaration problem, reported there
    # Asset coercion applies to stamped literals exactly as to links: an
    # asset<T> stamp into a T input coerces at input resolution, so the
    # redundancy warning, the cardinality error, and the advisory type
    # check below all judge the coerced result.
    accepted_as_is = expected.accepts_concrete(typed.type_id)
    equivalence = (
        plan_type_equivalence(typed.type_id, expected, level.known_types)
        if known and isinstance(level.known_types, TypeEquivalenceProvider)
        else None
    )
    coercion = (
        plan_asset_coercion(typed.type_id, expected)
        if known and not accepted_as_is and equivalence is None
        else None
    )
    if expected.runtime_type_id() is not None and accepted_as_is:
        # Redundant only when the plain form would wrap identically; a
        # coercible stamp (asset<T> into T) genuinely carries information.
        diags.append(
            Diagnostic(
                "warning",
                "typed-literal-on-concrete",
                f"input '{input_id}' resolves to a runtime type; the "
                "explicit stamp is redundant (emission rule: plain "
                "literals stay canonical on concrete inputs)",
                where,
                input_id,
            )
        )
    if coercion is not None:
        _check_coercion_providers(level, where, input_id, typed.type_id, coercion, diags)
        return
    if equivalence is not None:
        return
    # Cardinality is structural (DESIGN 3.13), mirroring the link path: the
    # stamp is definite, so a shape mismatch with a definite input is an
    # error, not a type warning.
    stamp_card = "list" if is_list_stamp else "scalar"
    in_card = expected.cardinality()
    if in_card != "unknown" and stamp_card != in_card:
        if stamp_card == "list":
            diags.append(
                Diagnostic(
                    "error",
                    "list-into-scalar",
                    f"input '{input_id}' expects one value but carries a "
                    f"typed literal stamped {typed.type_id}",
                    where,
                    input_id,
                )
            )
        else:
            diags.append(
                Diagnostic(
                    "error",
                    "scalar-into-list",
                    f"input '{input_id}' expects a list but carries a "
                    f"typed literal stamped {typed.type_id}",
                    where,
                    input_id,
                )
            )
        return
    # Type check, exactly as for links (skipped when the stamp is already an
    # error-level unknown). Most mismatches warn; core.combo is hard.
    if known and not expected.accepts_concrete(typed.type_id):
        diags.append(
            Diagnostic(
                ("error" if combo_type_mismatch_is_error(typed.type_id, expected) else "warning"),
                "type-mismatch",
                f"input '{input_id}' expects {expected.kind}"
                f"{list(expected.types)} but carries a typed literal "
                f"stamped {typed.type_id}",
                where,
                input_id,
            )
        )


def validate(
    graph: Graph,
    schemas: Mapping[str, NodeSchema],
    targets: Sequence[str],
    effective: Mapping[str, NodeSchema] | None = None,
    known_types: Container[str] | None = None,
) -> list[Diagnostic]:
    """Structural validation over the elaborated document.

    Pass the map from elaborate_graph() to guarantee validation and execution
    see the same interfaces; when omitted, it is computed here (and its
    diagnostics included). Full-document validity is required: errors
    anywhere in the document fail the run, even on branches the targets do
    not reach - dangling dynamic members are document bugs, not latent state.

    ``known_types`` is the registered type-atom vocabulary (the engine
    passes its registry, which is a Container of atoms). When supplied,
    typed-literal stamps naming an unregistered atom are errors; when
    omitted (pure document tooling with no registry at hand), stamp checks
    stay syntactic.

    Region bodies validate recursively; their diagnostics carry hierarchical
    node ids (``region/bodyNode``, nested as ``outer/inner/node``).
    """
    top_effective, regions, diags = _interface_pass(
        graph, schemas, "", include_elab=effective is None
    )
    if effective is not None:
        top_effective = dict(effective)

    for target in targets:
        if target not in graph.nodes:
            diags.append(
                Diagnostic("error", "unknown-target", f"target node does not exist: {target}")
            )

    level = _Level(
        graph,
        schemas,
        top_effective,
        regions,
        ports=None,
        prefix="",
        known_types=known_types,
    )
    _validate_level(level, diags)
    return diags


def _validate_level(level: _Level, diags: list[Diagnostic]) -> None:
    if PORTS_NODE_ID in level.graph.nodes:
        diags.append(
            Diagnostic(
                "error",
                "reserved-node-id",
                f"'{PORTS_NODE_ID}' is reserved for region port references",
                level.prefix + PORTS_NODE_ID,
            )
        )
    for node_id in level.graph.nodes:
        # The iteration-id/diagnostic-path grammar is closed (DESIGN 3.13):
        # prefixed ids like "r[3]/node" parse back to document node ids
        # mechanically ONLY because '/', '[' and ']' can never appear in a
        # node id, at any nesting level. Same closure move as '<'/'>' in
        # type atoms and the $region reservation.
        if not node_id or any(ch in node_id for ch in NODE_ID_FORBIDDEN_CHARS):
            diags.append(
                Diagnostic(
                    "error",
                    "invalid-node-id",
                    f"node ids must be non-empty and may not contain '/', '[' or ']': {node_id!r}",
                    level.prefix + node_id,
                )
            )
    for node_id, node in level.graph.nodes.items():
        if isinstance(node, RegionNode):
            _validate_region(level, node_id, node, diags)
        else:
            _validate_graph_node(level, node_id, node, diags)
    cycle = find_cycle(level.graph)
    if cycle:
        diags.append(
            Diagnostic(
                "error",
                "cycle",
                "graph contains a cycle: " + " -> ".join(level.prefix + n for n in cycle),
            )
        )


def _validate_region(
    level: _Level, node_id: str, region: RegionNode, diags: list[Diagnostic]
) -> None:
    where = level.prefix + node_id
    data = level.regions[node_id]

    def shape_error(message: str, input_id: str | None = None) -> None:
        diags.append(Diagnostic("error", "region-shape", message, where, input_id))

    if region.kind not in ("map", "fold", "while"):
        diags.append(
            Diagnostic(
                "error",
                "unknown-region-kind",
                f"unknown region kind {region.kind!r}; expected map, fold, or while",
                where,
            )
        )
    if region.binding not in ("zip", "cross", "broadcast"):
        diags.append(
            Diagnostic(
                "error",
                "unknown-binding-mode",
                f"unknown binding mode {region.binding!r}; expected zip, cross, or broadcast",
                where,
            )
        )
    for out_id, out in region.outputs.items():
        if out.mode not in ("gather", "compact", "state", "flatten"):
            diags.append(
                Diagnostic(
                    "error",
                    "unknown-output-mode",
                    f"region output '{out_id}' has unknown mode {out.mode!r}; "
                    "expected gather, compact, state, or flatten",
                    where,
                    out_id,
                )
            )

    for input_id in region.inputs:
        if input_id not in region.ports:
            diags.append(
                Diagnostic(
                    "error",
                    "undeclared-port",
                    f"region input '{input_id}' has no declared port type",
                    where,
                    input_id,
                )
            )
    for port in (*region.element_ports, *region.state_ports):
        if port not in region.ports:
            diags.append(
                Diagnostic(
                    "error",
                    "undeclared-port",
                    f"port '{port}' has no declared port type",
                    where,
                    port,
                )
            )
        if port not in region.inputs:
            diags.append(
                Diagnostic(
                    "error",
                    "missing-input",
                    f"region requires input '{port}'",
                    where,
                    port,
                )
            )
    overlap = set(region.element_ports) & set(region.state_ports)
    for port in sorted(overlap):
        shape_error(f"port '{port}' cannot be both an element and a state port", port)

    if region.kind == "map":
        if not region.element_ports:
            shape_error("map region requires at least one element port")
        if region.state_ports:
            shape_error("map region declares state ports; use a fold region")
        if region.continue_source is not None:
            shape_error("map region declares a continue source; use a while region")
    elif region.kind == "fold":
        if not region.element_ports:
            shape_error("fold region requires at least one element port")
        if not region.state_ports:
            shape_error("fold region requires at least one state port")
        if region.continue_source is not None:
            shape_error("fold region declares a continue source; use a while region")
    elif region.kind == "while":
        if region.element_ports:
            shape_error(
                "while region binds no element ports; feed data through state or broadcast ports"
            )
        if not region.state_ports:
            shape_error("while region requires at least one state port")
        if region.continue_source is None:
            shape_error("while region requires a continue source")
        if region.max_iterations is None:
            shape_error("while region requires max_iterations (no infinite graphs)")
        if region.binding in ("cross", "broadcast"):
            shape_error(f"while region has no element bindings to {region.binding}")
    if region.max_iterations is not None and region.max_iterations < 1:
        shape_error("max_iterations must be >= 1")

    if not region.outputs:
        shape_error("region declares no outputs, so nothing in it can execute")
    state_output_keys = {out_id for out_id, out in region.outputs.items() if out.mode == "state"}
    for port in region.state_ports:
        if port not in state_output_keys:
            diags.append(
                Diagnostic(
                    "error",
                    "state-chain",
                    f"state port '{port}' needs a state-mode output of the "
                    "same id naming the body output that carries it forward",
                    where,
                    port,
                )
            )
    for out_id in sorted(state_output_keys - set(region.state_ports)):
        diags.append(
            Diagnostic(
                "error",
                "state-chain",
                f"state-mode output '{out_id}' does not correspond to a declared state port",
                where,
                out_id,
            )
        )

    for out_id, out in region.outputs.items():
        if not _body_output_exists(region, data, out.source):
            diags.append(
                Diagnostic(
                    "error",
                    "dangling-output",
                    f"region output '{out_id}' references missing body "
                    f"output '{out.source.node_id}/{out.source.output_id}'",
                    where,
                    out_id,
                )
            )
            continue
        src_type = _body_output_type(region, data, out.source)
        if src_type is None:
            continue  # body-side problem, reported there
        if out.mode in ("gather", "compact"):
            # Gather builds list values (including empty ones for zero
            # iterations), so the element type must be runtime-resolvable -
            # the same rule as literals and worker outputs.
            if src_type.runtime_type_id() is None:
                mode_name = "compact gather" if out.mode == "compact" else "gather"
                diags.append(
                    Diagnostic(
                        "error",
                        "gather-nonconcrete",
                        f"region output '{out_id}' uses {mode_name} on a "
                        f"{src_type.kind}-typed body output; {mode_name} requires "
                        "a concrete (runtime-resolvable) type",
                        where,
                        out_id,
                    )
                )
        elif out.mode == "flatten":
            if src_type.cardinality() != "list":
                diags.append(
                    Diagnostic(
                        "error",
                        "flatten-non-list",
                        f"region output '{out_id}' flattens a scalar body output; "
                        "flatten requires a list-typed body output",
                        where,
                        out_id,
                    )
                )
            elif src_type.runtime_type_id() is None:
                diags.append(
                    Diagnostic(
                        "error",
                        "flatten-nonconcrete",
                        f"region output '{out_id}' flattens a non-runtime-resolvable list type",
                        where,
                        out_id,
                    )
                )
        elif out.mode == "state":
            port_type = region.ports.get(out_id)
            src_runtime = src_type.runtime_type_id()
            if (
                port_type is not None
                and src_runtime is not None
                and not port_type.accepts_concrete(src_runtime)
            ):
                diags.append(
                    Diagnostic(
                        (
                            "error"
                            if combo_type_mismatch_is_error(src_runtime, port_type)
                            else "warning"
                        ),
                        "type-mismatch",
                        f"state port '{out_id}' expects {port_type.kind}"
                        f"{list(port_type.types)} but its chain produces "
                        f"{src_runtime}",
                        where,
                        out_id,
                    )
                )

    if region.continue_source is not None:
        if not _body_output_exists(region, data, region.continue_source):
            diags.append(
                Diagnostic(
                    "error",
                    "dangling-output",
                    "continue source references missing body output "
                    f"'{region.continue_source.node_id}/"
                    f"{region.continue_source.output_id}'",
                    where,
                )
            )
        else:
            cont_type = _body_output_type(region, data, region.continue_source)
            cont_runtime = None if cont_type is None else cont_type.runtime_type_id()
            if cont_runtime is not None and cont_runtime != "core.boolean":
                expected_boolean = TypeExpr.concrete("core.boolean")
                diags.append(
                    Diagnostic(
                        (
                            "error"
                            if combo_type_mismatch_is_error(cont_runtime, expected_boolean)
                            else "warning"
                        ),
                        "type-mismatch",
                        f"continue source produces {cont_runtime}, expected core.boolean",
                        where,
                    )
                )

    for input_id, value in region.inputs.items():
        port_type = region.ports.get(input_id)
        expected = (
            None
            if port_type is None
            else TypeExpr.list_of(port_type)
            if input_id in region.element_ports
            else port_type
        )
        # Region inputs behave like required skip-policy inputs (an absent
        # arriving at any region input skips the whole region), so no
        # maybe-absent warning applies.
        _check_input_value(level, node_id, input_id, value, expected, "skip", diags)

    body_level = _Level(
        graph=region.body,
        schemas=level.schemas,
        effective=data.body_effective,
        regions=data.body_regions,
        ports=region.ports,
        prefix=f"{level.prefix}{node_id}/",
        known_types=level.known_types,
    )
    _validate_level(body_level, diags)


def _validate_graph_node(
    level: _Level, node_id: str, node: GraphNode, diags: list[Diagnostic]
) -> None:
    where = level.prefix + node_id
    base_schema = level.schemas.get(node.node_type)
    if base_schema is None:
        return  # elaborate_graph reported unknown-node-type

    for fam in base_schema.input_families:
        suffixes = {
            suffix.split(".", 1)[0]
            for key in node.inputs
            if (suffix := fam.member_suffix(key)) is not None
        }
        suffixes.update(
            suffix.split(".", 1)[0]
            for key in node.slot_variants
            if (suffix := fam.member_suffix(key)) is not None
        )
        count = len(suffixes)
        if count < fam.min_members:
            diags.append(
                Diagnostic(
                    "error",
                    "family-too-few",
                    f"family '{fam.id}' has {count} members, needs >= {fam.min_members}",
                    where,
                )
            )
        if fam.max_members is not None and count > fam.max_members:
            diags.append(
                Diagnostic(
                    "error",
                    "family-too-many",
                    f"family '{fam.id}' has {count} members, allows <= {fam.max_members}",
                    where,
                )
            )

    schema = level.effective.get(node_id)
    if schema is None:
        return  # elaboration failed; elaborate_graph reported it

    for out_fam in base_schema.output_families:
        count = len(node.output_members.get(out_fam.id, ()))
        if count < out_fam.min_members:
            diags.append(
                Diagnostic(
                    "error",
                    "family-too-few",
                    f"output family '{out_fam.id}' has {count} members, "
                    f"needs >= {out_fam.min_members}",
                    where,
                )
            )
        if out_fam.max_members is not None and count > out_fam.max_members:
            diags.append(
                Diagnostic(
                    "error",
                    "family-too-many",
                    f"output family '{out_fam.id}' has {count} members, "
                    f"allows <= {out_fam.max_members}",
                    where,
                )
            )

    for input_id, value in node.inputs.items():
        spec = schema.input(input_id)
        if spec is None:
            diags.append(
                Diagnostic(
                    "error",
                    "unknown-input",
                    f"{node.node_type} has no input '{input_id}'",
                    where,
                    input_id,
                )
            )
            continue
        _check_input_value(
            level,
            node_id,
            input_id,
            value,
            spec.type,
            spec.absent_policy(),
            diags,
            asset_stamp_required=(
                (
                    isinstance(spec.widget, AssetWidget)
                    or (
                        isinstance(spec.widget, WidgetRepresentations)
                        and all(
                            isinstance(representation.widget, AssetWidget)
                            for representation in spec.widget.representations
                        )
                    )
                )
                and spec.type.kind == "concrete"
                and spec.type != TypeExpr.concrete("dinkster.asset")
            ),
        )

    for spec in schema.inputs:
        if spec.required and spec.default is None and spec.id not in node.inputs:
            diags.append(
                Diagnostic(
                    "error",
                    "missing-input",
                    f"{node.node_type} requires input '{spec.id}'",
                    where,
                    spec.id,
                )
            )


def has_errors(diags: Sequence[Diagnostic]) -> bool:
    return any(d.severity == "error" for d in diags)
