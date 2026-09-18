"""ComfyUI API prompt adapter: v1 prompt JSON -> native Dinkster Graph.

ComfyUI's headless submission format is a flat mapping of node ids to
``{"class_type": <v1 name>, "inputs": {...}}`` where a link is exactly a
2-list ``[node_id, output_index]`` (mirroring comfy_execution.graph_utils
``is_link``) and everything else is a literal. This module translates that
shape - once, at the boundary - into a native Graph plus targets, so the
engine, graph model, and server never learn that ``class_type`` or
positional output indexes existed.

Resolution goes through ``NodeSchema.aliases`` (the compat translator
records each v1 class_type as an alias of its namespaced Dinkster node type),
never by parsing type ids back apart. An alias claimed by two schemas is
refused at use with the candidates listed - v1's flat global namespace is
a bug we surface, not a tie we break silently.

Translation is strict where silence would mislead (unknown class_type,
ambiguous alias, dangling link, out-of-range output index, node ids
violating the native grammar) and permissive where the engine already
validates (input names/values pass through untouched; graph validation
anchors any type or interface errors exactly as it does for native
submissions). Every refusal is an anchored problem, and all problems are
collected before failing - one bad node does not hide the next.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from dinkster_graph import lower_selectors
from dinkster_graph.model import (
    NODE_ID_FORBIDDEN_CHARS,
    PORTS_NODE_ID,
    Graph,
    GraphNode,
    Link,
)
from dinkster_graph.wire import LINK_KEY
from dinkster_schema import (
    DynamicComboSpec,
    DynamicEntry,
    DynamicSlotSpec,
    InputFamilySpec,
    NodeSchema,
)
from dinkster_workers import CompatGateDiagnostic

from .listmap import lower_implicit_list_maps
from .usdu import lower_ultimate_sd_upscale

__all__ = [
    "InputAdapter",
    "PromptProblem",
    "PromptTranslation",
    "PromptTranslationError",
    "build_alias_index",
    "extract_prompt",
    "translate_prompt",
]

InputAdapter = Callable[[str, dict[str, object]], tuple[dict[str, object], list["PromptProblem"]]]
"""Rewrites one translated node's inputs at the prompt boundary.

The seam for v1 input shapes a natively-ported node no longer speaks
(e.g. SaveImage's raw ``filename_prefix`` becoming a structured save
target). Called with (node_id, built inputs) after links are resolved,
keyed by the submitted class_type first and the RESOLVED node type as a
fallback, so several legacy classes folded into one native node can each
keep their own input shape; returns the adapted inputs plus any anchored
problems. Adapters run only at this boundary - native submissions never
pass through them."""


@dataclass(frozen=True)
class PromptProblem:
    """One anchored refusal; ``node_id``/``input_id`` are empty when the
    problem is prompt-level (nothing to anchor to)."""

    code: str
    message: str
    node_id: str = ""
    input_id: str = ""
    compat_diagnostic: CompatGateDiagnostic | None = None

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {"code": self.code, "message": self.message}
        if self.node_id:
            wire["nodeId"] = self.node_id
        if self.input_id:
            wire["inputId"] = self.input_id
        if self.compat_diagnostic is not None:
            wire["compatDiagnostic"] = self.compat_diagnostic.to_wire()
        return wire


class PromptTranslationError(ValueError):
    """The prompt could not be translated; carries every collected problem."""

    def __init__(self, problems: list[PromptProblem]) -> None:
        self.problems = tuple(problems)
        summary = "; ".join(
            (f"{p.node_id}: " if p.node_id else "") + p.message for p in problems[:5]
        )
        if len(problems) > 5:
            summary += f" (and {len(problems) - 5} more)"
        super().__init__(f"comfy prompt translation failed: {summary}")


@dataclass(frozen=True)
class PromptTranslation:
    """A translated prompt: the native graph plus derived targets.

    Targets are every prompt node whose schema declares ``output_node``
    (v1 OUTPUT_NODE) - the same rule ComfyUI's executor applies to a
    prompt, made explicit because Dinkster jobs always carry targets.
    """

    graph: Graph
    targets: tuple[str, ...]


def _is_link(value: object) -> bool:
    """Byte-for-byte mirror of comfy_execution.graph_utils.is_link: a
    2-list of [str, int|float]. bool is excluded (it is not an index)."""
    if not isinstance(value, list):
        return False
    items = cast("list[object]", value)
    if len(items) != 2:
        return False
    if not isinstance(items[0], str):
        return False
    index = items[1]
    return isinstance(index, (int, float)) and not isinstance(index, bool)


def _contains_reserved_key(value: object) -> bool:
    """Literal dicts carrying the graph wire's reserved ``$link`` key cannot
    round-trip through the native wire; refuse rather than guess."""
    if isinstance(value, dict):
        mapping = cast("dict[Any, Any]", value)
        if LINK_KEY in mapping:
            return True
        return any(_contains_reserved_key(item) for item in mapping.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_reserved_key(item) for item in cast("Any", value))
    return False


def build_alias_index(schemas: Mapping[str, NodeSchema]) -> dict[str, list[str]]:
    """class_type -> candidate node types. Exact node_type ids resolve too,
    so a caller already speaking native type ids is not punished for it."""
    index: dict[str, list[str]] = {}
    for node_type, schema in schemas.items():
        index.setdefault(node_type, []).append(node_type)
        for alias in schema.aliases:
            if alias == node_type:
                continue
            index.setdefault(alias, []).append(node_type)
            if (
                node_type.startswith("dinkster.")
                and "." not in alias
                and f"comfy.{alias}" not in schemas
            ):
                index.setdefault(f"comfy.{alias}", []).append(node_type)
    return {alias: list(dict.fromkeys(candidates)) for alias, candidates in index.items()}


def _valid_node_id(node_id: str) -> bool:
    if not node_id or node_id == PORTS_NODE_ID:
        return False
    return not any(ch in NODE_ID_FORBIDDEN_CHARS for ch in node_id)


def _extract_combo_choices(
    entries: Sequence[DynamicEntry],
    parent_path: str,
    inputs: dict[str, object],
    choices: dict[str, str],
    problems: list[PromptProblem],
    node_id: str,
) -> None:
    """Move active Comfy DynamicCombo selector values out of the flat
    prompt inputs and into Dinkster's document-state choice map."""
    for entry in entries:
        path = f"{parent_path}.{entry.id}" if parent_path else entry.id
        if isinstance(entry, DynamicComboSpec):
            if path not in inputs:
                continue
            raw = inputs.get(path)
            if not isinstance(raw, str):
                problems.append(
                    PromptProblem(
                        code="prompt.bad_dynamic_choice",
                        message=f"dynamic combo selector {path!r} must be a string",
                        node_id=node_id,
                        input_id=path,
                    )
                )
                continue
            option = entry.option(raw)
            if option is None:
                problems.append(
                    PromptProblem(
                        code="prompt.bad_dynamic_choice",
                        message=f"dynamic combo selector {path!r} has unknown option {raw!r}",
                        node_id=node_id,
                        input_id=path,
                    )
                )
                continue
            inputs.pop(path)
            choices[path] = raw
            _extract_combo_choices(
                option.inputs,
                path,
                inputs,
                choices,
                problems,
                node_id,
            )
        elif isinstance(entry, InputFamilySpec):
            prefix = path + "."
            suffixes = dict.fromkeys(
                input_id[len(prefix) :].split(".", 1)[0]
                for input_id in inputs
                if input_id.startswith(prefix)
            )
            for suffix in suffixes:
                _extract_combo_choices(
                    entry.template,
                    f"{path}.{suffix}",
                    inputs,
                    choices,
                    problems,
                    node_id,
                )
        elif isinstance(entry, DynamicSlotSpec) and path in inputs:
            _extract_combo_choices(
                entry.inputs,
                path,
                inputs,
                choices,
                problems,
                node_id,
            )


def translate_prompt(
    prompt: Mapping[str, Any],
    schemas: Mapping[str, NodeSchema],
    *,
    input_adapters: Mapping[str, InputAdapter] | None = None,
    skipped_classes: Mapping[str, CompatGateDiagnostic] | None = None,
) -> PromptTranslation:
    """Translate a ComfyUI API prompt into a native Graph + targets.

    Raises PromptTranslationError with every collected problem if any part
    of the prompt cannot be translated faithfully.
    """
    problems: list[PromptProblem] = []
    if not prompt:
        raise PromptTranslationError(
            [PromptProblem(code="prompt.empty", message="prompt has no nodes")]
        )

    index = build_alias_index(schemas)

    # Pass 1: resolve every node's class_type so pass 2 can map link output
    # indexes through the *referenced* node's schema.
    resolved: dict[str, NodeSchema] = {}
    for node_id, entry in prompt.items():
        # JSON object keys are always strings; only the grammar needs checking.
        if not _valid_node_id(node_id):
            problems.append(
                PromptProblem(
                    code="prompt.bad_node_id",
                    message=(
                        f"node id {node_id!r} is not a valid Dinkster node id "
                        "(non-empty, not '$region', no '/', '[' or ']')"
                    ),
                    node_id=node_id,
                )
            )
            continue
        if not isinstance(entry, Mapping):
            problems.append(
                PromptProblem(
                    code="prompt.bad_node",
                    message="node entry must be an object",
                    node_id=node_id,
                )
            )
            continue
        entry_map = cast("Mapping[str, Any]", entry)
        class_type = entry_map.get("class_type")
        if not isinstance(class_type, str) or not class_type:
            problems.append(
                PromptProblem(
                    code="prompt.missing_class_type",
                    message="node entry has no 'class_type'",
                    node_id=node_id,
                )
            )
            continue
        candidates = index.get(class_type, [])
        if not candidates:
            diagnostic = (skipped_classes or {}).get(class_type)
            if diagnostic is not None:
                problems.append(
                    PromptProblem(
                        code=diagnostic.code,
                        message=diagnostic.reason,
                        node_id=node_id,
                        input_id=diagnostic.input_id or "",
                        compat_diagnostic=diagnostic,
                    )
                )
                continue
            problems.append(
                PromptProblem(
                    code="prompt.unknown_class_type",
                    message=f"unknown class_type {class_type!r}: no loaded node resolves it",
                    node_id=node_id,
                )
            )
            continue
        if len(candidates) > 1:
            listed = ", ".join(sorted(candidates))
            problems.append(
                PromptProblem(
                    code="prompt.ambiguous_class_type",
                    message=(
                        f"class_type {class_type!r} is claimed by multiple loaded "
                        f"packs ({listed}); submit the full node type id instead"
                    ),
                    node_id=node_id,
                )
            )
            continue
        resolved[node_id] = schemas[candidates[0]]

    # Pass 2: inputs. Links map positional output indexes to the referenced
    # schema's output ids (translation preserves v1 output order); literals
    # pass through untouched.
    nodes: dict[str, GraphNode] = {}
    for node_id, schema in resolved.items():
        entry_map = cast("Mapping[str, Any]", prompt[node_id])
        raw_inputs = entry_map.get("inputs", {})
        if not isinstance(raw_inputs, Mapping):
            problems.append(
                PromptProblem(
                    code="prompt.bad_inputs",
                    message="'inputs' must be an object",
                    node_id=node_id,
                )
            )
            continue
        inputs: dict[str, object] = {}
        for input_id, value in cast("Mapping[Any, Any]", raw_inputs).items():
            if not isinstance(input_id, str):
                problems.append(
                    PromptProblem(
                        code="prompt.bad_input_id",
                        message=f"input name {input_id!r} must be a string",
                        node_id=node_id,
                    )
                )
                continue
            if _is_link(value):
                link_list = cast("list[Any]", value)
                ref_id, raw_index = cast(str, link_list[0]), link_list[1]
                ref_index = int(cast("int | float", raw_index))
                if ref_index != raw_index or ref_index < 0:
                    problems.append(
                        PromptProblem(
                            code="prompt.bad_output_index",
                            message=f"link output index {raw_index!r} is not a valid index",
                            node_id=node_id,
                            input_id=input_id,
                        )
                    )
                    continue
                if ref_id not in prompt:
                    problems.append(
                        PromptProblem(
                            code="prompt.dangling_link",
                            message=f"link references node {ref_id!r}, which is not in the prompt",
                            node_id=node_id,
                            input_id=input_id,
                        )
                    )
                    continue
                ref_schema = resolved.get(ref_id)
                if ref_schema is None:
                    # The referenced node already carries its own problem.
                    continue
                if ref_index >= len(ref_schema.outputs):
                    problems.append(
                        PromptProblem(
                            code="prompt.bad_output_index",
                            message=(
                                f"link output index {ref_index} is out of range for "
                                f"{ref_schema.node_type} "
                                f"({len(ref_schema.outputs)} outputs)"
                            ),
                            node_id=node_id,
                            input_id=input_id,
                        )
                    )
                    continue
                inputs[input_id] = Link(node_id=ref_id, output_id=ref_schema.outputs[ref_index].id)
                continue
            if _contains_reserved_key(value):
                problems.append(
                    PromptProblem(
                        code="prompt.reserved_key",
                        message=(
                            f"literal input contains the reserved key {LINK_KEY!r} "
                            "and cannot be represented in the native graph wire"
                        ),
                        node_id=node_id,
                        input_id=input_id,
                    )
                )
                continue
            inputs[input_id] = value
        adapters = input_adapters or {}
        submitted_class = entry_map.get("class_type")
        adapter = (
            adapters.get(submitted_class) if isinstance(submitted_class, str) else None
        ) or adapters.get(schema.node_type)
        if adapter is not None:
            inputs, adapter_problems = adapter(node_id, inputs)
            problems.extend(adapter_problems)
        choices: dict[str, str] = {}
        _extract_combo_choices(
            (*schema.input_families, *schema.combos, *schema.slots),
            "",
            inputs,
            choices,
            problems,
            node_id,
        )
        nodes[node_id] = GraphNode(
            node_type=schema.node_type,
            inputs=inputs,
            slot_variants=choices,
        )

    targets = tuple(node_id for node_id in nodes if resolved[node_id].output_node)
    if not problems and not targets:
        problems.append(
            PromptProblem(
                code="prompt.no_output_nodes",
                message=("prompt contains no output nodes (v1 OUTPUT_NODE); nothing to target"),
            )
        )

    if problems:
        raise PromptTranslationError(problems)
    lowered = lower_selectors(Graph(nodes=nodes), targets, schemas)
    if lowered.problems:
        raise PromptTranslationError(
            [PromptProblem(p.code, p.message, p.node_id, p.input_id) for p in lowered.problems]
        )
    usdu = lower_ultimate_sd_upscale(lowered.graph, schemas)
    if usdu.problems:
        raise PromptTranslationError(
            [
                PromptProblem(problem.code, problem.message, problem.node_id, problem.input_id)
                for problem in usdu.problems
            ]
        )
    list_mapped = lower_implicit_list_maps(usdu.graph, schemas)
    if list_mapped.problems:
        raise PromptTranslationError(
            [
                PromptProblem(problem.code, problem.message, problem.node_id)
                for problem in list_mapped.problems
            ]
        )
    return PromptTranslation(graph=list_mapped.graph, targets=lowered.targets)


def extract_prompt(
    body: Any,
) -> tuple[Mapping[str, Any], str | None, Mapping[str, Any] | None]:
    """Accept both submission shapes ComfyUI clients use: the bare prompt
    object, or the ``{"prompt": {...}, "client_id": ...}`` wrapper POSTed
    to ComfyUI's ``/prompt``.

    The rule is deterministic: a body is a wrapper iff it carries a
    ``prompt`` key whose value is an object and the body itself is not a
    prompt (some top-level value is not a node entry). A prompt that
    happens to contain a node literally named "prompt" therefore still
    parses as a prompt.
    """
    if not isinstance(body, Mapping):
        raise PromptTranslationError(
            [PromptProblem(code="prompt.bad_body", message="body must be a JSON object")]
        )
    body_map = cast("Mapping[str, Any]", body)

    def looks_like_prompt(candidate: Mapping[str, Any]) -> bool:
        return bool(candidate) and all(
            isinstance(entry, Mapping) and "class_type" in cast("Mapping[str, Any]", entry)
            for entry in candidate.values()
        )

    wrapped = body_map.get("prompt")
    if isinstance(wrapped, Mapping) and not looks_like_prompt(body_map):
        client_id = body_map.get("client_id")
        extra_pnginfo = None
        extra_data = body_map.get("extra_data")
        if isinstance(extra_data, Mapping):
            candidate = cast("Mapping[str, Any]", extra_data).get("extra_pnginfo")
            if candidate is not None and not isinstance(candidate, Mapping):
                raise PromptTranslationError(
                    [
                        PromptProblem(
                            code="prompt.bad_extra_pnginfo",
                            message="extra_data.extra_pnginfo must be an object or null",
                        )
                    ]
                )
            if candidate is not None:
                extra_pnginfo = cast("Mapping[str, Any]", candidate)
        return (
            cast("Mapping[str, Any]", wrapped),
            client_id if isinstance(client_id, str) and client_id else None,
            extra_pnginfo,
        )
    return body_map, None, None
