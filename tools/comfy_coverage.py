"""Measure maintained ComfyUI translation coverage over pinned workflows."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dinkster_inference import builtin_families
from dinkster_workers.manifest import (
    COMFY_ALIASES_FILENAME,
    COMFY_ALIASES_MAX_BYTES,
    COMFY_GROUPS_FILENAME,
    ManifestError,
    load_manifest,
)

if __package__:
    from tools.comfy_confidence import ConfidenceReceiptError, load_receipt, verify_receipt
else:
    from comfy_confidence import ConfidenceReceiptError, load_receipt, verify_receipt

FORMAT = "dinkster-comfy-coverage/1"
EVIDENCE_FORMAT = "dinkster-capability-evidence/1"
EVIDENCE_SOURCE_FORMAT = "dinkster-capability-evidence-source/1"
SNAPSHOT_FORMAT = "comfy-registry-download-snapshot/1"
SOURCE_PARITY_BASELINE_FORMAT = "dinkster-comfy-source-parity-baseline/1"
TEMPLATE_REVISION = "d3b4a9e89573162b005961865164c18c8ae2206b"
COMFYUI_REVISION = "15eb748b3ec5f8a0a2d470b7fb280e2d7579f916"
_REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_TEMPLATE_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_TEMPLATE_FILES = 2_048
MAX_JSON_DEPTH = 64
MAX_JSON_ITEMS = 1_000_000
MAX_WORKFLOW_NODES = 100_000
MAX_RECEIPT_TREE_ITEMS = 100_000
MAX_RECEIPT_TREE_DEPTH = 16

STATUSES = ("mapped", "quarantine", "unavailable", "unsupported", "structural")
MAPPING_KINDS = ("op", "family")
CONFIDENCE_TIERS = ("exact", "parametric", "equivalent", "grouped")
EVIDENCE_TIERS = ("T0", "T1", "T2", "T3", "T4")
EVIDENCE_STATES = ("proven", "unverified", "absent", "refused")
EXPOSURES = ("supported", "implemented", "unlisted")
EVIDENCE_KINDS = frozenset(
    {
        "cold-performance",
        "compatibility-test",
        "correctness",
        "focused-test",
        "gpu-memory",
        "host-memory",
        "matched-e2e",
        "official-checkpoint",
        "residual-memory",
        "schema-test",
        "source-correctness",
        "template-adapter",
        "template-lora",
        "warm-performance",
    }
)
TEST_EVIDENCE_KINDS = frozenset(
    {
        "compatibility-test",
        "correctness",
        "focused-test",
        "matched-e2e",
        "schema-test",
        "source-correctness",
    }
)
RANK_BANDS = ("core", "top-355", "rank-356-900", "outside-top-900", "unresolved")
STRUCTURAL_NODE_TYPES = frozenset(
    {"Reroute", "Note", "MarkdownNote", "PrimitiveNode", "SetNode", "GetNode"}
)


class CoverageError(ValueError):
    """Coverage input is malformed, unsafe, or inconsistent."""


@dataclass(frozen=True)
class NodeOccurrence:
    node_type: str
    explicit_pack: str | None
    node_id: int | str | None = None
    widgets: tuple[object, ...] = ()


@dataclass(frozen=True)
class Workflow:
    path: str
    nodes: tuple[NodeOccurrence, ...]
    subgraph_ids: frozenset[str]
    subgraph_count: int


@dataclass(frozen=True)
class RegistryMapping:
    registry_id: str
    record_digest: str
    registry_kind: str
    mapping_kind: str
    source_pack: str
    source_name: str
    revision: str
    carrier: str
    target_provider: str
    tier: str
    evidence: tuple[str, ...]
    tolerances: tuple[tuple[str, str, float], ...]
    family_id: str | None
    family_provider: str | None
    refusal: bool = False


@dataclass(frozen=True)
class RegistryCatalog:
    mappings: tuple[RegistryMapping, ...]
    available_providers: frozenset[str]
    sidecars: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DownloadSnapshot:
    captured_at: str
    source: str
    total_packs: int
    total_downloads: int
    rank_355_downloads: int
    rank_900_downloads: int
    ranks: Mapping[str, tuple[int, str, int]]


@dataclass(frozen=True)
class EvidenceItem:
    kind: str
    selector: str


@dataclass(frozen=True)
class ModelEvidence:
    family_id: str
    state: str
    tier: str | None
    exposure: str
    source_identifiers: tuple[str, ...]
    evidence: tuple[EvidenceItem, ...]


@dataclass(frozen=True)
class LowStepLoraPath:
    path: str
    lora_node_id: int | str
    lora_node_type: str
    sampler_node_id: int | str
    sampler_node_type: str
    step_widget_index: int
    steps: int


@dataclass(frozen=True)
class EvidenceSource:
    model_families: tuple[ModelEvidence, ...]
    lora_node_types: frozenset[str]
    adapter_node_types: frozenset[str]
    low_step_lora_paths: tuple[LowStepLoraPath, ...]


@dataclass(frozen=True)
class ComfyUISource:
    node_ids: frozenset[str]
    class_names: frozenset[str] = frozenset()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CoverageError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise CoverageError(f"non-finite JSON number: {value}")


def _finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CoverageError(f"non-finite JSON number: {value}")
    return result


def _is_reparse_point(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _identity(value: os.stat_result) -> tuple[int, ...]:
    identity = (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    return identity if os.name == "nt" else (*identity, value.st_ctime_ns)


def _read_regular(path: Path, root: Path, maximum: int, kind: str) -> bytes:
    try:
        root_resolved = root.resolve(strict=True)
        path_stat = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CoverageError(f"cannot inspect {kind} {path}: {error}") from error
    if not resolved.is_relative_to(root_resolved):
        raise CoverageError(f"{kind} path escapes its root: {path}")
    if path.is_symlink() or _is_reparse_point(path_stat) or not stat.S_ISREG(path_stat.st_mode):
        raise CoverageError(f"{kind} must be an ordinary file: {path}")
    if not 0 < path_stat.st_size <= maximum:
        raise CoverageError(f"{kind} size must be from 1 through {maximum} bytes: {path}")
    try:
        with path.open("rb") as file:
            opened = os.fstat(file.fileno())
            if _identity(opened) != _identity(path_stat) or not stat.S_ISREG(opened.st_mode):
                raise CoverageError(f"{kind} changed while opening: {path}")
            data = file.read(maximum + 1)
            after = os.fstat(file.fileno())
    except CoverageError:
        raise
    except OSError as error:
        raise CoverageError(f"cannot read {kind} {path}: {error}") from error
    if len(data) > maximum or _identity(after) != _identity(opened):
        raise CoverageError(f"{kind} changed or exceeded its size limit: {path}")
    return data


def _validate_json_budget(value: object, where: str) -> None:
    remaining = MAX_JSON_ITEMS
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        remaining -= 1
        if remaining < 0:
            raise CoverageError(f"{where} exceeds the {MAX_JSON_ITEMS}-item limit")
        if depth > MAX_JSON_DEPTH:
            raise CoverageError(f"{where} exceeds the {MAX_JSON_DEPTH}-level limit")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)


def _decode_json(data: bytes, path: Path, kind: str) -> object:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except CoverageError:
        raise
    except (OverflowError, RecursionError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise CoverageError(f"{kind} is not valid JSON: {path}: {error}") from error
    _validate_json_budget(value, str(path))
    return value


def _load_json(path: Path, root: Path, maximum: int, kind: str) -> object:
    return _decode_json(_read_regular(path, root, maximum, kind), path, kind)


def _object(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CoverageError(f"{where} must be an object with string keys")
    return cast("dict[str, Any]", value)


def _array(value: object, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise CoverageError(f"{where} must be an array")
    return cast("list[Any]", value)


def _fields(value: object, where: str, required: set[str]) -> dict[str, Any]:
    obj = _object(value, where)
    missing = sorted(required - set(obj))
    unknown = sorted(set(obj) - required)
    if missing:
        raise CoverageError(f"{where} is missing fields: {', '.join(missing)}")
    if unknown:
        raise CoverageError(f"{where} has unknown fields: {', '.join(unknown)}")
    return obj


def _string(value: object, where: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise CoverageError(f"{where} must be a non-empty string of at most {maximum} chars")
    return value


def _ordinary_directory(path: Path, kind: str) -> None:
    try:
        value = path.lstat()
    except OSError as error:
        raise CoverageError(f"cannot inspect {kind} directory {path}: {error}") from error
    if path.is_symlink() or _is_reparse_point(value) or not stat.S_ISDIR(value.st_mode):
        raise CoverageError(f"{kind} root must be an ordinary directory: {path}")


def _workflow_nodes(value: object, where: str) -> tuple[NodeOccurrence, ...]:
    nodes: list[NodeOccurrence] = []
    for index, raw in enumerate(_array(value, where)):
        node = _object(raw, f"{where}[{index}]")
        node_type = _string(node.get("type"), f"{where}[{index}].type")
        node_id = node.get("id")
        if node_id is not None and not (
            type(node_id) is int or isinstance(node_id, str) and bool(node_id)
        ):
            raise CoverageError(f"{where}[{index}].id must be a non-empty string or integer")
        widgets: tuple[object, ...] = ()
        if "widgets_values" in node:
            raw_widgets = node["widgets_values"]
            if isinstance(raw_widgets, list):
                widgets = tuple(raw_widgets)
        explicit_pack: str | None = None
        properties = node.get("properties")
        if properties is not None:
            property_object = _object(properties, f"{where}[{index}].properties")
            provider = property_object.get("cnr_id")
            if provider is not None:
                explicit_pack = _string(provider, f"{where}[{index}].properties.cnr_id", 128)
        nodes.append(
            NodeOccurrence(
                node_type=node_type,
                explicit_pack=explicit_pack,
                node_id=cast("int | str | None", node_id),
                widgets=widgets,
            )
        )
    return tuple(nodes)


def load_workflows(templates_root: Path) -> tuple[tuple[Workflow, ...], tuple[str, ...]]:
    """Load bounded workflow JSON as data and report non-workflow JSON separately."""
    _ordinary_directory(templates_root, "template")
    paths = sorted(templates_root.glob("*.json"), key=lambda path: path.name)
    if len(paths) > MAX_TEMPLATE_FILES:
        raise CoverageError(f"template file count exceeds {MAX_TEMPLATE_FILES}")
    workflows: list[Workflow] = []
    ignored: list[str] = []
    total_nodes = 0
    for path in paths:
        document = _load_json(path, templates_root, MAX_TEMPLATE_BYTES, "template")
        if not isinstance(document, dict) or not isinstance(document.get("nodes"), list):
            ignored.append(path.name)
            continue
        root = cast("dict[str, Any]", document)
        nodes = list(_workflow_nodes(root["nodes"], f"{path.name}.nodes"))
        definitions = root.get("definitions")
        subgraphs: list[Any] = []
        if definitions is not None:
            definition_object = _object(definitions, f"{path.name}.definitions")
            if "subgraphs" in definition_object:
                subgraphs = _array(
                    definition_object["subgraphs"], f"{path.name}.definitions.subgraphs"
                )
        subgraph_ids: set[str] = set()
        for index, raw in enumerate(subgraphs):
            where = f"{path.name}.definitions.subgraphs[{index}]"
            subgraph = _object(raw, where)
            subgraph_id = _string(subgraph.get("id"), f"{where}.id")
            if subgraph_id in subgraph_ids:
                raise CoverageError(f"{path.name} has duplicate subgraph id {subgraph_id!r}")
            subgraph_ids.add(subgraph_id)
            nodes.extend(_workflow_nodes(subgraph.get("nodes"), f"{where}.nodes"))
        total_nodes += len(nodes)
        if total_nodes > MAX_WORKFLOW_NODES:
            raise CoverageError(f"workflow node count exceeds {MAX_WORKFLOW_NODES}")
        workflows.append(
            Workflow(
                path=path.name,
                nodes=tuple(nodes),
                subgraph_ids=frozenset(subgraph_ids),
                subgraph_count=len(subgraphs),
            )
        )
    return tuple(workflows), tuple(ignored)


def _node_id(value: object, where: str) -> int | str:
    if type(value) is int:
        return cast("int", value)
    return _string(value, where, 128)


def _string_array(value: object, where: str) -> tuple[str, ...]:
    items = tuple(
        _string(item, f"{where}[{index}]", 512) for index, item in enumerate(_array(value, where))
    )
    if len(set(items)) != len(items):
        raise CoverageError(f"{where} must not contain duplicates")
    return items


def _validate_evidence_selector(
    selector: str, repo_root: Path, *, require_collected_test: bool = False
) -> None:
    pieces = selector.split("::")
    if len(pieces) > 2 or not pieces[0]:
        raise CoverageError(f"invalid evidence selector: {selector!r}")
    relative = Path(pieces[0])
    if relative.is_absolute() or ".." in relative.parts:
        raise CoverageError(f"evidence selector must be repository-relative: {selector!r}")
    if require_collected_test and (
        len(pieces) != 2
        or not re.fullmatch(r"(?:test_.+|.+_test)\.py", relative.name)
        or not re.fullmatch(r"test_[A-Za-z0-9_]+", pieces[1])
    ):
        raise CoverageError(f"test-backed evidence must name a collected test: {selector!r}")
    path = repo_root / relative
    data = _read_regular(path, repo_root, MAX_SOURCE_BYTES, "evidence")
    if len(pieces) == 1:
        return
    name = pieces[1]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise CoverageError(f"invalid evidence test name: {selector!r}")
    try:
        tree = ast.parse(data.decode("utf-8-sig"), filename=str(path))
    except (SyntaxError, UnicodeError) as error:
        raise CoverageError(f"cannot parse evidence selector {selector!r}: {error}") from error
    definitions = tree.body if require_collected_test else ast.walk(tree)
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in definitions
    ):
        raise CoverageError(f"evidence selector names no test function: {selector!r}")


def load_evidence_source(path: Path, repo_root: Path) -> EvidenceSource:
    document = _load_json(path, repo_root, MAX_TEMPLATE_BYTES, "capability evidence source")
    root = _fields(
        document,
        "capability evidence source",
        {"format", "modelFamilies", "templateFeatures"},
    )
    if root["format"] != EVIDENCE_SOURCE_FORMAT:
        raise CoverageError(f"unsupported capability evidence source format: {root['format']!r}")

    models: list[ModelEvidence] = []
    model_ids: set[str] = set()
    for index, raw in enumerate(_array(root["modelFamilies"], "modelFamilies")):
        where = f"modelFamilies[{index}]"
        model = _fields(
            raw,
            where,
            {"id", "state", "tier", "exposure", "sourceIdentifiers", "evidence"},
        )
        family_id = _string(model["id"], f"{where}.id", 128)
        if family_id in model_ids:
            raise CoverageError(f"duplicate model family evidence: {family_id}")
        model_ids.add(family_id)
        state = _string(model["state"], f"{where}.state", 32)
        if state not in EVIDENCE_STATES:
            raise CoverageError(f"{where}.state must be one of {', '.join(EVIDENCE_STATES)}")
        raw_tier = model["tier"]
        tier = None if raw_tier is None else _string(raw_tier, f"{where}.tier", 2)
        if tier is not None and tier not in EVIDENCE_TIERS:
            raise CoverageError(f"{where}.tier must be one of {', '.join(EVIDENCE_TIERS)}")
        if (state == "proven") != (tier is not None):
            raise CoverageError(f"{where} must have a tier exactly when state is proven")
        exposure = _string(model["exposure"], f"{where}.exposure", 32)
        if exposure not in EXPOSURES:
            raise CoverageError(f"{where}.exposure must be one of {', '.join(EXPOSURES)}")
        source_identifiers = _string_array(model["sourceIdentifiers"], f"{where}.sourceIdentifiers")
        if not source_identifiers:
            raise CoverageError(f"{where}.sourceIdentifiers must not be empty")
        evidence: list[EvidenceItem] = []
        for evidence_index, raw_evidence in enumerate(
            _array(model["evidence"], f"{where}.evidence")
        ):
            evidence_where = f"{where}.evidence[{evidence_index}]"
            item = _fields(raw_evidence, evidence_where, {"kind", "selector"})
            evidence_item = EvidenceItem(
                kind=_string(item["kind"], f"{evidence_where}.kind", 64),
                selector=_string(item["selector"], f"{evidence_where}.selector", 512),
            )
            if evidence_item.kind not in EVIDENCE_KINDS:
                raise CoverageError(f"{evidence_where}.kind is not a recognized evidence kind")
            _validate_evidence_selector(
                evidence_item.selector,
                repo_root,
                require_collected_test=evidence_item.kind in TEST_EVIDENCE_KINDS,
            )
            evidence.append(evidence_item)
        kinds = {item.kind for item in evidence}
        required_kinds: set[str] = set()
        if tier == "T0":
            required_kinds = {"schema-test"}
        elif tier == "T1":
            required_kinds = {"compatibility-test"}
        elif tier == "T2":
            required_kinds = {"focused-test"}
        elif tier == "T3":
            required_kinds = {"focused-test", "official-checkpoint", "source-correctness"}
        elif tier == "T4":
            required_kinds = {
                "focused-test",
                "official-checkpoint",
                "matched-e2e",
                "correctness",
                "cold-performance",
                "warm-performance",
                "gpu-memory",
                "host-memory",
                "residual-memory",
                "source-correctness",
                "template-lora",
                "template-adapter",
            }
        missing_kinds = sorted(required_kinds - kinds)
        if missing_kinds:
            raise CoverageError(
                f"{where} lacks evidence kinds required by {tier}: {', '.join(missing_kinds)}"
            )
        if state == "proven" and not evidence:
            raise CoverageError(f"{where}.evidence must not be empty for a proven capability")
        models.append(
            ModelEvidence(
                family_id=family_id,
                state=state,
                tier=tier,
                exposure=exposure,
                source_identifiers=source_identifiers,
                evidence=tuple(evidence),
            )
        )

    native_ids = {family.id for family in builtin_families()}
    if model_ids != native_ids:
        missing = sorted(native_ids - model_ids)
        unknown = sorted(model_ids - native_ids)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise CoverageError(
            f"model evidence does not match builtin_families(): {'; '.join(details)}"
        )

    features = _fields(
        root["templateFeatures"],
        "templateFeatures",
        {"loraNodeTypes", "adapterNodeTypes", "lowStepLoraPaths"},
    )
    lora_node_types = frozenset(_string_array(features["loraNodeTypes"], "loraNodeTypes"))
    adapter_node_types = frozenset(_string_array(features["adapterNodeTypes"], "adapterNodeTypes"))
    if not lora_node_types or not adapter_node_types or lora_node_types & adapter_node_types:
        raise CoverageError("LoRA and adapter node type sets must be non-empty and disjoint")
    low_step_paths: list[LowStepLoraPath] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(_array(features["lowStepLoraPaths"], "lowStepLoraPaths")):
        where = f"lowStepLoraPaths[{index}]"
        item = _fields(raw, where, {"path", "loraNode", "samplerNode"})
        path_value = _string(item["path"], f"{where}.path", 512)
        if path_value in seen_paths:
            raise CoverageError(f"duplicate low-step LoRA path: {path_value}")
        seen_paths.add(path_value)
        lora_node = _fields(item["loraNode"], f"{where}.loraNode", {"id", "type"})
        sampler_node = _fields(
            item["samplerNode"],
            f"{where}.samplerNode",
            {"id", "type", "stepWidgetIndex", "steps"},
        )
        lora_type = _string(lora_node["type"], f"{where}.loraNode.type", 128)
        if lora_type not in lora_node_types:
            raise CoverageError(f"{where}.loraNode.type is not a declared LoRA node type")
        widget_index = sampler_node["stepWidgetIndex"]
        steps = sampler_node["steps"]
        if type(widget_index) is not int or widget_index < 0:
            raise CoverageError(
                f"{where}.samplerNode.stepWidgetIndex must be a non-negative integer"
            )
        if type(steps) is not int or not 1 <= steps <= 8:
            raise CoverageError(f"{where}.samplerNode.steps must be an integer from 1 through 8")
        low_step_paths.append(
            LowStepLoraPath(
                path=path_value,
                lora_node_id=_node_id(lora_node["id"], f"{where}.loraNode.id"),
                lora_node_type=lora_type,
                sampler_node_id=_node_id(sampler_node["id"], f"{where}.samplerNode.id"),
                sampler_node_type=_string(sampler_node["type"], f"{where}.samplerNode.type", 128),
                step_widget_index=widget_index,
                steps=steps,
            )
        )
    return EvidenceSource(
        model_families=tuple(sorted(models, key=lambda item: item.family_id)),
        lora_node_types=lora_node_types,
        adapter_node_types=adapter_node_types,
        low_step_lora_paths=tuple(sorted(low_step_paths, key=lambda item: item.path)),
    )


def _ast_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _comfyui_symbols(path: Path, root: Path) -> tuple[set[str], set[str]]:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
        root_resolved = root.resolve(strict=True)
    except OSError as error:
        raise CoverageError(f"cannot inspect ComfyUI source {path}: {error}") from error
    if info.st_size == 0:
        if (
            path.is_symlink()
            or _is_reparse_point(info)
            or not stat.S_ISREG(info.st_mode)
            or not resolved.is_relative_to(root_resolved)
        ):
            raise CoverageError(f"ComfyUI source must be an ordinary file: {path}")
        return set(), set()
    data = _read_regular(path, root, MAX_SOURCE_BYTES, "ComfyUI source")
    try:
        tree = ast.parse(data.decode("utf-8-sig"), filename=str(path))
    except (SyntaxError, UnicodeError) as error:
        raise CoverageError(f"cannot parse ComfyUI source {path}: {error}") from error
    result: set[str] = set()
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            value = node.value
            if any(
                isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS"
                for target in targets
            ):
                if isinstance(value, ast.Dict):
                    result.update(
                        key
                        for raw_key in value.keys
                        if raw_key is not None and (key := _ast_string(raw_key)) is not None
                    )
            if any(isinstance(target, ast.Name) and target.id == "node_id" for target in targets):
                if (node_id := _ast_string(value)) is not None:
                    result.add(node_id)
        elif isinstance(node, ast.Call) and _call_name(node.func) == "Schema":
            for keyword in node.keywords:
                if keyword.arg == "node_id" and (node_id := _ast_string(keyword.value)) is not None:
                    result.add(node_id)
    return result, class_names


def load_comfyui_source(root: Path) -> ComfyUISource:
    _ordinary_directory(root, "ComfyUI")
    nodes_path = root / "nodes.py"
    if not nodes_path.is_file():
        raise CoverageError(f"ComfyUI checkout has no nodes.py: {root}")
    models_path = root / "comfy" / "supported_models.py"
    if not models_path.is_file():
        raise CoverageError(f"ComfyUI checkout has no comfy/supported_models.py: {root}")
    files = [nodes_path, models_path]
    for relative in ("comfy_extras", "comfy_api_nodes"):
        directory = root / relative
        _ordinary_directory(directory, "ComfyUI source")
        files.extend(sorted(directory.rglob("*.py"), key=lambda path: path.as_posix()))
    nodes: set[str] = set()
    class_names: set[str] = set()
    for path in files:
        path_nodes, path_classes = _comfyui_symbols(path, root)
        nodes.update(path_nodes)
        if path == models_path:
            class_names.update(path_classes)
    if not nodes:
        raise CoverageError("ComfyUI static source scan found no registered nodes")
    return ComfyUISource(node_ids=frozenset(nodes), class_names=frozenset(class_names))


def _tolerance_wire(tolerance: object) -> tuple[str, str, float]:
    return (
        cast("Any", tolerance).metric,
        cast("Any", tolerance).operator,
        float(cast("Any", tolerance).value),
    )


def _case_is_refusal(case: Any) -> bool:
    sources = [source for _, source in case.inputs]
    for _, family in case.input_families:
        sources.extend(source for _, source in family.inputs)
        for member in family.members:
            sources.extend(source for _, source in member.inputs)
    return any(
        source.transform is not None
        and source.transform.kind == "enumRename"
        and not source.transform.map
        for source in sources
    )


def load_registry_catalog(packages_root: Path) -> RegistryCatalog:
    """Load manifests and adjacent registries without importing pack code."""
    _ordinary_directory(packages_root, "packages")
    mappings: list[RegistryMapping] = []
    providers: set[str] = set()
    sidecars: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    seen_sources: set[tuple[str, str, str]] = set()
    for manifest_path in sorted(packages_root.glob("*/dinkster-pack.toml")):
        _ordinary_directory(manifest_path.parent, "pack")
        _read_regular(
            manifest_path,
            packages_root,
            COMFY_ALIASES_MAX_BYTES,
            "pack manifest",
        )
        manifest = load_manifest(manifest_path)
        providers.add(manifest.name)
        registries = (
            ("alias", manifest.comfy_aliases, COMFY_ALIASES_FILENAME),
            ("group", manifest.comfy_groups, COMFY_GROUPS_FILENAME),
        )
        for registry_kind, registry, filename in registries:
            if registry is None:
                continue
            sidecar = manifest_path.with_name(filename)
            data = _read_regular(sidecar, packages_root.parent, COMFY_ALIASES_MAX_BYTES, filename)
            raw_registry = _object(
                _decode_json(data, sidecar, filename),
                filename,
            )
            raw_values = _array(raw_registry.get("records"), f"{filename}.records")
            raw_records = {}
            for index, value in enumerate(raw_values):
                raw_record = _object(value, f"{filename}.records[{index}]")
                raw_records[_string(raw_record.get("id"), "record id")] = raw_record
            sidecars.append(
                (
                    sidecar.relative_to(packages_root.parent).as_posix(),
                    f"sha256:{hashlib.sha256(data).hexdigest()}",
                )
            )
            for record in registry.records:
                source_name = (
                    cast("Any", record).source.node_class
                    if registry_kind == "alias"
                    else cast("Any", record).source.name
                )
                source_key = (registry_kind, cast("Any", record).source.pack, source_name)
                if record.id in seen_ids or source_key in seen_sources:
                    raise CoverageError(f"duplicate maintained registry mapping: {record.id}")
                seen_ids.add(record.id)
                seen_sources.add(source_key)
                family = record.family
                refusal = registry_kind == "alias" and all(
                    _case_is_refusal(case) for case in record.replacement.cases
                )
                mappings.append(
                    RegistryMapping(
                        registry_id=record.id,
                        record_digest=(
                            "sha256:"
                            + hashlib.sha256(canonical_json(raw_records[record.id])).hexdigest()
                        ),
                        registry_kind=registry_kind,
                        mapping_kind=record.mapping_kind,
                        source_pack=record.source.pack,
                        source_name=source_name,
                        revision=record.source.revision,
                        carrier=record.carrier,
                        target_provider=manifest.name,
                        tier=record.confidence.tier,
                        evidence=record.confidence.evidence,
                        tolerances=tuple(
                            _tolerance_wire(tolerance) for tolerance in record.confidence.tolerances
                        ),
                        family_id=family.id if family is not None else None,
                        family_provider=family.provider if family is not None else None,
                        refusal=refusal,
                    )
                )
    return RegistryCatalog(
        mappings=tuple(sorted(mappings, key=lambda item: item.registry_id)),
        available_providers=frozenset(providers),
        sidecars=tuple(sorted(sidecars)),
    )


def load_download_snapshot(path: Path) -> DownloadSnapshot:
    document = _load_json(path, path.parent, MAX_TEMPLATE_BYTES, "registry snapshot")
    root = _fields(
        document,
        "registry snapshot",
        {"format", "capturedAt", "source", "totalPacks", "totalDownloads", "cutlines", "packs"},
    )
    if root["format"] != SNAPSHOT_FORMAT:
        raise CoverageError(f"unsupported registry snapshot format: {root['format']!r}")
    total_packs = root["totalPacks"]
    total_downloads = root["totalDownloads"]
    if type(total_packs) is not int or total_packs < 900:
        raise CoverageError("registry snapshot totalPacks must be an integer of at least 900")
    if type(total_downloads) is not int or total_downloads <= 0:
        raise CoverageError("registry snapshot totalDownloads must be a positive integer")
    packs = _array(root["packs"], "registry snapshot.packs")
    if len(packs) != 900:
        raise CoverageError("registry snapshot must contain exactly the top 900 packs")
    ranks: dict[str, tuple[int, str, int]] = {}
    previous: tuple[int, str, str] | None = None
    cumulative: list[int] = []
    running = 0
    for index, raw in enumerate(packs, start=1):
        pack = _fields(raw, f"registry snapshot.packs[{index - 1}]", {"rank", "id", "downloads"})
        pack_id = _string(pack["id"], f"registry snapshot.packs[{index - 1}].id", 128)
        downloads = pack["downloads"]
        if pack["rank"] != index or type(downloads) is not int or downloads < 0:
            raise CoverageError(
                f"registry snapshot pack rank/downloads are invalid at rank {index}"
            )
        canonical = pack_id.casefold()
        if canonical in ranks:
            raise CoverageError(f"registry snapshot has duplicate pack id: {pack_id}")
        order = (-downloads, canonical, pack_id)
        if previous is not None and order < previous:
            raise CoverageError("registry snapshot packs are not in canonical download order")
        previous = order
        ranks[canonical] = (index, pack_id, downloads)
        running += downloads
        cumulative.append(running)
    cutlines = _array(root["cutlines"], "registry snapshot.cutlines")
    if len(cutlines) != 2:
        raise CoverageError("registry snapshot must contain exactly two cutlines")
    expected = {355: cumulative[354], 900: cumulative[899]}
    actual: dict[int, int] = {}
    for index, raw in enumerate(cutlines):
        cutline = _fields(
            raw,
            f"registry snapshot.cutlines[{index}]",
            {"rank", "cumulativeDownloads"},
        )
        rank = cutline["rank"]
        downloads = cutline["cumulativeDownloads"]
        if type(rank) is not int or type(downloads) is not int:
            raise CoverageError("registry snapshot cutlines must contain integer values")
        if rank in actual:
            raise CoverageError(f"registry snapshot has duplicate rank {rank} cutline")
        actual[rank] = downloads
    if actual != expected or total_downloads < cumulative[-1]:
        raise CoverageError("registry snapshot cutlines do not match pack downloads")
    return DownloadSnapshot(
        captured_at=_string(root["capturedAt"], "registry snapshot.capturedAt", 64),
        source=_string(root["source"], "registry snapshot.source", 512),
        total_packs=total_packs,
        total_downloads=total_downloads,
        rank_355_downloads=expected[355],
        rank_900_downloads=expected[900],
        ranks=ranks,
    )


def load_source_parity_baseline(path: Path, packages_root: Path) -> dict[str, Any]:
    document = _load_json(path, path.parent, MAX_TEMPLATE_BYTES, "source parity baseline")
    root = _fields(
        document,
        "source parity baseline",
        {
            "format",
            "scope",
            "parityUnit",
            "excludedFromParityCounts",
            "measured",
            "grandfatheredUnreceiptedDigest",
            "excludedNativePacks",
        },
    )
    if root["format"] != SOURCE_PARITY_BASELINE_FORMAT:
        raise CoverageError(f"unsupported source parity baseline format: {root['format']!r}")
    if root["scope"] != "maintained-comfy-mapping-records":
        raise CoverageError("source parity baseline has an unsupported scope")
    if root["parityUnit"] != "unique-mapping-with-verified-passing-receipt":
        raise CoverageError("source parity baseline has an unsupported parity unit")
    excluded_counts = _array(
        root["excludedFromParityCounts"], "source parity baseline.excludedFromParityCounts"
    )
    if excluded_counts != [
        "maintained-native-refusals",
        "mapped-workflow-occurrences",
        "translation-ready-workflows",
    ]:
        raise CoverageError("source parity baseline must exclude translation coverage counts")

    measured = _fields(
        root["measured"],
        "source parity baseline.measured",
        {
            "translationDeclarations",
            "receiptBackedMappings",
            "unreceiptedMappings",
            "passingReceiptCases",
        },
    )
    for key, value in measured.items():
        if type(value) is not int or value < 0:
            raise CoverageError(
                f"source parity baseline.measured.{key} must be a nonnegative integer"
            )
    if (
        measured["translationDeclarations"] - measured["receiptBackedMappings"]
        != measured["unreceiptedMappings"]
        or measured["passingReceiptCases"] < measured["receiptBackedMappings"]
    ):
        raise CoverageError("source parity baseline measured counts are inconsistent")
    grandfathered_digest = root["grandfatheredUnreceiptedDigest"]
    if (
        not isinstance(grandfathered_digest, str)
        or len(grandfathered_digest) != 71
        or not grandfathered_digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in grandfathered_digest[7:])
    ):
        raise CoverageError("source parity baseline has an invalid grandfathered mapping digest")

    exclusions = _array(root["excludedNativePacks"], "source parity baseline.excludedNativePacks")
    seen_packages: set[str] = set()
    for index, value in enumerate(exclusions):
        where = f"source parity baseline.excludedNativePacks[{index}]"
        exclusion = _fields(value, where, {"package", "disposition", "reason"})
        package = _string(exclusion["package"], f"{where}.package", 128)
        disposition = _string(exclusion["disposition"], f"{where}.disposition", 128)
        _string(exclusion["reason"], f"{where}.reason", 1_024)
        if package in seen_packages:
            raise CoverageError(f"source parity baseline repeats excluded package {package}")
        if disposition not in {
            "development-only-no-comfy-execution-counterpart",
            "remote-service-no-local-comfy-execution-counterpart",
            "schema-only-no-comfy-execution-counterpart",
        }:
            raise CoverageError(
                f"source parity baseline has unsupported disposition {disposition!r}"
            )
        manifest_path = packages_root / package / "dinkster-pack.toml"
        if not manifest_path.is_file():
            raise CoverageError(f"source parity baseline excluded package is absent: {package}")
        manifest = load_manifest(manifest_path)
        if manifest.name != package:
            raise CoverageError(
                f"source parity baseline package name does not match {manifest_path}"
            )
        if manifest.comfy_aliases is not None or manifest.comfy_groups is not None:
            raise CoverageError(f"source parity baseline cannot exclude mapped package {package}")
        seen_packages.add(package)
    return root


def _rank_band(pack: str | None, snapshot: DownloadSnapshot) -> str:
    if pack is None:
        return "unresolved"
    if pack == "comfy-core":
        return "core"
    ranked = snapshot.ranks.get(pack.casefold())
    if ranked is None:
        return "outside-top-900"
    return "top-355" if ranked[0] <= 355 else "rank-356-900"


def _complete_counts(values: Counter[str], keys: Iterable[str]) -> dict[str, int]:
    return {key: values[key] for key in keys}


def _receipt_paths(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    _ordinary_directory(root, "receipt")
    paths: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    count = 0
    while stack:
        directory, depth = stack.pop()
        if depth > MAX_RECEIPT_TREE_DEPTH:
            raise CoverageError(f"receipt tree exceeds depth {MAX_RECEIPT_TREE_DEPTH}")
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise CoverageError(f"cannot read receipt directory {directory}: {error}") from error
        for entry in entries:
            count += 1
            if count > MAX_RECEIPT_TREE_ITEMS:
                raise CoverageError(f"receipt tree exceeds {MAX_RECEIPT_TREE_ITEMS} items")
            path = Path(entry.path)
            try:
                info = path.lstat()
            except OSError as error:
                raise CoverageError(f"cannot inspect receipt tree item {path}: {error}") from error
            if path.is_symlink() or _is_reparse_point(info):
                raise CoverageError(f"receipt tree contains a link or reparse point: {path}")
            if stat.S_ISDIR(info.st_mode):
                stack.append((path, depth + 1))
            elif not stat.S_ISREG(info.st_mode):
                raise CoverageError(f"receipt tree contains a special file: {path}")
            elif path.name.endswith(".receipt.json"):
                paths.append(path)
    return tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))


def _receipt_summary(
    root: Path,
    catalog: RegistryCatalog,
) -> dict[str, object]:
    records = {record.registry_id: record for record in catalog.mappings}
    cases: list[dict[str, object]] = []
    seen_cases: set[str] = set()
    for path in _receipt_paths(root):
        receipt = verify_receipt(load_receipt(path), root)
        case_id = cast("str", receipt["caseId"])
        if case_id in seen_cases:
            raise CoverageError(f"duplicate confidence receipt caseId: {case_id}")
        seen_cases.add(case_id)
        mapping = cast("dict[str, Any]", receipt["mapping"])
        registry_id = cast("str", mapping["registryId"])
        record = records.get(registry_id)
        if record is None:
            raise CoverageError(
                f"confidence receipt names unknown maintained record: {registry_id}"
            )
        if record.refusal:
            raise CoverageError(f"confidence receipt cannot claim parity for a refusal: {case_id}")
        source = cast("dict[str, object]", mapping["source"])
        target = cast("dict[str, object]", mapping["target"])
        expected_source = {
            "pack": record.source_pack,
            "name": record.source_name,
            "revision": record.revision,
        }
        if (
            mapping["mappingKind"] != record.mapping_kind
            or mapping["tier"] != record.tier
            or any(source[key] != value for key, value in expected_source.items())
            or target
            != {
                "kind": "node" if record.registry_kind == "alias" else "group",
                "id": record.carrier,
            }
        ):
            raise CoverageError(f"confidence receipt does not match maintained record: {case_id}")
        comparison = cast("dict[str, Any]", receipt["comparison"])
        parameters = cast("dict[str, Any]", receipt["parameters"])
        if parameters.get("mappingDigest") != record.record_digest:
            raise CoverageError(f"confidence receipt has a stale mapping digest: {case_id}")
        receipt_tolerances: dict[tuple[str, str], float] = {}
        for item in cast("list[dict[str, Any]]", comparison.get("tolerances", [])):
            key = (cast("str", item["metric"]), cast("str", item["operator"]))
            receipt_tolerances[key] = float(cast("int | float", item["value"]))
        declared_tolerances = {
            (metric, operator): value for metric, operator, value in record.tolerances
        }
        if set(receipt_tolerances) != set(declared_tolerances):
            raise CoverageError(f"confidence receipt tolerances do not match record: {case_id}")
        for key, value in receipt_tolerances.items():
            declared = declared_tolerances[key]
            if (key[1] == "<=" and value > declared) or (key[1] == ">=" and value < declared):
                raise CoverageError(f"confidence receipt widens a maintained tolerance: {case_id}")
        cases.append(
            {
                "caseId": case_id,
                "mappingKind": record.mapping_kind,
                "pass": receipt["pass"],
                "path": path.relative_to(root).as_posix(),
                "registryId": registry_id,
                "registryKind": record.registry_kind,
                "source": expected_source,
                "target": target,
                "tier": record.tier,
                "tolerances": [
                    {"metric": metric, "operator": operator, "value": value}
                    for metric, operator, value in record.tolerances
                ],
            }
        )
    kinds = Counter(cast("str", case["mappingKind"]) for case in cases)
    tiers = Counter(cast("str", case["tier"]) for case in cases)
    kind_tiers: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        kind_tiers[cast("str", case["mappingKind"])][cast("str", case["tier"])] += 1
    passing_records = {cast("str", case["registryId"]) for case in cases if case["pass"] is True}
    passing_kinds = Counter(records[record_id].mapping_kind for record_id in passing_records)
    return {
        "cases": sorted(cases, key=lambda item: cast("str", item["caseId"])),
        "countsByMappingKind": _complete_counts(kinds, MAPPING_KINDS),
        "countsByTier": _complete_counts(tiers, CONFIDENCE_TIERS),
        "countsByTierAndMappingKind": {
            kind: _complete_counts(kind_tiers[kind], CONFIDENCE_TIERS) for kind in MAPPING_KINDS
        },
        "failed": sum(case["pass"] is False for case in cases),
        "passing": sum(case["pass"] is True for case in cases),
        "recordsWithPassingReceiptsByMappingKind": _complete_counts(passing_kinds, MAPPING_KINDS),
        "recordsWithPassingReceipts": sorted(passing_records),
        "total": len(cases),
    }


def _source_parity_summary(
    catalog: RegistryCatalog,
    receipts: Mapping[str, object],
    baseline: Mapping[str, object] | None,
) -> dict[str, object]:
    backed_by_kind = cast("Mapping[str, int]", receipts["recordsWithPassingReceiptsByMappingKind"])
    receipt_backed = sum(backed_by_kind.values())
    mappings = tuple(record for record in catalog.mappings if not record.refusal)
    refusals = tuple(record for record in catalog.mappings if record.refusal)
    declarations = len(mappings)
    backed_records = set(cast("Sequence[str]", receipts["recordsWithPassingReceipts"]))
    unreceipted = {
        record.registry_id: record.record_digest
        for record in mappings
        if record.registry_id not in backed_records
    }
    unreceipted_digest = "sha256:" + hashlib.sha256(canonical_json(unreceipted)).hexdigest()
    summary: dict[str, object] = {
        "scope": "maintained-comfy-mapping-records",
        "parityUnit": "unique-mapping-with-verified-passing-receipt",
        "translationDeclarations": declarations,
        "receiptBackedMappings": receipt_backed,
        "unreceiptedMappings": declarations - receipt_backed,
        "unreceiptedMappingDigest": unreceipted_digest,
        "refusedMappings": len(refusals),
        "refusedMappingIds": [record.registry_id for record in refusals],
        "passingReceiptCases": receipts["passing"],
        "failingReceiptCases": receipts["failed"],
        "excludedFromParityCounts": [
            "maintained-native-refusals",
            "mapped-workflow-occurrences",
            "translation-ready-workflows",
        ],
    }
    if baseline is None:
        return summary
    measured = cast("Mapping[str, int]", baseline["measured"])
    if cast("int", receipts["failed"]):
        raise CoverageError("source parity baseline forbids failing committed receipts")
    if receipt_backed < measured["receiptBackedMappings"]:
        raise CoverageError(
            "source parity receipt-backed mapping count fell below the measured baseline"
        )
    if declarations - receipt_backed > measured["unreceiptedMappings"]:
        raise CoverageError("source parity unreceipted mapping debt exceeds the measured baseline")
    if unreceipted_digest != baseline["grandfatheredUnreceiptedDigest"]:
        raise CoverageError(
            "source parity baseline does not match the current unreceipted mappings: "
            f"{declarations} declarations, {receipt_backed} receipt-backed, "
            f"{declarations - receipt_backed} unreceipted, digest {unreceipted_digest}"
        )
    summary["baseline"] = dict(baseline)
    return summary


def _mapping_wire(
    record: RegistryMapping,
    snapshot: DownloadSnapshot,
    available_providers: frozenset[str],
) -> dict[str, object]:
    result: dict[str, object] = {
        "evidence": list(record.evidence),
        "id": record.registry_id,
        "mappingKind": record.mapping_kind,
        "semanticDigest": record.record_digest,
        "registryKind": record.registry_kind,
        "source": {
            "name": record.source_name,
            "pack": record.source_pack,
            "rankBand": _rank_band(record.source_pack, snapshot),
            "revision": record.revision,
        },
        "target": {
            "available": not record.refusal and record.target_provider in available_providers,
            "id": record.carrier,
            "provider": record.target_provider,
        },
        "tier": record.tier,
        "tolerances": [
            {"metric": metric, "operator": operator, "value": value}
            for metric, operator, value in record.tolerances
        ],
    }
    if record.family_id is not None:
        result["family"] = {"id": record.family_id, "provider": record.family_provider}
    if record.refusal:
        result["refusal"] = True
    return result


def build_report(
    *,
    workflows: tuple[Workflow, ...],
    ignored_files: tuple[str, ...],
    catalog: RegistryCatalog,
    snapshot: DownloadSnapshot,
    receipts_root: Path,
    source_parity_baseline: Mapping[str, object] | None = None,
) -> dict[str, object]:
    aliases = tuple(item for item in catalog.mappings if item.registry_kind == "alias")
    alias_by_source = {(item.source_pack, item.source_name): item for item in aliases}
    alias_packs: dict[str, set[str]] = defaultdict(set)
    for item in aliases:
        alias_packs[item.source_name].add(item.source_pack)
    provider_hints: dict[str, set[str]] = defaultdict(set)
    for workflow in workflows:
        for node in workflow.nodes:
            if node.explicit_pack is not None:
                provider_hints[node.node_type].add(node.explicit_pack)

    status_totals: Counter[str] = Counter()
    kind_occurrences: Counter[str] = Counter()
    tier_occurrences: Counter[str] = Counter()
    kind_tier_occurrences: dict[str, Counter[str]] = defaultdict(Counter)
    source_resolutions: Counter[str] = Counter()
    rank_statuses: dict[str, Counter[str]] = defaultdict(Counter)
    workflow_rows: list[dict[str, object]] = []
    blocker_totals: dict[tuple[str, str, str | None, str], list[object]] = {}
    evidence_state_workflows: Counter[str] = Counter()
    evidence_tier_workflows: Counter[str] = Counter()
    ready = 0
    supported = 0
    for workflow in workflows:
        local_statuses: Counter[str] = Counter()
        local_kinds: Counter[str] = Counter()
        local_tiers: Counter[str] = Counter()
        local_kind_tiers: dict[str, Counter[str]] = defaultdict(Counter)
        local_rows: dict[tuple[object, ...], tuple[dict[str, object], int]] = {}
        for node in workflow.nodes:
            disposition: dict[str, object] = {"nodeType": node.node_type}
            mapping: RegistryMapping | None = None
            pack: str | None = None
            resolution: str
            if node.node_type in STRUCTURAL_NODE_TYPES or node.node_type in workflow.subgraph_ids:
                status = "structural"
                resolution = "structural"
                evidence_state = "not-applicable"
                evidence_tier: str | None = None
                reason = (
                    "subgraph-instance"
                    if node.node_type in workflow.subgraph_ids
                    else "workflow-structure"
                )
                band = "unresolved"
                disposition["reason"] = reason
            else:
                hints = provider_hints.get(node.node_type, set())
                candidates = alias_packs.get(node.node_type, set())
                if node.explicit_pack is not None:
                    pack = node.explicit_pack
                    resolution = "explicit"
                elif len(hints) == 1:
                    pack = next(iter(hints))
                    resolution = "workflow-evidence"
                elif len(hints) > 1:
                    resolution = "ambiguous"
                elif len(candidates) == 1:
                    pack = next(iter(candidates))
                    resolution = "registry-evidence"
                elif len(candidates) > 1:
                    resolution = "ambiguous"
                else:
                    resolution = "unresolved"
                band = _rank_band(pack, snapshot)
                if pack is None:
                    status = "unsupported"
                    evidence_state = "absent"
                    evidence_tier = None
                    if resolution == "ambiguous":
                        providers = sorted(hints or candidates)
                        disposition["providers"] = providers
                        disposition["reason"] = "ambiguous-source-pack"
                    else:
                        disposition["reason"] = "source-pack-unknown"
                else:
                    disposition["sourcePack"] = pack
                    mapping = alias_by_source.get((pack, node.node_type))
                    if mapping is None:
                        status = "quarantine"
                        evidence_state = "absent"
                        evidence_tier = None
                        disposition["reason"] = "no-maintained-native-alias"
                    elif mapping.refusal:
                        status = "unsupported"
                        evidence_state = "refused"
                        evidence_tier = None
                        disposition["reason"] = "maintained-native-refusal"
                        disposition["registryId"] = mapping.registry_id
                    elif mapping.target_provider not in catalog.available_providers:
                        status = "unavailable"
                        evidence_state = "refused"
                        evidence_tier = None
                        disposition["reason"] = "native-provider-unavailable"
                    else:
                        status = "mapped"
                        evidence_state = "proven"
                        evidence_tier = "T1"
                        disposition.update(
                            {
                                "mappingKind": mapping.mapping_kind,
                                "registryId": mapping.registry_id,
                                "target": mapping.carrier,
                                "tier": mapping.tier,
                            }
                        )
                        local_kinds[mapping.mapping_kind] += 1
                        local_tiers[mapping.tier] += 1
                        local_kind_tiers[mapping.mapping_kind][mapping.tier] += 1
                        kind_occurrences[mapping.mapping_kind] += 1
                        tier_occurrences[mapping.tier] += 1
                        kind_tier_occurrences[mapping.mapping_kind][mapping.tier] += 1
            disposition.update(
                {
                    "evidenceState": evidence_state,
                    "highestProvenTier": evidence_tier,
                    "rankBand": band,
                    "sourceResolution": resolution,
                    "status": status,
                    "supported": evidence_tier in {"T2", "T3", "T4"},
                }
            )
            key = (
                status,
                node.node_type,
                pack,
                disposition.get("reason"),
                disposition.get("registryId"),
                resolution,
            )
            row, count = local_rows.get(key, (disposition, 0))
            local_rows[key] = (row, count + 1)
            local_statuses[status] += 1
            status_totals[status] += 1
            source_resolutions[resolution] += 1
            rank_statuses[band][status] += 1
            if status in {"quarantine", "unavailable", "unsupported"}:
                blocker_key = (
                    status,
                    node.node_type,
                    pack,
                    cast("str", disposition["reason"]),
                )
                blocker = blocker_totals.setdefault(blocker_key, [0, set()])
                blocker[0] = cast("int", blocker[0]) + 1
                cast("set[str]", blocker[1]).add(workflow.path)
        translation_ready = not any(
            local_statuses[status] for status in ("quarantine", "unavailable", "unsupported")
        )
        ready += int(translation_ready)
        dispositions = []
        for row, count in local_rows.values():
            dispositions.append({**row, "count": count})
        executable = [item for item in dispositions if item["status"] != "structural"]
        if any(item["evidenceState"] == "refused" for item in executable):
            workflow_evidence_state = "refused"
            workflow_evidence_tier = None
        elif any(item["evidenceState"] == "absent" for item in executable):
            workflow_evidence_state = "absent"
            workflow_evidence_tier = None
        elif not executable:
            workflow_evidence_state = "unverified"
            workflow_evidence_tier = None
        else:
            workflow_evidence_state = "proven"
            workflow_evidence_tier = min(
                (cast("str", disposition["highestProvenTier"]) for disposition in executable),
                key=EVIDENCE_TIERS.index,
            )
        workflow_supported = workflow_evidence_state == "proven" and workflow_evidence_tier in {
            "T2",
            "T3",
            "T4",
        }
        weakest = [
            {
                "evidenceState": item["evidenceState"],
                "highestProvenTier": item["highestProvenTier"],
                "nodeType": item["nodeType"],
                "sourcePack": item.get("sourcePack"),
            }
            for item in executable
            if (
                workflow_evidence_state != "proven"
                and item["evidenceState"] == workflow_evidence_state
                or workflow_evidence_state == "proven"
                and item["highestProvenTier"] == workflow_evidence_tier
            )
        ]
        weakest.sort(
            key=lambda item: (
                cast("str", item["nodeType"]),
                cast("str", item["sourcePack"] or ""),
            )
        )
        evidence_state_workflows[workflow_evidence_state] += 1
        if workflow_evidence_tier is not None:
            evidence_tier_workflows[workflow_evidence_tier] += 1
        supported += int(workflow_supported)
        workflow_rows.append(
            {
                "capabilityEvidence": {
                    "highestProvenTier": workflow_evidence_tier,
                    "state": workflow_evidence_state,
                    "supported": workflow_supported,
                    "weakestCapabilities": weakest,
                },
                "confidenceTiers": _complete_counts(local_tiers, CONFIDENCE_TIERS),
                "confidenceTiersByMappingKind": {
                    kind: _complete_counts(local_kind_tiers[kind], CONFIDENCE_TIERS)
                    for kind in MAPPING_KINDS
                },
                "dispositions": sorted(
                    dispositions,
                    key=lambda item: (
                        cast("str", item["status"]),
                        cast("str", item["nodeType"]),
                        cast("str", item.get("sourcePack", "")),
                    ),
                ),
                "mappingKinds": _complete_counts(local_kinds, MAPPING_KINDS),
                "nodeCount": len(workflow.nodes),
                "path": workflow.path,
                "statusCounts": _complete_counts(local_statuses, STATUSES),
                "subgraphCount": workflow.subgraph_count,
                "translationReady": translation_ready,
            }
        )

    declaration_kinds = Counter(item.mapping_kind for item in catalog.mappings)
    declaration_tiers = Counter(item.tier for item in catalog.mappings)
    declaration_kind_tiers: dict[str, Counter[str]] = defaultdict(Counter)
    for item in catalog.mappings:
        declaration_kind_tiers[item.mapping_kind][item.tier] += 1
    declaration_registry_kinds = Counter(item.registry_kind for item in catalog.mappings)
    blockers = [
        {
            "nodeType": node_type,
            "occurrences": cast("int", values[0]),
            "reason": reason,
            "sourcePack": pack,
            "status": status,
            "workflows": len(cast("set[str]", values[1])),
        }
        for (status, node_type, pack, reason), values in blocker_totals.items()
    ]
    blockers.sort(
        key=lambda item: (
            -cast("int", item["occurrences"]),
            cast("str", item["status"]),
            cast("str", item["nodeType"]),
            cast("str", item["sourcePack"] or ""),
        )
    )
    receipts = _receipt_summary(receipts_root, catalog)
    return {
        "coverage": {
            "blockers": blockers,
            "confidenceTierOccurrences": _complete_counts(tier_occurrences, CONFIDENCE_TIERS),
            "confidenceTierOccurrencesByMappingKind": {
                kind: _complete_counts(kind_tier_occurrences[kind], CONFIDENCE_TIERS)
                for kind in MAPPING_KINDS
            },
            "mappingKindOccurrences": _complete_counts(kind_occurrences, MAPPING_KINDS),
            "rankBands": {
                band: _complete_counts(rank_statuses[band], STATUSES) for band in RANK_BANDS
            },
            "sourceResolutionCounts": dict(sorted(source_resolutions.items())),
            "statusCounts": _complete_counts(status_totals, STATUSES),
            "supportedWorkflows": supported,
            "translationReadyWorkflows": ready,
            "workflowEvidenceStates": _complete_counts(evidence_state_workflows, EVIDENCE_STATES),
            "workflowEvidenceTiers": _complete_counts(evidence_tier_workflows, EVIDENCE_TIERS),
        },
        "corpus": {
            "ignoredJsonFiles": list(ignored_files),
            "jsonFiles": len(workflows) + len(ignored_files),
            "nodes": sum(len(workflow.nodes) for workflow in workflows),
            "subgraphs": sum(workflow.subgraph_count for workflow in workflows),
            "workflows": len(workflows),
        },
        "format": FORMAT,
        "inputs": {
            "comfyuiRevision": COMFYUI_REVISION,
            "registrySnapshot": {
                "capturedAt": snapshot.captured_at,
                "cutlines": [
                    {"cumulativeDownloads": snapshot.rank_355_downloads, "rank": 355},
                    {"cumulativeDownloads": snapshot.rank_900_downloads, "rank": 900},
                ],
                "source": snapshot.source,
                "totalDownloads": snapshot.total_downloads,
                "totalPacks": snapshot.total_packs,
            },
            "templateRevision": TEMPLATE_REVISION,
        },
        "providerEvidence": {
            "ambiguousClasses": sum(len(values) > 1 for values in provider_hints.values()),
            "classes": len(provider_hints),
            "uniqueClasses": sum(len(values) == 1 for values in provider_hints.values()),
        },
        "receipts": receipts,
        "registry": {
            "declarationsByKind": _complete_counts(declaration_kinds, MAPPING_KINDS),
            "declarationsByRegistryKind": {
                "alias": declaration_registry_kinds["alias"],
                "group": declaration_registry_kinds["group"],
            },
            "declarationsByTier": _complete_counts(declaration_tiers, CONFIDENCE_TIERS),
            "declarationsByTierAndMappingKind": {
                kind: _complete_counts(declaration_kind_tiers[kind], CONFIDENCE_TIERS)
                for kind in MAPPING_KINDS
            },
            "records": [
                _mapping_wire(item, snapshot, catalog.available_providers)
                for item in catalog.mappings
            ],
            "sidecars": [{"path": path, "sha256": digest} for path, digest in catalog.sidecars],
        },
        "sourceParity": _source_parity_summary(catalog, receipts, source_parity_baseline),
        "workflows": workflow_rows,
    }


def _supported_document_paths(path: Path) -> tuple[Path, ...]:
    if path.is_dir():
        paths = list(path.glob("*.md"))
    else:
        paths = [path]
        if path.name == "SUPPORTED.md":
            paths.extend((path.parent / "docs" / "supported").glob("*.md"))
    boundary_name = "implemented-capabilities-with-limited-or-specialized-exposure.md"
    return tuple(sorted(paths, key=lambda item: (item.name == boundary_name, item.as_posix())))


def lint_supported_claims(source: EvidenceSource, path: Path, repo_root: Path) -> None:
    documents = []
    for document_path in _supported_document_paths(path):
        data = _read_regular(
            document_path, repo_root, MAX_SOURCE_BYTES, "supported capability document"
        )
        try:
            documents.append(data.decode("utf-8"))
        except UnicodeError as error:
            raise CoverageError(
                f"supported capability document is not UTF-8: {document_path}: {error}"
            ) from error
    text = "\n".join(documents)
    implemented_heading = "## Implemented capabilities with limited or specialized exposure"
    boundary = text.find(implemented_heading)
    if boundary < 0:
        raise CoverageError(f"supported capability documents lack {implemented_heading!r}")
    markers = re.findall(r"<!-- capability:([a-z0-9_.-]+) -->", text)
    counts = Counter(markers)
    expected = {item.family_id for item in source.model_families if item.exposure != "unlisted"}
    if set(markers) != expected or any(count != 1 for count in counts.values()):
        missing = sorted(expected - set(markers))
        unexpected = sorted(set(markers) - expected)
        duplicate = sorted(marker for marker, count in counts.items() if count != 1)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        if duplicate:
            details.append(f"duplicate {', '.join(duplicate)}")
        raise CoverageError(f"supported capability document markers drifted: {'; '.join(details)}")
    for item in source.model_families:
        if item.exposure == "unlisted":
            continue
        if item.state != "proven" or item.tier not in {"T2", "T3", "T4"}:
            raise CoverageError(
                "supported capability documents cannot claim "
                f"{item.family_id} without T2 or stronger evidence"
            )
        marker_position = text.index(f"<!-- capability:{item.family_id} -->")
        if item.exposure == "supported" and marker_position > boundary:
            raise CoverageError(f"supported family is listed as not exposed: {item.family_id}")
        if item.exposure == "implemented" and marker_position < boundary:
            raise CoverageError(f"unexposed family is listed as supported: {item.family_id}")


def build_template_features(
    workflows: tuple[Workflow, ...], source: EvidenceSource
) -> dict[str, object]:
    paths: dict[str, list[dict[str, object]]] = {"adapter": [], "lora": []}
    workflow_by_path = {workflow.path: workflow for workflow in workflows}
    for workflow in workflows:
        node_types = {node.node_type for node in workflow.nodes}
        for node_type in node_types:
            lower = node_type.casefold()
            if (
                "lora" in lower
                or "adapter" in lower
                or "controlnet" in lower
                or node_type == "ModelPatchLoader"
            ) and node_type not in source.lora_node_types | source.adapter_node_types:
                raise CoverageError(
                    f"template feature node type is not classified as LoRA or adapter: {node_type}"
                )
        for kind, configured in (
            ("lora", source.lora_node_types),
            ("adapter", source.adapter_node_types),
        ):
            matched = sorted(node_types & configured)
            if matched:
                paths[kind].append({"nodeTypes": matched, "path": workflow.path})

    low_step_rows: list[dict[str, object]] = []
    for declared in source.low_step_lora_paths:
        workflow = workflow_by_path.get(declared.path)
        if workflow is None:
            raise CoverageError(f"low-step LoRA template is absent: {declared.path}")
        lora_nodes = [
            node
            for node in workflow.nodes
            if node.node_id == declared.lora_node_id and node.node_type == declared.lora_node_type
        ]
        sampler_nodes = [
            node
            for node in workflow.nodes
            if node.node_id == declared.sampler_node_id
            and node.node_type == declared.sampler_node_type
        ]
        if len(lora_nodes) != 1 or len(sampler_nodes) != 1:
            raise CoverageError(f"low-step LoRA template node evidence drifted: {declared.path}")
        sampler = sampler_nodes[0]
        if (
            declared.step_widget_index >= len(sampler.widgets)
            or sampler.widgets[declared.step_widget_index] != declared.steps
        ):
            raise CoverageError(f"low-step LoRA template step evidence drifted: {declared.path}")
        low_step_rows.append(
            {
                "loraNode": {
                    "id": declared.lora_node_id,
                    "type": declared.lora_node_type,
                },
                "path": declared.path,
                "samplerNode": {
                    "id": declared.sampler_node_id,
                    "stepWidgetIndex": declared.step_widget_index,
                    "steps": declared.steps,
                    "type": declared.sampler_node_type,
                },
            }
        )
    return {
        "adapter": paths["adapter"],
        "lora": paths["lora"],
        "lowStepLora": low_step_rows,
    }


def _model_capabilities(source: EvidenceSource) -> list[dict[str, object]]:
    display_names = {family.id: family.display_name for family in builtin_families()}
    return [
        {
            "capabilityId": f"model-family:{item.family_id}",
            "evidence": [
                {"kind": evidence.kind, "selector": evidence.selector} for evidence in item.evidence
            ],
            "exposure": item.exposure,
            "highestProvenTier": item.tier,
            "kind": "model-family",
            "native": {
                "displayName": display_names[item.family_id],
                "familyId": item.family_id,
            },
            "source": {
                "identifiers": list(item.source_identifiers),
                "repository": "https://github.com/Comfy-Org/ComfyUI",
                "revision": COMFYUI_REVISION,
            },
            "state": item.state,
            "supported": item.exposure == "supported"
            and item.state == "proven"
            and item.tier in {"T2", "T3", "T4"},
        }
        for item in source.model_families
    ]


def _registry_capabilities(
    catalog: RegistryCatalog, report: Mapping[str, object], repo_root: Path
) -> list[dict[str, object]]:
    occurrence_counts: Counter[str] = Counter()
    workflow_counts: dict[str, set[str]] = defaultdict(set)
    for workflow in cast("list[dict[str, Any]]", report["workflows"]):
        for disposition in cast("list[dict[str, Any]]", workflow["dispositions"]):
            registry_id = disposition.get("registryId")
            if isinstance(registry_id, str):
                occurrence_counts[registry_id] += cast("int", disposition["count"])
                workflow_counts[registry_id].add(cast("str", workflow["path"]))
    validated: set[str] = set()
    result: list[dict[str, object]] = []
    for item in catalog.mappings:
        for selector in item.evidence:
            if selector not in validated:
                _validate_evidence_selector(selector, repo_root, require_collected_test=True)
                validated.add(selector)
        available = not item.refusal and item.target_provider in catalog.available_providers
        result.append(
            {
                "capabilityId": item.registry_id,
                "evidence": [
                    {
                        "kind": (
                            "compatibility-test" if item.registry_kind == "alias" else "schema-test"
                        ),
                        "selector": selector,
                    }
                    for selector in item.evidence
                ],
                "highestProvenTier": ("T1" if item.registry_kind == "alias" else "T0")
                if available
                else None,
                "kind": (
                    "compatibility-alias"
                    if item.registry_kind == "alias"
                    else "compatibility-group"
                ),
                "observed": {
                    "occurrences": occurrence_counts[item.registry_id],
                    "workflows": len(workflow_counts[item.registry_id]),
                },
                "source": {
                    "identifiers": [item.source_name],
                    "pack": item.source_pack,
                    "repository": (
                        "https://github.com/Comfy-Org/ComfyUI"
                        if item.source_pack == "comfy-core"
                        else f"comfy-registry:{item.source_pack}"
                    ),
                    "revision": item.revision,
                },
                "state": "proven" if available else "refused",
                "supported": False,
                "target": {
                    "available": available,
                    "id": item.carrier,
                    "provider": item.target_provider,
                },
            }
        )
    return result


def _absent_template_capabilities(
    report: Mapping[str, object], comfyui: ComfyUISource
) -> list[dict[str, object]]:
    aggregates: dict[tuple[str | None, str], dict[str, object]] = {}
    for workflow in cast("list[dict[str, Any]]", report["workflows"]):
        for disposition in cast("list[dict[str, Any]]", workflow["dispositions"]):
            if disposition["evidenceState"] != "absent":
                continue
            pack = cast("str | None", disposition.get("sourcePack"))
            node_type = cast("str", disposition["nodeType"])
            key = (pack, node_type)
            aggregate = aggregates.setdefault(
                key,
                {
                    "occurrences": 0,
                    "reasons": set(),
                    "workflows": set(),
                },
            )
            aggregate["occurrences"] = cast("int", aggregate["occurrences"]) + cast(
                "int", disposition["count"]
            )
            cast("set[str]", aggregate["reasons"]).add(cast("str", disposition["reason"]))
            cast("set[str]", aggregate["workflows"]).add(cast("str", workflow["path"]))
    result: list[dict[str, object]] = []
    for (pack, node_type), aggregate in aggregates.items():
        core = pack == "comfy-core"
        source: dict[str, object] = {
            "identifiers": [node_type],
            "pack": pack,
            "repository": (
                "https://github.com/Comfy-Org/ComfyUI"
                if core
                else "https://github.com/Comfy-Org/workflow_templates"
            ),
            "revision": COMFYUI_REVISION if core else TEMPLATE_REVISION,
        }
        if core:
            source["registeredAtSource"] = node_type in comfyui.node_ids
        result.append(
            {
                "capabilityId": f"template-node:{pack or 'unresolved'}/{node_type}",
                "evidence": [],
                "highestProvenTier": None,
                "kind": "template-node",
                "observed": {
                    "occurrences": aggregate["occurrences"],
                    "reasons": sorted(cast("set[str]", aggregate["reasons"])),
                    "workflows": len(cast("set[str]", aggregate["workflows"])),
                },
                "source": source,
                "state": "absent",
                "supported": False,
            }
        )
    return result


def build_evidence_ledger(
    *,
    report: Mapping[str, object],
    workflows: tuple[Workflow, ...],
    catalog: RegistryCatalog,
    source: EvidenceSource,
    comfyui: ComfyUISource,
    repo_root: Path,
) -> dict[str, object]:
    missing_aliases = sorted(
        item.source_name
        for item in catalog.mappings
        if item.registry_kind == "alias"
        and item.source_pack == "comfy-core"
        and item.source_name not in comfyui.node_ids
    )
    if missing_aliases:
        raise CoverageError(
            "maintained core aliases name nodes missing from pinned ComfyUI: "
            f"{', '.join(missing_aliases)}"
        )
    for model in source.model_families:
        missing_identifiers = sorted(set(model.source_identifiers) - comfyui.class_names)
        if missing_identifiers:
            raise CoverageError(
                f"model evidence {model.family_id} names missing ComfyUI source identifiers: "
                f"{', '.join(missing_identifiers)}"
            )
    capabilities = [
        *_model_capabilities(source),
        *_registry_capabilities(catalog, report, repo_root),
        *_absent_template_capabilities(report, comfyui),
    ]
    capabilities.sort(key=lambda item: cast("str", item["capabilityId"]))
    states = Counter(cast("str", item["state"]) for item in capabilities)
    tiers = Counter(
        cast("str", item["highestProvenTier"])
        for item in capabilities
        if item["highestProvenTier"] is not None
    )
    kinds = Counter(cast("str", item["kind"]) for item in capabilities)
    return {
        "capabilities": capabilities,
        "format": EVIDENCE_FORMAT,
        "inputs": {
            "comfyui": {
                "modelClasses": len(comfyui.class_names),
                "registeredNodes": len(comfyui.node_ids),
                "repository": "https://github.com/Comfy-Org/ComfyUI",
                "revision": COMFYUI_REVISION,
            },
            "templates": {
                "repository": "https://github.com/Comfy-Org/workflow_templates",
                "revision": TEMPLATE_REVISION,
            },
        },
        "summary": {
            "byKind": dict(sorted(kinds.items())),
            "byState": _complete_counts(states, EVIDENCE_STATES),
            "byTier": _complete_counts(tiers, EVIDENCE_TIERS),
            "supportedCapabilities": sum(item["supported"] is True for item in capabilities),
            "totalCapabilities": len(capabilities),
        },
        "templateFeaturePaths": build_template_features(workflows, source),
        "tierDefinitions": {
            "T0": {
                "countsAsSupported": False,
                "requirement": "Schema translation only.",
            },
            "T1": {
                "countsAsSupported": False,
                "requirement": "Compatibility alias demonstrably lowers to a native operation.",
            },
            "T2": {
                "countsAsSupported": True,
                "requirement": "Native execution with focused tests.",
            },
            "T3": {
                "countsAsSupported": True,
                "requirement": (
                    "Pinned authoritative-source correctness using an official checkpoint."
                ),
            },
            "T4": {
                "countsAsSupported": True,
                "requirement": (
                    "Matched source-versus-Dinkster official-weight same-GPU end-to-end evidence "
                    "covering correctness, cold and warm performance, GPU and host memory, "
                    "residual memory, and template LoRA and adapter paths."
                ),
            },
        },
        "workflows": [
            {"capabilityEvidence": workflow["capabilityEvidence"], "path": workflow["path"]}
            for workflow in cast("list[dict[str, object]]", report["workflows"])
        ],
    }


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def _markdown_cell(value: object) -> str:
    text = "".join(
        character if ord(character) >= 0x20 and ord(character) != 0x7F else " "
        for character in str(value)
    )
    return text.replace("|", "\\|").encode("ascii", "backslashreplace").decode("ascii")


def render_markdown(report: Mapping[str, object]) -> bytes:
    corpus = cast("dict[str, Any]", report["corpus"])
    coverage = cast("dict[str, Any]", report["coverage"])
    registry = cast("dict[str, Any]", report["registry"])
    receipts = cast("dict[str, Any]", report["receipts"])
    source_parity = cast("dict[str, Any]", report["sourceParity"])
    inputs = cast("dict[str, Any]", report["inputs"])
    snapshot = cast("dict[str, Any]", inputs["registrySnapshot"])
    statuses = cast("dict[str, int]", coverage["statusCounts"])
    kind_tiers = cast(
        "dict[str, dict[str, int]]", coverage["confidenceTierOccurrencesByMappingKind"]
    )
    declaration_kind_tiers = cast(
        "dict[str, dict[str, int]]", registry["declarationsByTierAndMappingKind"]
    )
    receipt_kind_tiers = cast("dict[str, dict[str, int]]", receipts["countsByTierAndMappingKind"])
    lines = [
        "# ComfyUI translation coverage baseline",
        "",
        f"Pinned workflow templates: `{inputs['templateRevision']}`. Canonical per-workflow "
        "translation data: "
        "[comfy-translation-coverage.json](comfy-translation-coverage.json). Canonical capability "
        "evidence: [comfy-capability-evidence.json](comfy-capability-evidence.json).",
        "The previous 580-workflow denominator remains in "
        "[the historical baseline](research/comfy-translation-coverage-aa3661d9.md).",
        "",
        "## Corpus",
        "",
        f"{corpus['workflows']} workflows, {corpus['subgraphs']} subgraphs, and "
        f"{corpus['nodes']} node occurrences were parsed from {corpus['jsonFiles']} JSON files; "
        f"{len(corpus['ignoredJsonFiles'])} non-workflow JSON files were reported and ignored.",
        "",
        "| Status | Node occurrences |",
        "| --- | ---: |",
    ]
    for status in STATUSES:
        lines.append(f"| {status} | {statuses[status]} |")
    lines.extend(
        [
            "",
            f"Translation-ready workflows: {coverage['translationReadyWorkflows']} / "
            f"{corpus['workflows']}.",
            f"Evidence-supported workflows (weakest link T2 or stronger): "
            f"{coverage['supportedWorkflows']} / {corpus['workflows']}.",
            "",
            "`quarantine` means the source pack is known but no maintained native alias exists. "
            "`unsupported` means source ownership cannot be resolved safely or a maintained "
            "mapping explicitly refuses the source node. `unavailable` means a maintained "
            "mapping exists but its native provider is absent.",
            "",
            "## Op and family confidence",
            "",
            "| Mapping kind | Tier | Declarations | Mapped occurrences | Receipt cases |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for kind in MAPPING_KINDS:
        for tier in CONFIDENCE_TIERS:
            lines.append(
                f"| {kind} | {tier} | {declaration_kind_tiers[kind][tier]} | "
                f"{kind_tiers[kind][tier]} | {receipt_kind_tiers[kind][tier]} |"
            )
    lines.extend(
        [
            "",
            f"Receipt results: {receipts['passing']} passing, {receipts['failed']} failing, "
            f"{receipts['total']} total. Each canonical receipt and its source/native artifacts "
            "are replay-verified; registry evidence alone is not parity evidence.",
            "",
            "## Source-parity receipt debt",
            "",
            f"{source_parity['receiptBackedMappings']} / "
            f"{source_parity['translationDeclarations']} maintained translation declarations have "
            "at least one verified passing source/native receipt; "
            f"{source_parity['unreceiptedMappings']} remain unreceipted. The parity unit is one "
            "unique maintained mapping, not a receipt-case count.",
            "",
            f"{source_parity['refusedMappings']} maintained fail-closed mappings are reported as "
            "refused and excluded from parity counts; they make no native-equivalence claim.",
            "",
            "Mapped workflow occurrences and translation-ready workflow totals are translation "
            "coverage only. They are excluded from source-parity counts.",
            "",
            "Dinkster-only native-pack exclusions and their non-equivalence "
            "dispositions are recorded in `comfy-source-parity-baseline.json`.",
            "",
            "## Registry cutlines",
            "",
            f"Snapshot `{snapshot['capturedAt']}` contains {snapshot['totalPacks']} packs and "
            f"{snapshot['totalDownloads']} downloads. Rank 355 reaches "
            f"{snapshot['cutlines'][0]['cumulativeDownloads']} downloads "
            f"({snapshot['cutlines'][0]['cumulativeDownloads'] / snapshot['totalDownloads']:.6%}); "
            f"rank 900 reaches {snapshot['cutlines'][1]['cumulativeDownloads']} "
            f"({snapshot['cutlines'][1]['cumulativeDownloads'] / snapshot['totalDownloads']:.6%}).",
            "",
            "| Source band | Mapped | Quarantine | Unavailable | Unsupported | Structural |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    rank_bands = cast("dict[str, dict[str, int]]", coverage["rankBands"])
    for band in RANK_BANDS:
        values = rank_bands[band]
        lines.append(
            f"| {band} | {values['mapped']} | {values['quarantine']} | "
            f"{values['unavailable']} | {values['unsupported']} | {values['structural']} |"
        )
    lines.extend(
        [
            "",
            "## Largest unresolved surfaces",
            "",
            "| Status | Node type | Source pack | Occurrences | Workflows | Reason |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for blocker in cast("list[dict[str, object]]", coverage["blockers"])[:25]:
        lines.append(
            "| {status} | `{node}` | `{pack}` | {occurrences} | {workflows} | {reason} |".format(
                status=_markdown_cell(blocker["status"]),
                node=_markdown_cell(blocker["nodeType"]),
                pack=_markdown_cell(blocker["sourcePack"] or "unknown"),
                occurrences=blocker["occurrences"],
                workflows=blocker["workflows"],
                reason=_markdown_cell(blocker["reason"]),
            )
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def render_evidence_markdown(ledger: Mapping[str, object]) -> bytes:
    inputs = cast("dict[str, Any]", ledger["inputs"])
    summary = cast("dict[str, Any]", ledger["summary"])
    definitions = cast("dict[str, dict[str, object]]", ledger["tierDefinitions"])
    capabilities = cast("list[dict[str, Any]]", ledger["capabilities"])
    features = cast("dict[str, list[dict[str, object]]]", ledger["templateFeaturePaths"])
    model_capabilities = [item for item in capabilities if item["kind"] == "model-family"]
    lines = [
        "# ComfyUI capability evidence",
        "",
        "Canonical data: [comfy-capability-evidence.json](comfy-capability-evidence.json).",
        "",
        f"Workflow templates: `{inputs['templates']['revision']}`. ComfyUI source: "
        f"`{inputs['comfyui']['revision']}` ({inputs['comfyui']['registeredNodes']} statically "
        "registered node IDs).",
        "",
        "## Evidence tiers",
        "",
        "| Tier | Counts as supported | Requirement |",
        "| --- | --- | --- |",
    ]
    for tier in EVIDENCE_TIERS:
        definition = definitions[tier]
        lines.append(
            f"| {tier} | {'yes' if definition['countsAsSupported'] else 'no'} | "
            f"{_markdown_cell(definition['requirement'])} |"
        )
    lines.extend(
        [
            "",
            "T0 and T1 are translation evidence only and never establish support. Missing "
            "evidence remains `absent`, `refused`, or `unverified`; the census does not infer "
            "support from a translated schema.",
            "",
            "## Current ledger",
            "",
            f"{summary['totalCapabilities']} capabilities: "
            + ", ".join(f"{state} {summary['byState'][state]}" for state in EVIDENCE_STATES)
            + f". Supported capability claims: {summary['supportedCapabilities']}.",
            "",
            "| Native model family | Exposure | State | Highest tier | Evidence |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for item in model_capabilities:
        lines.append(
            f"| `{_markdown_cell(item['native']['familyId'])}` | {item['exposure']} | "
            f"{item['state']} | {item['highestProvenTier'] or '-'} | {len(item['evidence'])} |"
        )
    lines.extend(
        [
            "",
            "## Official template feature paths",
            "",
            f"LoRA paths: {len(features['lora'])}. Adapter paths: {len(features['adapter'])}. "
            f"Explicit low-step LoRA paths: {len(features['lowStepLora'])}.",
            "",
            "| Low-step LoRA template | Steps | LoRA node | Sampler node |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in features["lowStepLora"]:
        lora_node = cast("dict[str, object]", row["loraNode"])
        sampler_node = cast("dict[str, object]", row["samplerNode"])
        lines.append(
            f"| `{_markdown_cell(row['path'])}` | {sampler_node['steps']} | "
            f"`{_markdown_cell(lora_node['type'])}` | "
            f"`{_markdown_cell(sampler_node['type'])}` |"
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def _verify_revision(root: Path, expected: str, kind: str) -> None:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        changes = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                ".",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CoverageError(f"cannot verify {kind} revision: {error}") from error
    if revision.stdout.strip() != expected:
        raise CoverageError(
            f"{kind} must be checked out at {expected}; found {revision.stdout.strip()}"
        )
    if changes.stdout:
        raise CoverageError(f"{kind} must be clean at the pinned revision")


def verify_template_revision(templates_root: Path) -> None:
    _verify_revision(templates_root, TEMPLATE_REVISION, "workflow templates")


def verify_comfyui_revision(comfyui_root: Path) -> None:
    _verify_revision(comfyui_root, COMFYUI_REVISION, "ComfyUI source")


def _write_or_check(path: Path, data: bytes, check: bool) -> bool:
    if check:
        return path.is_file() and path.read_bytes() == data
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=_REPO_ROOT / "tools" / "data" / "comfy_registry_downloads_2026-08-26.json",
    )
    parser.add_argument("--packages", type=Path, default=_REPO_ROOT / "packages")
    parser.add_argument(
        "--evidence-source",
        type=Path,
        default=_REPO_ROOT / "tools" / "data" / "comfy_capability_evidence.json",
    )
    parser.add_argument(
        "--supported",
        type=Path,
        default=_REPO_ROOT / "SUPPORTED.md",
        help="capability index or directory of capability documents",
    )
    parser.add_argument(
        "--receipts", type=Path, default=_REPO_ROOT / "docs" / "comfy-confidence-receipts"
    )
    parser.add_argument(
        "--source-parity-baseline",
        type=Path,
        default=_REPO_ROOT / "docs" / "comfy-source-parity-baseline.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=_REPO_ROOT / "docs" / "comfy-translation-coverage.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=_REPO_ROOT / "docs" / "comfy-translation-coverage.md",
    )
    parser.add_argument(
        "--evidence-json-output",
        type=Path,
        default=_REPO_ROOT / "docs" / "comfy-capability-evidence.json",
    )
    parser.add_argument(
        "--evidence-markdown-output",
        type=Path,
        default=_REPO_ROOT / "docs" / "comfy-capability-evidence.md",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        verify_template_revision(args.templates)
        verify_comfyui_revision(args.comfyui)
        workflows, ignored = load_workflows(args.templates)
        comfyui = load_comfyui_source(args.comfyui)
        catalog = load_registry_catalog(args.packages)
        snapshot = load_download_snapshot(args.snapshot)
        source_parity_baseline = load_source_parity_baseline(
            args.source_parity_baseline, args.packages
        )
        evidence_source = load_evidence_source(args.evidence_source, _REPO_ROOT)
        lint_supported_claims(evidence_source, args.supported, _REPO_ROOT)
        report = build_report(
            workflows=workflows,
            ignored_files=ignored,
            catalog=catalog,
            snapshot=snapshot,
            receipts_root=args.receipts,
            source_parity_baseline=source_parity_baseline,
        )
        ledger = build_evidence_ledger(
            report=report,
            workflows=workflows,
            catalog=catalog,
            source=evidence_source,
            comfyui=comfyui,
            repo_root=_REPO_ROOT,
        )
        outputs = (
            (args.json_output, canonical_json(report)),
            (args.markdown_output, render_markdown(report)),
            (args.evidence_json_output, canonical_json(ledger)),
            (args.evidence_markdown_output, render_evidence_markdown(ledger)),
        )
        current = [_write_or_check(path, data, args.check) for path, data in outputs]
    except (ConfidenceReceiptError, CoverageError, ManifestError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not all(current):
        stale = [
            str(path) for (path, _), matches in zip(outputs, current, strict=True) if not matches
        ]
        print(f"error: generated coverage output is stale: {', '.join(stale)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
