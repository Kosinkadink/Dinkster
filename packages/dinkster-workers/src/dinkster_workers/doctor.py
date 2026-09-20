"""dinkster doctor - the pack linter (DESIGN 3.9).

Doctor makes drifting into bad practice visible *before* users report
breakage: manifest completeness, import-time side effects, private/host
imports, schema problems, and codec-less types that the pack's own schemas
send across boundaries. Two halves:

- **Static checks** run here, in the doctor's process: manifest shape,
  dependency pinning, and a source scan for import contract violations.
  No pack code is ever imported into the doctor's process.
- **The probe** (``_doctor_probe``) imports the pack entries in a
  disposable subprocess and reports what happened as JSON - a broken or
  noisy import is a finding, never doctor pollution.

Every finding carries the fix, not just the problem (DESIGN 3.9): perf
findings are warnings, contract violations (private imports, host-machinery
imports) are errors even though Python cannot physically prevent them.

The report is structured (``DoctorReport.to_json()``) so CI, dev mode, and
frontends consume the same diagnostics the CLI renders. The JSON report is
the stable machine interface: versioned via ``reportVersion`` (bumped only
for breaking shape changes; new fields are additive), and it names the
contract the run validated against (``apiVersion`` = the dinkster_api surface,
``doctorVersion`` = this package). Exit codes are part of the contract:
0 = healthy, 1 = unhealthy (at least one error finding), 2 = usage error.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from dinkster_schema import claim_covers, reserved_root
from packaging.requirements import InvalidRequirement, Requirement

from .accelerator import detect_accelerator
from .interpreter import InterpreterPreflightError, preflight_interpreter
from .manifest import (
    BLUEPRINT_PACK_MAX_BYTES,
    TEMPLATE_PACK_MAX_BYTES,
    ManifestError,
    PackManifest,
    load_manifest,
    unmatched_registry_providers,
    validate_pack_asset,
    validate_pack_blueprint,
    validate_pack_icon,
    validate_pack_template,
)
from .sandbox import ProbeJail, SandboxError, detect_probe_jail, probe_environment

Severity = Literal["error", "warning", "info"]

# The machine-report contract. reportVersion bumps only on breaking shape
# changes (field removal/retyping); additions ride the same version.
DOCTOR_REPORT_VERSION = 1
# The pack-author contract this doctor validates against (dinkster_api surface).
DOCTOR_API_VERSION = "v1"


def _doctor_version() -> str:
    """The doctor's own distribution version, for report provenance."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("dinkster-workers")
    except PackageNotFoundError:  # source checkout without install metadata
        return "0.0.0"


_HOST_MACHINERY = frozenset(
    {
        "dinkster_engine",
        "dinkster_protocol",
        "dinkster_server",
        "dinkster_workers",
        "dinkster_caches",
        "dinkster_graph",
        "dinkster_compat_comfy",
    }
)
_INTERNAL_SURFACE = frozenset(
    {"dinkster_schema", "dinkster_values", "dinkster_memory", "dinkster_assets"}
)
_PACK_IMPLEMENTATION_PREFIXES = ("dinkster_nodes_", "dinkster_model_")
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import\s+([\w.,\s*()]+)"
    r"|import\s+([A-Za-z_][\w.]*))"
)
_SKIP_DIRS = frozenset({"__pycache__", ".venv", ".git", "node_modules", ".tox"})
_PROBE_TIMEOUT_S = 300.0
_SLOW_IMPORT_MS = 2000.0
_OUTPUT_EXCERPT = 200
SHORT_DESCRIPTION_MAX_CHARS = 240


@dataclass(frozen=True)
class Finding:
    """One diagnostic: what is wrong, how bad, and how to fix it."""

    severity: Severity
    code: str
    message: str
    fix: str = ""
    location: str = ""


@dataclass(frozen=True)
class StaticDoctorReport:
    """Versioned findings from checks that do not execute pack code."""

    report_version: int
    findings: tuple[Finding, ...]


@dataclass(frozen=True)
class DocsCoverage:
    default_locale: str
    total_nodes: int
    locale_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class DoctorReport:
    pack_name: str
    manifest_path: str
    findings: tuple[Finding, ...]
    node_types: tuple[str, ...] = ()
    docs_coverage: DocsCoverage | None = None

    @property
    def ok(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    def to_json(self) -> str:
        wire: dict[str, object] = {
            "reportVersion": DOCTOR_REPORT_VERSION,
            "apiVersion": DOCTOR_API_VERSION,
            "doctorVersion": _doctor_version(),
            "pack": self.pack_name,
            "manifest": self.manifest_path,
            "ok": self.ok,
            "nodeTypes": list(self.node_types),
            "findings": [asdict(f) for f in self.findings],
        }
        if self.docs_coverage is not None:
            wire["docsCoverage"] = {
                "defaultLocale": self.docs_coverage.default_locale,
                "totalNodes": self.docs_coverage.total_nodes,
                "locales": dict(self.docs_coverage.locale_counts),
            }
        return json.dumps(wire, indent=2)


def _check_docs(
    manifest: PackManifest, nodes: tuple[dict[str, Any], ...]
) -> tuple[list[Finding], DocsCoverage | None]:
    findings: list[Finding] = []
    for problem in manifest.locale_catalog_problems:
        findings.append(
            Finding(
                "error",
                "docs.invalid",
                problem,
                "fix the invalid locale catalog or remove the file",
                str(manifest.path),
            )
        )
    known_types = {cast("str", node["node_type"]) for node in nodes}
    for catalog in manifest.locale_catalogs:
        for node_type in catalog.node_references:
            if node_type in known_types:
                continue
            findings.append(
                Finding(
                    "error",
                    "docs.invalid",
                    f"locale catalog {catalog.locale!r} references unknown node type {node_type!r}",
                    "translate a node type declared by this pack or remove the catalog entry",
                    str(manifest.path),
                )
            )
    if manifest.docs_problem is not None:
        findings.append(
            Finding(
                "error",
                "docs.invalid",
                f"[pack.docs] {manifest.docs_problem}",
                "declare a docs directory inside the pack and a supported default locale",
                str(manifest.path),
            )
        )
    for node in nodes:
        description = cast("str", node["description"])
        if len(description) > SHORT_DESCRIPTION_MAX_CHARS:
            findings.append(
                Finding(
                    "warning",
                    "docs.description-too-long",
                    f"node '{node['node_type']}' description is {len(description)} characters; "
                    f"the short-description limit is {SHORT_DESCRIPTION_MAX_CHARS}",
                    "keep one or two search-result sentences here and move details "
                    "into its docs page",
                    str(manifest.path),
                )
            )
    if manifest.docs is None:
        return findings, None
    for problem in manifest.docs.validation_problems:
        findings.append(
            Finding(
                "error",
                "docs.invalid",
                problem,
                "fix the invalid documentation block or remove the page",
                str(manifest.path),
            )
        )
    node_by_type = {cast("str", node["node_type"]): node for node in nodes}
    node_pages = [page for page in manifest.docs.pages if page.kind == "node"]
    for page in node_pages:
        node = node_by_type.get(page.id)
        if node is None:
            findings.append(
                Finding(
                    "warning",
                    "docs.unknown-node",
                    f"documentation page names unknown node type '{page.id}'",
                    "move the page under a node type declared by this pack or remove it",
                    str(manifest.path),
                )
            )
        elif page.schema_version is not None and page.schema_version < node["version"]:
            findings.append(
                Finding(
                    "warning",
                    "docs.stale-schema-version",
                    f"documentation for '{page.id}' targets schema version "
                    f"{page.schema_version}, but the node is version {node['version']}",
                    "review the page and update its schema_version",
                    str(manifest.path),
                )
            )
    known_types = set(node_by_type)
    seen_references: set[tuple[str, str, str]] = set()
    for page in manifest.docs.pages:
        for node_type in page.node_references:
            reference = (page.kind, page.id, node_type)
            if node_type in known_types or reference in seen_references:
                continue
            seen_references.add(reference)
            findings.append(
                Finding(
                    "warning",
                    "docs.unknown-node-reference",
                    f"documentation {page.kind} '{page.id}' references unknown node type "
                    f"'{node_type}'",
                    "reference a node type declared by this pack or remove the dinkster-node block",
                    str(manifest.path),
                )
            )
    locale_counts = tuple(
        (
            locale,
            len(
                {page.id for page in node_pages if page.locale == locale and page.id in known_types}
            ),
        )
        for locale in sorted({manifest.docs.default_locale, "en", "zh"})
    )
    return findings, DocsCoverage(manifest.docs.default_locale, len(known_types), locale_counts)


def _check_requirement_strings(
    manifest: PackManifest, requirements: tuple[str, ...], where: str
) -> list[Finding]:
    """Validate one requirement list as PEP 508: syntax errors are error
    findings (uv would refuse the whole install at provision time - the
    doctor catches it before any user does), unconstrained deps stay
    warnings. Pure parsing via ``packaging`` - no index, no network, no
    GPU framework anywhere near this."""
    findings: list[Finding] = []
    for raw in requirements:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as exc:
            findings.append(
                Finding(
                    severity="error",
                    code="manifest.invalid-requirement",
                    message=f"{where} entry {raw!r} is not valid PEP 508: {exc}",
                    fix="fix the requirement syntax; every entry must parse "
                    "as a PEP 508 dependency specification",
                    location=str(manifest.path),
                )
            )
            continue
        if not requirement.specifier and requirement.url is None:
            findings.append(
                Finding(
                    severity="warning",
                    code="manifest.unpinned-dep",
                    message=f"{where} dependency '{raw}' has no version constraint",
                    fix=(
                        f"pin a range, e.g. '{raw}>=1.0' - unpinned deps "
                        f"make pack venvs unreproducible"
                    ),
                    location=str(manifest.path),
                )
            )
    return findings


def _check_requires(manifest: PackManifest) -> list[Finding]:
    findings = _check_requirement_strings(manifest, manifest.requires, "[pack] requires")
    for accelerator, extras in manifest.extra_requires:
        findings.extend(
            _check_requirement_strings(manifest, extras, f"[pack.extra-requires] {accelerator}")
        )
    return findings


def _check_environment(manifest: PackManifest) -> list[Finding]:
    """Advisory host-fit findings: does the pack declare compatibility
    with the machine the doctor runs on? Detection is the side-effect-free
    driver-file/PATH inspection - never a torch import, never a GPU
    context. Info severity: the doctor host is often not the deploy host
    (CI runners have no GPU), so a mismatch is context, not a defect."""
    findings: list[Finding] = []
    if manifest.platforms and sys.platform not in manifest.platforms:
        findings.append(
            Finding(
                severity="info",
                code="environment.platform-excluded",
                message=f"this host ({sys.platform}) is not among the "
                f"declared [pack] platforms ({', '.join(manifest.platforms)})",
                fix="expected when linting for another OS; install plans "
                "on excluded hosts warn the same way",
                location=str(manifest.path),
            )
        )
    if manifest.extra_requires:
        accelerator = detect_accelerator()
        declared = tuple(key for key, _ in manifest.extra_requires)
        if accelerator not in declared:
            findings.append(
                Finding(
                    severity="info",
                    code="environment.accelerator-not-covered",
                    message=f"this host's accelerator ({accelerator}) has no "
                    f"[pack.extra-requires] entry (declared: {', '.join(declared)})",
                    fix="fine when base requires suffice on this "
                    "accelerator; add an entry if it needs extra dists",
                    location=str(manifest.path),
                )
            )
    return findings


def _check_sandbox_needs(manifest: PackManifest) -> list[Finding]:
    if manifest.sandbox_declared:
        return []
    return [
        Finding(
            severity="warning",
            code="sandbox.needs-undeclared",
            message="the manifest does not declare its GPU, network, or writable-mount needs",
            fix="add [pack.sandbox]; an empty table explicitly requests no sandbox grants",
            location=str(manifest.path),
        )
    ]


def _classify_import(
    module: str, names: str | None, own_modules: frozenset[str] = frozenset()
) -> tuple[str, str] | None:
    """Return (code, fix) for a problematic dinkster import, else None."""
    parts = module.split(".")
    root = parts[0]
    if not root.startswith("dinkster"):
        return None
    if any(part.startswith("_") for part in parts[1:]):
        return (
            "imports.private-module",
            "underscore-private modules carry no compatibility promise; use dinkster_api.v1",
        )
    if names and root != "dinkster_api":
        for name in re.split(r"[,\s()]+", names):
            if name.startswith("_") and name not in ("", "_"):
                return (
                    "imports.private-module",
                    "underscore-private names carry no compatibility promise; use dinkster_api.v1",
                )
    if root in _HOST_MACHINERY:
        return (
            "imports.host-machinery",
            f"packs must not import host machinery ({root}); everything a "
            f"pack needs is in dinkster_api.v1",
        )
    if root.startswith(_PACK_IMPLEMENTATION_PREFIXES) and root not in own_modules:
        return (
            "imports.pack-implementation",
            f"packs must not import another pack implementation ({root}); depend on "
            "registered ids or extract shared code into a versioned library",
        )
    if root in _INTERNAL_SURFACE:
        return (
            "imports.internal-module",
            f"import from dinkster_api.v1 instead of {root} directly - the "
            f"door is versioned, internals may move",
        )
    return None


def _check_reserved_namespaces(manifest: PackManifest) -> list[Finding]:
    """Reserved-root claims are legal manifest shape (core packs declare
    them like anyone else - H5, no privileged shortcut) but they compose
    only under explicit host trust and the registry never grants them, so
    a third-party pack carrying one is drifting toward an unpublishable,
    uninstallable state. Warning, not error: doctor cannot know it is not
    looking at a first-party pack."""
    findings: list[Finding] = []
    for claim in manifest.namespaces:
        root = reserved_root(claim)
        if root is not None:
            findings.append(
                Finding(
                    severity="warning",
                    code="namespace.reserved",
                    message=f"namespace claim '{claim}' falls under the reserved root '{root}'",
                    fix="reserved roots (std, comfy, core, dinkster) compose "
                    "only when the host explicitly trusts the pack and are "
                    "never granted by the registry; claim a namespace of "
                    "your own in [pack] namespaces",
                    location=str(manifest.path),
                )
            )
    return findings


def _check_namespace_coverage(manifest: PackManifest, node_types: tuple[str, ...]) -> list[Finding]:
    """Every node type the pack announces must fall under a declared claim
    (doctor validates, composition enforces, the registry
    grants). Error, because the registry rejects uncovered node types at
    publish and local composition refuses them at startup. Node types in
    [pack] executes are exempt, exactly as composition exempts them: the
    owning pack's claim covers them, not the executor's."""
    claims = manifest.namespaces
    findings: list[Finding] = []
    for node_type in node_types:
        if node_type in manifest.executes:
            continue
        if not any(claim_covers(claim, node_type) for claim in claims):
            findings.append(
                Finding(
                    severity="error",
                    code="namespace.uncovered",
                    message=f"node type '{node_type}' is outside the pack's "
                    f"declared namespaces ({', '.join(claims)})",
                    fix="add its namespace to [pack] namespaces, or rename "
                    "the node type under a declared claim - composition and "
                    "the registry both refuse uncovered node types",
                    location=str(manifest.path),
                )
            )
    return findings


def _check_icon(manifest: PackManifest) -> list[Finding]:
    """Publish-time gate for [pack.presentation] icon, via the SAME
    validator the loader applies (warn-and-drop there, error here) so the
    linter and the runtime can never disagree about what is valid. The
    loader drops an invalid icon before the manifest model sees it, so
    the raw declaration is re-read from the TOML."""
    try:
        with open(manifest.path, "rb") as fh:
            document = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return []  # load_manifest already succeeded or reported; not our lane
    pack = document.get("pack")
    if not isinstance(pack, dict):
        return []
    presentation = cast("dict[str, Any]", pack).get("presentation")
    if not isinstance(presentation, dict):
        return []
    declared = cast("dict[str, Any]", presentation).get("icon")
    if declared is None:
        return []
    if not isinstance(declared, str) or not declared:
        return [
            Finding(
                severity="error",
                code="presentation.icon-invalid",
                message="[pack.presentation] icon must be a non-empty string path",
                fix="declare icon as a pack-relative path to a 64x64 static PNG/WebP under 64 KiB",
                location=str(manifest.path),
            )
        ]
    _, problem = validate_pack_icon(manifest.path, declared)
    if problem is None:
        return []
    return [
        Finding(
            severity="error",
            code="presentation.icon-invalid",
            message=f"[pack.presentation] icon {declared!r} {problem}",
            fix="ship a 64x64 static PNG/WebP under 64 KiB inside the pack "
            "directory (the loader would warn and drop this icon)",
            location=str(manifest.path),
        )
    ]


def _check_blueprints(manifest: PackManifest) -> list[Finding]:
    """Publish-time gate for [[pack.blueprints]], via the SAME validator
    the loader applies (warn-and-drop there, error here) so the linter and
    the runtime can never disagree. The loader drops invalid entries before
    the manifest model sees them, so the raw declarations are re-read from
    the TOML. Checks are backend-ownable only: shape, path containment,
    UTF-8, well-formed JSON, size caps, duplicate ids - never workflow
    document semantics, which are frontend-owned."""
    try:
        with open(manifest.path, "rb") as fh:
            document = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return []  # load_manifest already succeeded or reported; not our lane
    pack = document.get("pack")
    if not isinstance(pack, dict):
        return []
    raw = cast("dict[str, Any]", pack).get("blueprints")
    if raw is None:
        return []
    if not isinstance(raw, list):
        return [
            Finding(
                severity="error",
                code="blueprints.invalid",
                message="[[pack.blueprints]] must be an array of tables",
                fix="declare each blueprint as a [[pack.blueprints]] entry with id, name, and file",
                location=str(manifest.path),
            )
        ]
    findings: list[Finding] = []
    seen: set[str] = set()
    total = 0
    for index, entry in enumerate(cast("list[Any]", raw)):
        blueprint, problem = validate_pack_blueprint(manifest.path, entry)
        if blueprint is None:
            findings.append(
                Finding(
                    severity="error",
                    code="blueprints.invalid",
                    message=f"[[pack.blueprints]] entry {index} {problem}",
                    fix="ship a well-formed JSON workflow document under "
                    "1 MiB inside the pack directory (the loader would "
                    "warn and drop this blueprint)",
                    location=str(manifest.path),
                )
            )
            continue
        if blueprint.id in seen:
            findings.append(
                Finding(
                    severity="error",
                    code="blueprints.duplicate-id",
                    message=f"[[pack.blueprints]] entry {index} duplicates id {blueprint.id!r}",
                    fix="blueprint ids are namespaced by the pack and must be unique within it",
                    location=str(manifest.path),
                )
            )
            continue
        seen.add(blueprint.id)
        total += len(blueprint.data)
    if total > BLUEPRINT_PACK_MAX_BYTES:
        findings.append(
            Finding(
                severity="error",
                code="blueprints.budget-exceeded",
                message=f"[[pack.blueprints]] entries total {total} bytes; "
                f"the per-pack budget is {BLUEPRINT_PACK_MAX_BYTES}",
                fix="blueprints are graphs, not assets - reference models "
                "and images by digest-pinned URLs instead of embedding them "
                "(the loader would drop entries past the budget)",
                location=str(manifest.path),
            )
        )
    return findings


def _check_assets(manifest: PackManifest) -> list[Finding]:
    """Publish-time gate for [[pack.assets]], via the SAME validator the
    loader applies (warn-and-drop there, error here) so the linter and the
    runtime can never disagree. The loader drops invalid entries before
    the manifest model sees them, so the raw declarations are re-read from
    the TOML. Checks are declaration-shape only: id grammar, digest pin,
    source presence, path containment - never whether the pack's code
    actually loads the asset (declarations are distribution records)."""
    try:
        with open(manifest.path, "rb") as fh:
            document = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return []  # load_manifest already succeeded or reported; not our lane
    pack = document.get("pack")
    if not isinstance(pack, dict):
        return []
    raw = cast("dict[str, Any]", pack).get("assets")
    if raw is None:
        return []
    if not isinstance(raw, list):
        return [
            Finding(
                severity="error",
                code="assets.invalid",
                message="[[pack.assets]] must be an array of tables",
                fix="declare each asset as a [[pack.assets]] entry with id, "
                "name, digest, and a packaged 'file' and/or remote 'urls'",
                location=str(manifest.path),
            )
        ]
    findings: list[Finding] = []
    seen: set[str] = set()
    for index, entry in enumerate(cast("list[Any]", raw)):
        asset, problem = validate_pack_asset(manifest.path, entry, manifest.name)
        if asset is None:
            findings.append(
                Finding(
                    severity="error",
                    code="assets.invalid",
                    message=f"[[pack.assets]] entry {index} {problem}",
                    fix="declare a digest-pinned asset with at least one "
                    "source: a packaged file inside the pack directory "
                    "and/or http(s) urls (the loader would warn and drop "
                    "this entry)",
                    location=str(manifest.path),
                )
            )
            continue
        if asset.id in seen:
            findings.append(
                Finding(
                    severity="error",
                    code="assets.duplicate-id",
                    message=f"[[pack.assets]] entry {index} duplicates id {asset.id!r}",
                    fix="asset ids are namespaced by the pack and must be unique within it",
                    location=str(manifest.path),
                )
            )
            continue
        seen.add(asset.id)
    return findings


def _check_templates(manifest: PackManifest) -> list[Finding]:
    """Publish-time gate for [[pack.templates]], via the SAME validator
    the loader applies (warn-and-drop there, error here) so the linter
    and the runtime can never disagree. The loader drops invalid entries
    before the manifest model sees them, so the raw declarations are
    re-read from the TOML. Checks are backend-ownable only - the same
    boundary as blueprints - plus the one referential rule the parser
    enforces: every referenced asset id must exist among the pack's
    surviving [[pack.assets]] declarations, or the template's acquisition
    plan can never be constructed."""
    try:
        with open(manifest.path, "rb") as fh:
            document = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return []  # load_manifest already succeeded or reported; not our lane
    pack = document.get("pack")
    if not isinstance(pack, dict):
        return []
    raw = cast("dict[str, Any]", pack).get("templates")
    if raw is None:
        return []
    if not isinstance(raw, list):
        return [
            Finding(
                severity="error",
                code="templates.invalid",
                message="[[pack.templates]] must be an array of tables",
                fix="declare each template as a [[pack.templates]] entry with id, name, and file",
                location=str(manifest.path),
            )
        ]
    declared_assets = {asset.id for asset in manifest.assets}
    findings: list[Finding] = []
    seen: set[str] = set()
    total = 0
    for index, entry in enumerate(cast("list[Any]", raw)):
        template, problem = validate_pack_template(manifest.path, entry)
        if template is None:
            findings.append(
                Finding(
                    severity="error",
                    code="templates.invalid",
                    message=f"[[pack.templates]] entry {index} {problem}",
                    fix="ship a well-formed JSON workflow document under "
                    "1 MiB inside the pack directory (the loader would "
                    "warn and drop this template)",
                    location=str(manifest.path),
                )
            )
            continue
        if template.id in seen:
            findings.append(
                Finding(
                    severity="error",
                    code="templates.duplicate-id",
                    message=f"[[pack.templates]] entry {index} duplicates id {template.id!r}",
                    fix="template ids are namespaced by the pack and must be unique within it",
                    location=str(manifest.path),
                )
            )
            continue
        dangling = [ref for ref in template.assets if ref not in declared_assets]
        if dangling:
            findings.append(
                Finding(
                    severity="error",
                    code="templates.dangling-asset",
                    message=f"[[pack.templates]] entry {index} ({template.id!r}) "
                    f"references undeclared asset ids "
                    f"{', '.join(repr(ref) for ref in dangling)}",
                    fix="declare each referenced asset as a [[pack.assets]] "
                    "entry so the template's requirements stay digest-pinned "
                    "(the loader would warn and drop this template)",
                    location=str(manifest.path),
                )
            )
            continue
        seen.add(template.id)
        total += len(template.data)
    if total > TEMPLATE_PACK_MAX_BYTES:
        findings.append(
            Finding(
                severity="error",
                code="templates.budget-exceeded",
                message=f"[[pack.templates]] entries total {total} bytes; "
                f"the per-pack budget is {TEMPLATE_PACK_MAX_BYTES}",
                fix="templates are graphs, not assets - reference models "
                "via [[pack.assets]] ids instead of embedding bytes "
                "(the loader would drop entries past the budget)",
                location=str(manifest.path),
            )
        )
    return findings


def _scan_sources(manifest: PackManifest) -> list[Finding]:
    findings: list[Finding] = []
    paths = tuple(
        path
        for path in sorted(manifest.root.rglob("*.py"))
        if not any(part in _SKIP_DIRS for part in path.parts)
    )
    own_module_names = {
        entry.partition(":")[0].partition(".")[0]
        for entry in (
            manifest.nodes_entry,
            manifest.types_entry,
            manifest.arm_nodes_entry,
            manifest.source_staging_entry,
            manifest.workgroup_handler_entry,
        )
        if entry is not None
    }
    own_module_names.update(path.stem for path in paths)
    own_module_names.update(path.parent.name for path in paths if path.name == "__init__.py")
    own_modules = frozenset(own_module_names)
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        imports: list[tuple[str, str | None, int]] = []
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            for lineno, line in enumerate(source.splitlines(), start=1):
                match = _IMPORT_RE.match(line)
                if match is not None:
                    imports.append((match.group(1) or match.group(3), match.group(2), lineno))
        else:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend((alias.name, None, node.lineno) for alias in node.names)
                elif (
                    isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None
                ):
                    imports.append(
                        (node.module, ",".join(alias.name for alias in node.names), node.lineno)
                    )
        for module, names, lineno in sorted(imports, key=lambda item: (item[2], item[0])):
            problem = _classify_import(module, names, own_modules)
            if problem is None:
                continue
            code, fix = problem
            findings.append(
                Finding(
                    severity="error"
                    if code
                    in (
                        "imports.private-module",
                        "imports.host-machinery",
                        "imports.pack-implementation",
                    )
                    else "warning",
                    code=code,
                    message=f"imports {module}",
                    fix=fix,
                    location=f"{path.relative_to(manifest.root)}:{lineno}",
                )
            )
    return findings


def _run_probe(
    manifest: PackManifest,
    probe_jail: ProbeJail | None,
    interpreter: Path | str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any] | Finding:
    python = str(interpreter) if interpreter is not None else sys.executable
    try:
        preflight_interpreter(python)
    except InterpreterPreflightError as exc:
        return Finding(
            severity="error",
            code="doctor.interpreter",
            message=str(exc),
            fix="select an executable Python >=3.12 for the pack import probe",
        )
    argv = [python, "-m", "dinkster_workers._doctor_probe", str(manifest.path)]
    try:
        if probe_jail is None:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=_PROBE_TIMEOUT_S,
                env=None if environment is None else {**os.environ, **environment},
            )
        else:
            with tempfile.TemporaryFile() as environment_file:
                environment_file.write(json.dumps(probe_environment(environment)).encode("utf-8"))
                environment_file.flush()
                environment_file.seek(0)
                env_fd = environment_file.fileno()
                proc = subprocess.run(
                    probe_jail.wrap(argv, manifest.root, env_fd=env_fd),
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=_PROBE_TIMEOUT_S,
                    env={},
                    pass_fds=(env_fd,),
                )
    except subprocess.TimeoutExpired:
        return Finding(
            severity="error",
            code="doctor.probe-timeout",
            message=f"pack import probe exceeded {_PROBE_TIMEOUT_S:.0f}s",
            fix="something at import time never returns; move long work "
            "into execute() or lazy initialization",
        )
    if proc.returncode != 0:
        detail = _redact_environment_values(proc.stderr.strip())[-500:]
        return Finding(
            severity="error",
            code="doctor.probe-failed",
            message=f"import probe crashed: {_excerpt(detail, limit=500)}",
            fix="run the probe by hand for the full traceback: "
            f"python -m dinkster_workers._doctor_probe {manifest.path}",
        )
    try:
        return cast("dict[str, Any]", json.loads(proc.stdout))
    except json.JSONDecodeError:
        return Finding(
            severity="error",
            code="doctor.probe-failed",
            message="import probe produced no readable report",
            fix="the pack's import likely wrote to stdout after redirection "
            "ended or corrupted the process; check for atexit handlers",
        )


def _redact_environment_values(text: str) -> str:
    environment_values = sorted(
        {value for value in os.environ.values() if len(value) >= 4},
        key=len,
        reverse=True,
    )
    for value in environment_values:
        text = text.replace(value, "<redacted>")
    return text


def _excerpt(text: str, *, limit: int = _OUTPUT_EXCERPT) -> str:
    text = _redact_environment_values(text)
    flat = " ".join(text.split())
    return flat[:limit] + ("..." if len(flat) > limit else "")


def _probe_findings(report: dict[str, Any], manifest: PackManifest) -> list[Finding]:
    findings: list[Finding] = []
    if report["entry_error"] is not None:
        findings.append(
            Finding(
                severity="error",
                code="entry.unresolvable",
                message=f"[pack.entry] failed to import: {report['entry_error']}",
                fix="check the module:attr targets exist and the pack's "
                "requires are installed in this environment",
            )
        )
        return findings
    catalog = report.get("catalog")
    contributions: list[tuple[str, str]] = []
    if isinstance(catalog, dict):
        raw_contributions = cast("dict[str, object]", catalog).get("inferenceContributions", [])
        if isinstance(raw_contributions, list):
            for raw in cast("list[object]", raw_contributions):
                if not isinstance(raw, dict):
                    continue
                declaration = cast("dict[str, object]", raw)
                surface_id = declaration.get("surface_id")
                descriptor_id = declaration.get("id")
                if isinstance(surface_id, str) and isinstance(descriptor_id, str):
                    contributions.append((surface_id, descriptor_id))
    for provider in unmatched_registry_providers(manifest.provides, contributions):
        findings.append(
            Finding(
                severity="error",
                code="registry.provider-unregistered",
                message=f"pack {manifest.name!r} declares registry provider "
                f"{provider.registry}:{provider.id}, but its inference contribution "
                "does not register it",
                fix="return the declared descriptor from [pack.extension] inference, "
                "or remove the provider declaration",
            )
        )
    if report["nodes_entry_problem"] is not None:
        findings.append(
            Finding(
                severity="error",
                code="entry.not-node-classes",
                message=report["nodes_entry_problem"],
                fix="entry.nodes must name a list/tuple of Node subclasses "
                "(from dinkster_api.v1 import Node)",
            )
        )
    for problem in report["schema_errors"]:
        findings.append(
            Finding(
                severity="error",
                code="schema.invalid",
                message=f"define_schema() failed: {problem}",
                fix="the schema must build without error at load time; the "
                "message above is the contract being violated",
            )
        )
    for problem in report["replacement_errors"]:
        findings.append(
            Finding(
                severity="error",
                code="schema.replacement-invalid",
                message=f"replacement rule problem: {problem}",
                fix="replacement rules must reference static schema "
                "input/output ids only - no dynamic family ids or member "
                "paths; fix the rule or the schema it points at",
            )
        )
    for duplicate in report["duplicate_node_types"]:
        findings.append(
            Finding(
                severity="error",
                code="schema.duplicate-node-type",
                message=f"duplicate node type {duplicate}",
                fix="every node class in the pack must declare a unique node_type",
            )
        )
    if report["types_error"] is not None:
        findings.append(
            Finding(
                severity="error",
                code="entry.types-failed",
                message=f"[pack.entry] types callable failed: {report['types_error']}",
                fix="entry.types must accept a TypeRegistry and register "
                "the pack's types without raising",
            )
        )
    for stream in ("stdout", "stderr"):
        output = report[f"import_{stream}"]
        if output.strip():
            findings.append(
                Finding(
                    severity="warning",
                    code="import.side-effect-output",
                    message=f"importing the pack wrote to {stream}: {_excerpt(output)}",
                    fix="import time runs in every worker that loads the "
                    "pack; use pack_logger() from dinkster_api.v1 (doctor "
                    "reports it structurally) or lazy init instead",
                )
            )
    import_logs = report.get("import_logs", [])
    if import_logs:
        by_origin: dict[str, int] = {}
        for record in import_logs:
            by_origin[record["logger"]] = by_origin.get(record["logger"], 0) + 1
        summary = ", ".join(f"{origin} x{count}" for origin, count in sorted(by_origin.items()))
        sample = import_logs[0]
        findings.append(
            Finding(
                severity="info",
                code="import.log",
                message=f"importing the pack logged {len(import_logs)} "
                f"record(s) ({summary}); first: [{sample['level']}] "
                f"{sample['logger']}: {_excerpt(sample['message'])}",
                fix="informational - standardized logging at import time is "
                "fine, but heavy work behind it should still be lazy",
            )
        )
    pack_prefix = "dinkster.pack."
    foreign_origins = sorted(
        {
            record["logger"]
            for record in import_logs
            if record["logger"].startswith(pack_prefix)
            and record["logger"].removeprefix(pack_prefix).split(".", 1)[0] != manifest.name
        }
    )
    if foreign_origins:
        findings.append(
            Finding(
                severity="warning",
                code="import.log-foreign-origin",
                message=f"pack '{manifest.name}' logged under other packs' "
                f"origins: {', '.join(foreign_origins)}",
                fix=f"pack_logger() must be called with this pack's manifest "
                f"name ('{manifest.name}') so operators can attribute and "
                "silence output per pack",
            )
        )
    if report.get("cuda_initialized_on_import"):
        findings.append(
            Finding(
                severity="warning",
                code="import.side-effect-cuda",
                message="importing the pack initialized a CUDA context",
                fix="an idle CUDA context holds hundreds of MiB of VRAM per "
                "worker process for its whole lifetime (measured ~450 MiB "
                "on an RTX 4090) before any work runs; importing torch is "
                "free - keep the first GPU op inside execute(), never at "
                "import",
            )
        )
    if report["threads_started"] > 0:
        findings.append(
            Finding(
                severity="warning",
                code="import.side-effect-threads",
                message=f"importing the pack started {report['threads_started']} thread(s)",
                fix="background work at import time leaks into every host "
                "that loads the pack; start it lazily from execute()",
            )
        )
    if report["import_ms"] > _SLOW_IMPORT_MS:
        findings.append(
            Finding(
                severity="warning",
                code="import.slow",
                message=f"importing the pack took {report['import_ms']:.0f}ms",
                fix="defer heavy imports (torch, model code) into execute() "
                "or function bodies so workers start fast",
            )
        )
    crossing_fallbacks = sorted(
        set(report["fallback_codec_type_ids"]) & set(report["schema_atoms"])
    )
    for type_id in crossing_fallbacks:
        findings.append(
            Finding(
                severity="warning",
                code="types.fallback-codec",
                message=f"type '{type_id}' is used by this pack's schemas "
                f"but registered without a declared codec",
                fix="declare encode/decode in register(...) so boundary "
                "crossings avoid the default fallback codec",
            )
        )
    for atom in report["unregistered_schema_atoms"]:
        findings.append(
            Finding(
                severity="warning",
                code="types.unregistered",
                message=f"schemas reference type '{atom}' but neither core "
                f"nor this pack's types entry registers it",
                fix="register it in entry.types, or depend on the pack that "
                "does - unregistered types fail at wrap time",
            )
        )
    return findings


def diagnose_static(pack_root: Path | str, manifest: PackManifest) -> StaticDoctorReport:
    """Check an extracted pack tree without importing or executing pack code.

    ``pack_root`` is the extracted directory containing ``manifest``. The
    returned report carries the doctor report version and the ordered static
    findings for that manifest and tree. Import probing, schema evaluation,
    and namespace coverage are performed only by ``diagnose()``.
    """
    root = Path(pack_root).resolve()
    if root != manifest.root.resolve():
        raise ValueError("pack_root must contain the supplied manifest")

    findings: list[Finding] = []
    findings.extend(_check_requires(manifest))
    findings.extend(_check_environment(manifest))
    findings.extend(_check_sandbox_needs(manifest))
    findings.extend(_check_reserved_namespaces(manifest))
    findings.extend(_check_icon(manifest))
    findings.extend(_check_blueprints(manifest))
    findings.extend(_check_assets(manifest))
    findings.extend(_check_templates(manifest))
    findings.extend(_scan_sources(manifest))
    return StaticDoctorReport(
        report_version=DOCTOR_REPORT_VERSION,
        findings=tuple(findings),
    )


def diagnose(
    manifest_path: Path | str,
    *,
    probe_jail: ProbeJail | None = None,
    interpreter: Path | str | None = None,
    environment: Mapping[str, str] | None = None,
) -> DoctorReport:
    """Run every check against one pack and return the structured report.

    ``probe_jail`` (from ``detect_probe_jail()``) runs the import probe -
    the one stage that executes pack code - inside a network-less,
    rlimit-bounded bwrap jail. The static checks need no jail: they only
    read bytes. Findings are identical either way; the jail changes what a
    hostile pack can DO during the probe, never what the report means."""
    return _check_pack(
        manifest_path,
        authoring_checks=True,
        probe_jail=probe_jail,
        interpreter=interpreter,
        environment=environment,
    )


def prepare_catalog(
    manifest_path: Path | str,
    *,
    probe_jail: ProbeJail | None = None,
    interpreter: Path | str | None = None,
    environment: Mapping[str, str] | None = None,
) -> DoctorReport:
    """Probe trusted installed code and refresh its runtime schema catalog.

    This validates runtime declarations, not pack-authoring policy. Publishers
    and installation admission must use the complete ``diagnose()`` checks.
    """
    return _check_pack(
        manifest_path,
        authoring_checks=False,
        probe_jail=probe_jail,
        interpreter=interpreter,
        environment=environment,
    )


def _check_pack(
    manifest_path: Path | str,
    *,
    authoring_checks: bool,
    probe_jail: ProbeJail | None,
    interpreter: Path | str | None,
    environment: Mapping[str, str] | None,
) -> DoctorReport:
    from .catalog import catalog_path, source_digest, write_catalog

    catalog_path(Path(manifest_path).resolve()).unlink(missing_ok=True)
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError as exc:
        return DoctorReport(
            pack_name="<unknown>",
            manifest_path=str(manifest_path),
            findings=(
                Finding(
                    severity="error",
                    code="manifest.invalid",
                    message=str(exc),
                    fix="a pack ships a dinkster-pack.toml with [pack] name and "
                    "[pack.entry] nodes at minimum",
                ),
            ),
        )

    source = source_digest(manifest)
    findings = list(diagnose_static(manifest.root, manifest).findings) if authoring_checks else []
    # Timestamp-valid bytecode can describe old schemas after a same-size edit.
    for source_file in manifest.root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in source_file.relative_to(manifest.root).parts):
            continue
        for bytecode in (source_file.parent / "__pycache__").glob(f"{source_file.stem}.*.pyc"):
            bytecode.unlink(missing_ok=True)

    node_types: tuple[str, ...] = ()
    docs_coverage: DocsCoverage | None = None
    declaration_findings: list[Finding] = []
    probed = _run_probe(manifest, probe_jail, interpreter, environment)
    if isinstance(probed, Finding):
        findings.append(probed)
    else:
        declaration_findings.extend(_probe_findings(probed, manifest))
        probed_nodes = tuple(cast("dict[str, Any]", node) for node in probed["nodes"])
        node_types = tuple(cast("str", node["node_type"]) for node in probed_nodes)
        declaration_findings.extend(_check_namespace_coverage(manifest, node_types))
        docs_findings, docs_coverage = _check_docs(manifest, probed_nodes)
        declaration_findings.extend(docs_findings)
        findings.extend(declaration_findings)
        if not node_types and not any(f.severity == "error" for f in findings):
            findings.append(
                Finding(
                    severity="warning",
                    code="entry.no-nodes",
                    message="the pack loads but declares zero nodes",
                    fix="entry.nodes resolved to an empty sequence",
                )
            )

    if not isinstance(probed, Finding) and "catalog" in probed:
        if source != source_digest(manifest):
            findings.append(
                Finding(
                    "error",
                    "catalog.source-changed",
                    "pack files changed during doctor",
                    "stop editing the pack and rerun doctor",
                )
            )
        elif not any(f.severity == "error" for f in declaration_findings):
            write_catalog(manifest, probed["catalog"], source=source)
    elif not authoring_checks and not isinstance(probed, Finding):
        findings.append(
            Finding(
                "error",
                "catalog.missing",
                "the runtime probe did not return a schema catalog",
                "use the execution environment matching this backend release",
            )
        )

    return DoctorReport(
        pack_name=manifest.name,
        manifest_path=str(manifest.path),
        findings=tuple(findings),
        node_types=node_types,
        docs_coverage=docs_coverage,
    )


_SEVERITY_TAG = {"error": "ERROR", "warning": "WARN ", "info": "INFO "}


def render_text(report: DoctorReport) -> str:
    lines = [f"dinkster doctor: {report.pack_name} ({report.manifest_path})"]
    if report.node_types:
        lines.append(f"  nodes: {', '.join(report.node_types)}")
    if report.docs_coverage is not None:
        lines.append("  docs coverage:")
        lines.append("    locale  nodes")
        for locale, count in report.docs_coverage.locale_counts:
            lines.append(f"    {locale:<6}  {count}/{report.docs_coverage.total_nodes}")
    for finding in report.findings:
        location = f" [{finding.location}]" if finding.location else ""
        lines.append(
            f"  {_SEVERITY_TAG[finding.severity]} {finding.code}: {finding.message}{location}"
        )
        if finding.fix:
            lines.append(f"        fix: {finding.fix}")
    errors = sum(1 for f in report.findings if f.severity == "error")
    warnings = sum(1 for f in report.findings if f.severity == "warning")
    verdict = "healthy" if report.ok else "unhealthy"
    lines.append(f"  {verdict}: {errors} error(s), {warnings} warning(s)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    sandboxed = "--sandbox" in args
    args = [a for a in args if a not in ("--json", "--sandbox")]
    interpreter: str | None = None
    if "--python" in args:
        index = args.index("--python")
        if index + 1 >= len(args):
            print("dinkster-doctor: --python requires a path", file=sys.stderr)
            return 2
        interpreter = args[index + 1]
        del args[index : index + 2]
    if len(args) != 1:
        print(
            "usage: dinkster-doctor [--json] [--sandbox] [--python PATH] <pack-dir-or-manifest>",
            file=sys.stderr,
        )
        return 2
    probe_jail: ProbeJail | None = None
    if sandboxed:
        try:
            probe_jail = detect_probe_jail()
        except SandboxError as exc:
            # A requested jail never degrades to an unjailed probe.
            print(f"dinkster-doctor: --sandbox refused: {exc}", file=sys.stderr)
            return 2
    target = Path(args[0])
    if target.is_dir():
        target = target / "dinkster-pack.toml"
    if interpreter is None:
        report = diagnose(target, probe_jail=probe_jail)
    else:
        report = diagnose(target, probe_jail=probe_jail, interpreter=interpreter)
    print(report.to_json() if as_json else render_text(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
