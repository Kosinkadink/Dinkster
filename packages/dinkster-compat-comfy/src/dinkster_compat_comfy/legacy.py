"""Load unmodified ComfyUI custom-node packs and translate them (DESIGN 3.8).

Child process only, same hazard rules as bootstrap.py: the engine's
interpreter never imports pack code, and a pack that fails to load fails
*here*, quarantined, with a classified reason - never mysteriously.

What this is: a best-effort on-ramp so existing packs can run inside a
Dinkster compat worker today. What this is not: bug-for-bug ComfyUI
emulation. Packs get the same import environment ComfyUI gives them at
startup (sys.path rooted at the install, a real headless PromptServer
built from ComfyUI's own server.py, ComfyUI's own import mechanics), and
their v1 NODE_CLASS_MAPPINGS are translated through translate.py exactly
like core nodes. Anything past that - routes actually being served,
executor monkey-patching taking effect, IS_LIST calling conventions - is
reported in the pack's LegacyPackReport instead of half-working.

Import mechanics deliberately mirror ComfyUI's ``load_custom_node`` v1
path (nodes.py), which is synchronous: module name from the directory
basename, sys.modules key with ``.`` replaced by ``_x_``, spec from
``__init__.py`` (or the file itself for single-file packs), module
registered in sys.modules before exec so relative imports resolve. The
async half of ComfyUI's loader exists only for V3 ``comfy_entrypoint``
packs, which this loader classifies and skips: V3 packs are the
port-them-properly case (``dinkster port`` translates them too), not the
quarantine case. A mixed pack (v1 mappings AND a V3 entrypoint) loads
its v1 half and flags ``v3_entrypoint_ignored`` - upstream ComfyUI
ignores the V3 half of a mixed pack silently; Dinkster reports it.

Node ids are namespaced per pack: ``comfy.<pack>.<v1 name>``. v1's flat
global node namespace made collisions a runtime surprise; two legacy
packs shipping the same key coexist here.

Environment contract (all read at pack-load time, operator-owned - node
authors and node users never see these):

- ``DINKSTER_COMFYUI_ROOT``: ComfyUI install the packs were written against
  (required; bootstrap.py contract).
- ``DINKSTER_LEGACY_PACKS``: os.pathsep-separated pack paths, each a
  directory containing ``__init__.py`` or a single ``.py`` file
  (required for this manifest).
- ``DINKSTER_LEGACY_REPORT``: optional path; the structured per-pack report
  is written there as JSON so the parent side can read diagnostics
  without parsing logs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .bootstrap import bootstrap_comfyui, ensure_prompt_server
from .translate import CompatError, CompatTranslation, translate_mappings

if TYPE_CHECKING:
    from collections.abc import Mapping

#: ComfyUI runtime modules a pack may hook into; references from pack code
#: into these mark surfaces Dinkster does not serve (reported, not emulated).
_HOOK_MODULES = ("server", "execution", "aiohttp")


@dataclass
class LegacyPackReport:
    """What happened when one pack was loaded. Always produced, success or
    not: the report is the diagnostic surface DESIGN 3.8 demands."""

    path: str
    pack_id: str
    status: str = "loaded"
    """One of: loaded, missing-dependency, import-error, no-mappings,
    v3-entrypoint."""
    error: str = ""
    missing_module: str = ""
    nodes_translated: int = 0
    nodes_skipped: dict[str, str] = field(default_factory=dict[str, str])
    server_routes_added: int = 0
    """Routes the pack registered on the headless PromptServer: web surface
    the pack expects that Dinkster will not serve."""
    hook_imports: tuple[str, ...] = ()
    """ComfyUI runtime modules the pack's own code holds references into."""
    web_directory: str = ""
    """WEB_DIRECTORY declared by the pack: frontend assets Dinkster ignores."""
    v3_entrypoint_ignored: bool = False
    """The pack ships BOTH v1 mappings and a V3 ``comfy_entrypoint`` (a
    mixed pack). The runtime loads the v1 half only; upstream ComfyUI
    silently ignores the V3 half in this case, Dinkster says so. The V3
    nodes port via ``dinkster port``, which probes both halves."""


def _pack_id(path: Path) -> str:
    name = path.stem if path.is_file() else path.name
    return name


def _hook_references(sys_module_name: str) -> tuple[str, ...]:
    """Which _HOOK_MODULES the pack's module tree actually references.

    The runtime modules are always loaded before packs are (bootstrap
    stands up the headless server), so "newly imported during the pack's
    import" would observe nothing. What matters is whether the *pack's*
    namespaces hold objects from those modules - ``import execution``,
    ``from server import PromptServer`` - because that is the surface the
    pack will poke at runtime."""
    hooks: set[str] = set()
    prefix = sys_module_name + "."
    # Snapshot as plain objects: sys.modules is typed ModuleType-valued but
    # can legally hold None (import-blocking convention).
    snapshot = cast("Mapping[str, object]", dict(sys.modules))
    for name, module in snapshot.items():
        if not isinstance(module, types.ModuleType):
            continue
        if not (name == sys_module_name or name.startswith(prefix)):
            continue
        for value in vars(module).values():
            if isinstance(value, types.ModuleType):
                if value.__name__ in _HOOK_MODULES:
                    hooks.add(value.__name__)
                continue
            try:
                origin = getattr(value, "__module__", None)
            except Exception:  # noqa: BLE001 - arbitrary pack objects
                continue
            if isinstance(origin, str):
                root = origin.split(".", 1)[0]
                if root in _HOOK_MODULES:
                    hooks.add(root)
    return tuple(sorted(hooks))


def _route_count(server_instance: object | None) -> int:
    if server_instance is None:
        return 0
    routes = getattr(server_instance, "routes", None)
    try:
        return len(cast("list[object]", list(routes)))  # type: ignore[arg-type]
    except TypeError:
        return 0


def pack_sys_module_name(path: Path) -> str:
    """The sys.modules key one pack imports under - ComfyUI's own naming
    (directory path with ``.`` replaced, file path without suffix). Shared
    with port_probe.py so it can retrieve an imported pack's module."""
    if path.is_file():
        return str(path.with_suffix(""))
    return str(path).replace(".", "_x_")


def _import_pack(path: Path) -> types.ModuleType:
    """ComfyUI's load_custom_node v1 import mechanics, synchronous and
    scoped to one pack: no global NODE_CLASS_MAPPINGS mutation, no
    ignore-set, no web-dir registry."""
    sys_module_name = pack_sys_module_name(path)
    if path.is_file():
        spec = importlib.util.spec_from_file_location(sys_module_name, path)
    else:
        init = path / "__init__.py"
        if not init.is_file():
            raise CompatError(f"not a loadable pack (no __init__.py): {path}")
        spec = importlib.util.spec_from_file_location(sys_module_name, init)
    if spec is None or spec.loader is None:
        raise CompatError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[sys_module_name] = module
    spec.loader.exec_module(module)
    return module


def load_legacy_pack(
    path: Path,
    translation: CompatTranslation,
    *,
    server_instance: object | None,
) -> LegacyPackReport:
    """Import one unmodified pack and translate its v1 mappings into
    ``translation``. Never raises for pack faults - every outcome lands in
    the report; only Dinkster-side bugs escape."""
    report = LegacyPackReport(path=str(path), pack_id=_pack_id(path))
    routes_before = _route_count(server_instance)
    try:
        module = _import_pack(path)
    except ModuleNotFoundError as exc:
        report.status = "missing-dependency"
        report.missing_module = exc.name or ""
        report.error = f"{type(exc).__name__}: {exc}"
        return report
    except BaseException as exc:  # noqa: BLE001 - pack code is arbitrary
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        report.status = "import-error"
        report.error = traceback.format_exception_only(type(exc), exc)[-1].strip()
        return report
    finally:
        report.server_routes_added = max(0, _route_count(server_instance) - routes_before)

    report.hook_imports = _hook_references(module.__name__)

    web_directory = getattr(module, "WEB_DIRECTORY", None)
    if isinstance(web_directory, str) and web_directory:
        report.web_directory = web_directory

    mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
    if not mappings:
        if callable(getattr(module, "comfy_entrypoint", None)):
            report.status = "v3-entrypoint"
            report.error = (
                "pack uses the V3 comfy_entrypoint API; port it to a native "
                "Dinkster pack instead of the legacy quarantine"
            )
        else:
            report.status = "no-mappings"
            report.error = "pack exposes no NODE_CLASS_MAPPINGS"
        return report

    if callable(getattr(module, "comfy_entrypoint", None)):
        # Mixed pack: v1 mappings win (ComfyUI's own precedence), but the
        # V3 half must not vanish silently - upstream's behavior, not ours.
        report.v3_entrypoint_ignored = True

    display = cast(
        "Mapping[str, str]",
        getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None) or {},
    )
    already_skipped = set(translation.skipped)
    before_count = len(translation.node_classes)
    translate_mappings(
        cast("Mapping[str, type]", mappings),
        display_names=display,
        namespace=report.pack_id,
        translation=translation,
    )
    report.nodes_translated = len(translation.node_classes) - before_count
    report.nodes_skipped = {
        name.removeprefix(f"{report.pack_id}."): reason
        for name, reason in translation.skipped.items()
        if name not in already_skipped
    }
    return report


def load_legacy_packs() -> tuple[CompatTranslation, list[LegacyPackReport]]:
    """Bootstrap ComfyUI, then load every pack named by DINKSTER_LEGACY_PACKS.

    The report list always matches the requested packs one-to-one; if
    DINKSTER_LEGACY_REPORT names a path the reports are also written there as
    JSON for the parent side."""
    packs_text = os.environ.get("DINKSTER_LEGACY_PACKS", "")
    paths = [Path(p) for p in packs_text.split(os.pathsep) if p.strip()]
    if not paths:
        raise CompatError(
            "DINKSTER_LEGACY_PACKS is not set; the legacy pack manifest needs "
            "at least one custom-node pack path"
        )

    bootstrap_comfyui()
    server_instance = ensure_prompt_server()

    translation = CompatTranslation()
    reports: list[LegacyPackReport] = []
    for path in paths:
        if not path.exists():
            reports.append(
                LegacyPackReport(
                    path=str(path),
                    pack_id=_pack_id(path),
                    status="import-error",
                    error=f"pack path does not exist: {path}",
                )
            )
            continue
        reports.append(load_legacy_pack(path, translation, server_instance=server_instance))

    report_path = os.environ.get("DINKSTER_LEGACY_REPORT", "")
    if report_path:
        Path(report_path).write_text(
            json.dumps([asdict(r) for r in reports], indent=2), encoding="utf-8"
        )
    for report in reports:
        line = (
            f"dinkster-compat-legacy: {report.pack_id}: {report.status}"
            f" ({report.nodes_translated} nodes"
            f", {len(report.nodes_skipped)} skipped"
            f", {report.server_routes_added} routes)"
        )
        if report.v3_entrypoint_ignored:
            line += (
                " - pack also ships a V3 comfy_entrypoint; the runtime loads "
                "the v1 half only (port the V3 nodes with 'dinkster port')"
            )
        if report.error:
            line += f" - {report.error}"
        print(line, file=sys.stderr)
    return translation, reports


__all__ = [
    "LegacyPackReport",
    "load_legacy_pack",
    "load_legacy_packs",
]
