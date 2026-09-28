"""Manifest entry points for the legacy custom-pack quarantine worker.

``dinkster-legacy-pack.toml`` names LEGACY_NODES and register_types here;
importing this module bootstraps ComfyUI *and* imports arbitrary custom
pack code, so it must only ever run inside a compat worker child (hazard
H5). Which packs load is operator configuration via DINKSTER_LEGACY_PACKS -
see legacy.py for the full environment contract and the per-pack
diagnostic report.

Unlike entry.py this does not re-export translated core nodes. It does expose
the fork-backed generation arm so custom nodes can consume and return resident
models without moving them between the core and quarantine workers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from dinkster_values import TypeRegistry
from dinkster_workers import CompatGateDiagnostic

from . import entry
from .devices import comfy_resident_meta
from .legacy import load_legacy_packs
from .pool import default_pool

_TRANSLATION, LEGACY_REPORTS = load_legacy_packs()

LEGACY_NODES = (*entry.COMFY_NODES, *_TRANSLATION.node_classes)
ARM_NODES = entry.ARM_NODES
combo_choices = entry.combo_choices


def translation_skips() -> Mapping[str, CompatGateDiagnostic]:
    """Classified skips keyed like legacy node types for host attribution."""
    return {
        f"comfy.{name}": replace(diagnostic, source_node=f"comfy.{name}")
        for name, diagnostic in _TRANSLATION.diagnostics.items()
    }


def register_types(registry: TypeRegistry) -> None:
    # Same residency rules as the core compat pack: loaded-hardware-state
    # types stay in this process behind the governed pool; everything else
    # crosses with correct-everywhere defaults.
    entry.register_types(registry)
    _TRANSLATION.register_types(registry, resident_meta=comfy_resident_meta, table=default_pool())
