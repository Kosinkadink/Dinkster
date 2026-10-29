"""Manifest entry points for the legacy custom-pack quarantine worker.

``dinkster-legacy-pack.toml`` names LEGACY_NODES and register_types here;
importing this module bootstraps ComfyUI *and* imports arbitrary custom
pack code, so it must only ever run inside a compat worker child (hazard
H5). Which packs load is operator configuration via DINKSTER_LEGACY_PACKS -
see legacy.py for the full environment contract and the per-pack
diagnostic report.

Unlike entry.py this deliberately does not re-export the translated core
nodes: a graph that needs core compat nodes and legacy pack nodes runs
one worker per manifest, and the engine composes them - keeping "core
ComfyUI surface" and "arbitrary downloaded code" separable placement and
sandbox-policy decisions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from dinkster_values import TypeRegistry
from dinkster_workers import CompatGateDiagnostic

from .devices import comfy_resident_meta
from .legacy import load_legacy_packs
from .pool import default_pool

_TRANSLATION, LEGACY_REPORTS = load_legacy_packs()

LEGACY_NODES = tuple(_TRANSLATION.node_classes)


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
    _TRANSLATION.register_types(registry, resident_meta=comfy_resident_meta, table=default_pool())
