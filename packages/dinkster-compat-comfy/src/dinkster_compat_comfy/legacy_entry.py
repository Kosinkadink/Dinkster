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

from dinkster_native.fork_nodes import FORK_NODES
from dinkster_values import TypeRegistry
from dinkster_workers import CompatGateDiagnostic

from . import entry
from .devices import comfy_resident_meta
from .legacy import load_legacy_packs
from .pool import default_pool

_TRANSLATION, LEGACY_REPORTS = load_legacy_packs()

LEGACY_NODES = (*entry.COMFY_NODES, *_TRANSLATION.node_classes)
_LEGACY_ARM_NODE_TYPES = (
    "dinkster.load_checkpoint",
    "dinkster.load_model_patch",
    "dinkster.apply_minimax_h3_fun_controlnet",
    "dinkster.load_diffusion_model",
    "dinkster.clip_text_encode",
    "dinkster.empty_latent_image",
    "dinkster.temporal_window_plan",
    "dinkster.spatial_tile_plan",
    "dinkster.explicit_window_plan",
    "dinkster.res4lyf_rk_beta_sampler",
    "dinkster.ksampler",
    "dinkster.vae_decode",
    "dinkster.load_clip",
    "dinkster.load_vae",
    "dinkster.empty_minimax_h3_av",
    "dinkster.minimax_h3_t2va_conditioning",
    "dinkster.minimax_h3_image_to_video",
    "dinkster.separate_av_latent",
    "dinkster.vae_decode_audio",
)
_FORK_NODES_BY_TYPE = {node.schema().node_type: node for node in FORK_NODES}
ARM_NODES = {
    "native": tuple(_FORK_NODES_BY_TYPE[node_type] for node_type in _LEGACY_ARM_NODE_TYPES)
}
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
