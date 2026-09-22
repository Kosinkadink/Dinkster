"""Manifest entry points for the compat pack (imported in the child only).

The native-only provider uses pinned import metadata and native bodies.
The optional compatibility provider bootstraps ComfyUI in its own worker;
the engine never imports that runtime.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_inference import (
    CLIP_TYPE_ID,
    CLIP_VISION_TYPE_ID,
    CONDITIONING_TYPE_ID,
    MODEL_TYPE_ID,
    VAE_TYPE_ID,
    register_conditioning_type,
    register_inference_types,
)
from dinkster_model_triposplat.types import (  # pyright: ignore[reportMissingTypeStubs]
    register_triposplat_types,
)
from dinkster_schema import Node
from dinkster_values import TypeRegistry, register_resident_type
from dinkster_workers import CompatGateDiagnostic

from .bootstrap import load_comfyui_nodes
from .comfy_execution import model_unload
from .devices import comfy_resident_meta
from .model3d import COMFY_MODEL3D_NODES
from .native import SAMPLER_CHOICES, SCHEDULER_CHOICES, merge_native_nodes, register_native_types
from .native_arm import GENERATION_PROVIDER_NODES, NATIVE_ARM_NODES, NATIVE_SCHEDULING_NODES
from .native_catalog import (
    COMFY_RUNTIME_NODE_IDS,
    NATIVE_SCHEDULING_SOURCE_NODE_NAMES,
)
from .pool import configure_compat_unload, default_pool
from .sampling import COMFY_SAMPLING_NODES
from .schema_snapshot import core_schema_snapshot
from .translate import CompatTranslation

log = logging.getLogger("dinkster.compat.comfy.entry")

configure_compat_unload(model_unload)

_ARM_SOURCE_NODES = tuple(NATIVE_SCHEDULING_SOURCE_NODE_NAMES.values())

_NATIVE_ONLY = os.environ.get("DINKSTER_COMFY_NATIVE_ONLY") == "1"
_TRANSLATION = (
    CompatTranslation() if _NATIVE_ONLY else load_comfyui_nodes(required=_ARM_SOURCE_NODES)
)
if _NATIVE_ONLY:
    _TRANSLATION.opaque_types.update(core_schema_snapshot()["opaqueTypes"])

# Native scheduling schemas replace their translated source schemas. Their
# aliases preserve source prompt lowering while their canonical IDs keep the
# source namespace out of executable catalogs.
_scheduling_source_types = {f"comfy.{name}" for name in _ARM_SOURCE_NODES}
_comfy_execution_by_type = {
    node.schema().node_type: node for node in (*COMFY_MODEL3D_NODES, *COMFY_SAMPLING_NODES)
}


def _with_comfy_execution(nodes: Sequence[type[Node]]) -> tuple[type[Node], ...]:
    return tuple(_comfy_execution_by_type.get(node.schema().node_type, node) for node in nodes)


_default_nodes = (
    *(
        node
        for node in _with_comfy_execution(merge_native_nodes(_TRANSLATION.node_classes))
        if node.schema().node_type not in _scheduling_source_types
    ),
    *NATIVE_SCHEDULING_NODES,
    *_with_comfy_execution(GENERATION_PROVIDER_NODES),
)
# Keep arm declarations and control-plane identities while selecting native
# bodies by default when the optional upstream runtime is absent.
COMFY_NODES = (
    tuple(
        {
            node.schema().node_type: node
            for node in (*_default_nodes, *NATIVE_ARM_NODES)
            if node.schema().node_type not in COMFY_RUNTIME_NODE_IDS
        }.values()
    )
    if _NATIVE_ONLY
    else _default_nodes
)

# Same-session alternatives to the default Comfy-backed bodies. The host
# proves these classes have exactly the manifest-declared node types and the
# default classes' schema signatures before it admits the worker.
ARM_NODES = {
    "native": tuple(
        node
        for node in _with_comfy_execution(NATIVE_ARM_NODES)
        if not _NATIVE_ONLY or node.schema().node_type not in COMFY_RUNTIME_NODE_IDS
    )
}


def translation_skips() -> Mapping[str, CompatGateDiagnostic]:
    """Classified source-node refusals for instance diagnostics."""
    return dict(_TRANSLATION.diagnostics)


def combo_choices() -> Mapping[str, Sequence[str]]:
    """Combo choice lists this worker serves behind /api/choices/{id}.

    Enumerated at worker startup from the live ComfyUI import in THIS
    process - the same lists the translated and native KSampler consume.
    Embeddings and LoRAs are first-class prompt inventories read directly
    from the live folder_paths registry. Every other filesystem listing
    the translation probe matched to a combo rides here under its
    ``comfy.files.<category>`` id.
    Scope caveat: legacy custom packs run in their own workers, so a
    pack that monkey-patches sampler lists over there is invisible here;
    a first-class cross-worker sampler registry is a tracked roadmap
    item, not this seam."""
    if _NATIVE_ONLY:
        return {"comfy.samplers": SAMPLER_CHOICES, "comfy.schedulers": SCHEDULER_CHOICES}
    samplers = cast("Any", importlib.import_module("comfy.samplers"))
    ksampler = getattr(samplers, "KSampler", None)
    sampler_names = getattr(ksampler, "SAMPLERS", None)
    scheduler_names = getattr(ksampler, "SCHEDULERS", None)
    if sampler_names is None or scheduler_names is None:
        log.warning(
            "DINKSTER_COMPAT_SAMPLER_CHOICES_UNAVAILABLE: ComfyUI does not expose "
            "KSampler.SAMPLERS/SCHEDULERS; serving empty choice lists"
        )
    folder_paths = cast("Any", importlib.import_module("folder_paths"))
    return {
        "comfy.samplers": tuple(str(name) for name in sampler_names or ()),
        "comfy.schedulers": tuple(str(name) for name in scheduler_names or ()),
        **_TRANSLATION.listing_snapshots,
        "comfy.files.embeddings": cast(
            "Sequence[str]", folder_paths.get_filename_list("embeddings")
        ),
        "comfy.files.loras": cast("Sequence[str]", folder_paths.get_filename_list("loras")),
    }


def register_types(registry: TypeRegistry) -> None:
    # The resident codec writes into the governed pool, not a bare table:
    # every model that crosses the boundary is thereby accounted, LRU-
    # tracked, and sheddable (device state only; stubs keep resolving).
    _TRANSLATION.register_types(registry, resident_meta=comfy_resident_meta, table=default_pool())
    register_native_types(registry)
    register_triposplat_types(registry)
    for type_id in (
        MODEL_TYPE_ID,
        CLIP_TYPE_ID,
        CLIP_VISION_TYPE_ID,
        VAE_TYPE_ID,
    ):
        if type_id not in registry:
            register_resident_type(
                registry,
                type_id,
                table=default_pool(),
                meta=comfy_resident_meta,
            )
    if CONDITIONING_TYPE_ID not in registry:
        register_conditioning_type(
            registry,
            resident_table=default_pool(),
            resident_meta=comfy_resident_meta,
        )
    register_inference_types(registry)
