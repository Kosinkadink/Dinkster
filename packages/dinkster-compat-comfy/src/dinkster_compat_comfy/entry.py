"""Manifest entry points for ComfyUI application-node compatibility."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_native.fork_nodes import FORK_NODES
from dinkster_native.native import merge_native_nodes, register_native_types
from dinkster_schema import Node
from dinkster_values import TypeRegistry
from dinkster_workers import CompatGateDiagnostic

from .bootstrap import load_comfyui_nodes
from .devices import comfy_resident_meta
from .pool import default_pool

_TRANSLATION = load_comfyui_nodes()
COMFY_NODES: tuple[type[Node], ...] = merge_native_nodes(_TRANSLATION.node_classes)
ARM_NODES = {"native": FORK_NODES}


def translation_skips() -> Mapping[str, CompatGateDiagnostic]:
    return dict(_TRANSLATION.diagnostics)


def combo_choices() -> Mapping[str, Sequence[str]]:
    samplers = cast("Any", importlib.import_module("dinkster_comfy.samplers"))
    folder_paths = cast("Any", importlib.import_module("folder_paths"))
    return {
        "comfy.samplers": tuple(str(name) for name in samplers.KSampler.SAMPLERS),
        "comfy.schedulers": tuple(str(name) for name in samplers.KSampler.SCHEDULERS),
        **_TRANSLATION.listing_snapshots,
        "comfy.files.embeddings": cast(
            "Sequence[str]", folder_paths.get_filename_list("embeddings")
        ),
        "comfy.files.loras": cast("Sequence[str]", folder_paths.get_filename_list("loras")),
    }


def register_types(registry: TypeRegistry) -> None:
    _TRANSLATION.register_types(
        registry,
        resident_meta=comfy_resident_meta,
        table=default_pool(),
    )
    register_native_types(registry)
