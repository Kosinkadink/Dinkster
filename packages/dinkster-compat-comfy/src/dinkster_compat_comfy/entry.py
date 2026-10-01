"""Manifest entry points for ComfyUI application-node compatibility."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_native.native import NATIVE_NODES, merge_native_nodes, register_native_types
from dinkster_schema import Node
from dinkster_values import TypeRegistry
from dinkster_workers import CompatGateDiagnostic

from .bootstrap import load_comfyui_nodes
from .devices import comfy_resident_meta
from .pool import default_pool

_COMPAT_ARM_NODE_TYPES = (
    "comfy.BlockSparseAttention",
    "dinkster.minimax_h3_cache_dit",
    "dinkster.load_checkpoint",
    "dinkster.load_model_patch",
    "dinkster.apply_minimax_h3_fun_controlnet",
    "dinkster.load_diffusion_model",
    "dinkster.clip_text_encode",
    "dinkster.empty_latent_image",
    "dinkster.ksampler",
    "dinkster.vae_decode",
    "dinkster.load_clip",
    "dinkster.load_vae",
    "dinkster.empty_minimax_h3_av",
    "dinkster.minimax_h3_t2va_conditioning",
    "dinkster.minimax_h3_image_to_video",
    "dinkster.separate_av_latent",
    "dinkster.vae_decode_audio",
    "dinkster.image_crop_to_mask",
    "dinkster.preview_mask",
    "dinkster.voxel_to_mesh",
    "dinkster.get_mesh_info",
    "dinkster.remesh_mesh",
    "dinkster.decimate_mesh",
    "dinkster.smooth_mesh_normals",
    "dinkster.unwrap_mesh",
    "dinkster.paint_mesh",
    "dinkster.bake_texture_from_voxel",
    "dinkster.bake_normal_map_from_mesh",
    "dinkster.bake_ambient_occlusion",
    "dinkster.render_uv_atlas",
    "dinkster.apply_texture_to_mesh",
    "dinkster.mesh_to_model3d",
)
_COMPAT_ARM_NODE_TYPE_SET = frozenset(_COMPAT_ARM_NODE_TYPES)
_TRANSLATION = load_comfyui_nodes()
COMFY_NODES: tuple[type[Node], ...] = tuple(
    node
    for node in merge_native_nodes(_TRANSLATION.node_classes)
    if node not in NATIVE_NODES or node.schema().node_type in _COMPAT_ARM_NODE_TYPE_SET
)
_NATIVE_NODES_BY_TYPE = {node.schema().node_type: node for node in NATIVE_NODES}
ARM_NODES = {
    "native": tuple(_NATIVE_NODES_BY_TYPE[node_type] for node_type in _COMPAT_ARM_NODE_TYPES)
}


def translation_skips() -> Mapping[str, CompatGateDiagnostic]:
    return dict(_TRANSLATION.diagnostics)


def combo_choices() -> Mapping[str, Sequence[str]]:
    samplers = cast("Any", importlib.import_module("dinkster_inference.samplers"))
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
