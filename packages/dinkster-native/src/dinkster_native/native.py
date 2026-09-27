"""Fork-backed native node and value-type registration."""

from __future__ import annotations

from collections.abc import Iterable

from dinkster_assets import (
    ASSET_TYPE,
    SAVE_TARGET_TYPE,
    register_asset_type,
    register_save_target_type,
    resolver_from_env,
)
from dinkster_inference import register_inference_types
from dinkster_inference.sampling_wire import register_sampling_type
from dinkster_schema import Node
from dinkster_values import TypeRegistry, register_curve_type, register_model3d_type

from .audio import register_audio_type
from .devices import comfy_resident_meta
from .fork_nodes import FORK_NODES
from .image import (
    register_image_asset_providers,
    register_image_type,
    register_image_type_equivalences,
)
from .latent import register_latent_type
from .pool import default_pool
from .resident import register_resident_type
from .video import register_video_type

NATIVE_NODES = FORK_NODES


def merge_native_nodes(translated: Iterable[type[Node]]) -> tuple[type[Node], ...]:
    """Replace translated nodes claimed by the fork-backed native nodes."""
    claimed: set[str] = set()
    for node in NATIVE_NODES:
        schema = node.schema()
        claimed.add(schema.node_type)
        claimed.update(f"comfy.{alias}" for alias in schema.aliases)
    return (
        *(node for node in translated if node.schema().node_type not in claimed),
        *NATIVE_NODES,
    )


def _register_resident(registry: TypeRegistry, type_id: str) -> None:
    if type_id not in registry:
        register_resident_type(
            registry,
            type_id,
            table=default_pool(),
            meta=comfy_resident_meta,
        )


def register_native_types(registry: TypeRegistry) -> None:
    """Register the wire and resident types used by native execution."""
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    if SAVE_TARGET_TYPE not in registry:
        register_save_target_type(registry)

    for type_id in ("dinkster.model", "dinkster.clip", "dinkster.vae"):
        _register_resident(registry, type_id)

    if "dinkster.latent" not in registry:
        register_latent_type(registry, "dinkster.latent")
    if "dinkster.conditioning" not in registry:
        registry.register("dinkster.conditioning")

    for type_id in ("dinkster.sampler", "dinkster.sigmas", "dinkster.noise"):
        register_sampling_type(registry, type_id)

    register_inference_types(registry)
    register_curve_type(registry)
    register_model3d_type(registry, "dinkster.model3d")

    for type_id in ("dinkster.image", "dinkster.mask"):
        if type_id not in registry:
            register_image_type(registry, type_id)
    register_image_type_equivalences(registry)
    if registry.asset_decoder_for("dinkster.image") is None:
        register_image_asset_providers(registry, "dinkster.image")

    if "comfy.AUDIO" not in registry:
        register_audio_type(registry, "comfy.AUDIO")
    if "dinkster.video" not in registry:
        register_video_type(registry, "dinkster.video")
