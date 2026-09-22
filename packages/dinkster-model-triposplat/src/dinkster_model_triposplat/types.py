"""TripoSplat resident component value types."""

from __future__ import annotations

from dinkster_values import TypeRegistry, register_resident_type, register_splat_type

TRIPOSPLAT_VISION_TYPE = "dinkster.triposplat_vision"
TRIPOSPLAT_DECODER_TYPE = "dinkster.triposplat_decoder"
SPLAT_TYPE = "dinkster.splat"


def register_triposplat_types(registry: TypeRegistry) -> None:
    """Register the worker-local component handles published by this pack."""

    for type_id in (TRIPOSPLAT_VISION_TYPE, TRIPOSPLAT_DECODER_TYPE):
        if type_id not in registry:
            register_resident_type(registry, type_id)
    register_splat_type(registry, SPLAT_TYPE)


__all__ = [
    "TRIPOSPLAT_DECODER_TYPE",
    "TRIPOSPLAT_VISION_TYPE",
    "SPLAT_TYPE",
    "register_triposplat_types",
]
