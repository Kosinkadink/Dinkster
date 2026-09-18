"""Comfy v1 resident types over the shared dinkster-values resident mechanism.

The mechanism (ResidencyTable, resident-stub codec, lookup errors) lives in
``dinkster_values.resident``; this module keeps the comfy-specific vocabulary
and re-exports the mechanism for existing callers.
"""

from __future__ import annotations

from dinkster_values import (
    ResidencyTable,
    ResidentLookupError,
    register_resident_type,
    resident_resource_id,
)

DEFAULT_RESIDENT_V1_TYPES = frozenset(
    {
        "MODEL",
        "MODEL_PATCH",
        "CLIP",
        "VAE",
        "CONTROL_NET",
        "CLIP_VISION",
        "STYLE_MODEL",
        "GLIGEN",
        "UPSCALE_MODEL",
        "PHOTOMAKER",
        "MOGE_MODEL",
        "BACKGROUND_REMOVAL",
    }
)
"""v1 types whose values are loaded hardware state, not data."""


__all__ = [
    "DEFAULT_RESIDENT_V1_TYPES",
    "ResidencyTable",
    "ResidentLookupError",
    "register_resident_type",
    "resident_resource_id",
]
