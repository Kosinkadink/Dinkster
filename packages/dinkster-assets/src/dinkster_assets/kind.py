"""Asset kinds: an open, namespaced vocabulary (DESIGN 3.12 + roadmap
"templates/asset distribution").

A kind says what an asset *is for* - the axis pickers filter on and
substitution UX groups by - without repeating ComfyUI's folder taxonomy
(where "kind" was literally which directory a file sat in, so a new model
family meant patching folder_paths). Kinds are plain namespaced strings,
``namespace/name`` with optional deeper segments: ``media/image``,
``model/lora``, ``model/diffusion``. The vocabulary is OPEN - packs and
templates may mint kinds freely - while the grammar is closed, so kinds
sort, compare, and prefix-filter mechanically forever.

Kind is presentation/index metadata, never execution identity: it joins
widgets, needs, and catalogs, and it never joins schema signatures or
cache keys.
"""

from __future__ import annotations

import re

from .identity import AssetError

_KIND_RE = re.compile(r"^[a-z0-9][a-z0-9-]*(/[a-z0-9][a-z0-9-]*)+$")

# Conventional core kinds. A convention, not an enum: validation accepts
# any grammatical kind, and unknown kinds are first-class everywhere.
KIND_MEDIA_IMAGE = "media/image"
KIND_MEDIA_VIDEO = "media/video"
KIND_MEDIA_AUDIO = "media/audio"
KIND_MEDIA_MODEL3D = "media/model3d"
KIND_MODEL_CHECKPOINT = "model/checkpoint"
KIND_MODEL_DIFFUSION = "model/diffusion"
KIND_MODEL_LORA = "model/lora"
KIND_MODEL_VAE = "model/vae"
KIND_MODEL_TEXT_ENCODER = "model/text-encoder"
KIND_MODEL_EMBEDDING = "model/embedding"


def is_asset_kind(text: str) -> bool:
    return _KIND_RE.match(text) is not None


def require_asset_kind(text: str) -> str:
    """Validate a kind string: lowercase alphanumeric/hyphen segments,
    '/'-separated, at least two segments (a bare word is not namespaced)."""
    if not is_asset_kind(text):
        raise AssetError(
            "not a valid asset kind (expected 'namespace/name' with "
            f"lowercase [a-z0-9-] segments): {text!r}"
        )
    return text
