"""First-party image operations with portable media and layer values."""

from functools import partial

from dinkster_api.v1 import (
    ASSET_TYPE,
    COLOR_TYPE,
    LAYERS_TYPE,
    TypeRegistry,
    register_asset_type,
    register_box_types,
    register_media_types,
    resolver_from_env,
)
from dinkster_values import (
    decode_latent,
    encode_latent,
    latent_fingerprint,
    validate_latent_encoded,
)

from .ported import NODES


def register_types(registry: TypeRegistry) -> None:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_media_types(registry)
    register_box_types(registry)
    if COLOR_TYPE not in registry:
        registry.register(COLOR_TYPE)
    if LAYERS_TYPE not in registry:
        registry.register(
            LAYERS_TYPE,
            encode=encode_latent,
            decode=partial(decode_latent, require_samples=False),
            fingerprint=latent_fingerprint(LAYERS_TYPE),
            validate_encoded=partial(validate_latent_encoded, require_samples=False),
        )


__all__ = ["NODES", "register_types"]
