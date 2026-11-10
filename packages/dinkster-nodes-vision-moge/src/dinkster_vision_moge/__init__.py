"""ComfyUI-origin nodes with independent first-party identities."""

from functools import partial

from dinkster_api.v1 import (
    ASSET_TYPE,
    MOGE_GEOMETRY_TYPE,
    MOGE_MODEL_TYPE,
    TypeRegistry,
    comfy_resident_meta,
    decode_latent,
    default_pool,
    encode_latent,
    inference_nodes,
    latent_fingerprint,
    register_asset_type,
    register_media_types,
    register_mesh_type,
    register_resident_type,
    resolver_from_env,
    validate_latent_encoded,
)

from .loader import LoadMoGeModel
from .moge import MoGeGeometryToFOV
from .moge_inference import MoGeInference
from .moge_mesh import MoGePointMapToMesh
from .moge_panorama import MoGePanoramaInference
from .moge_render import MoGeRender

NODES = (
    LoadMoGeModel,
    MoGeGeometryToFOV,
    MoGeInference,
    MoGePanoramaInference,
    MoGePointMapToMesh,
    MoGeRender,
)


def register_types(registry: TypeRegistry) -> None:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    register_media_types(registry)
    register_mesh_type(registry)
    if MOGE_MODEL_TYPE not in registry:
        register_resident_type(
            registry, MOGE_MODEL_TYPE, table=default_pool(), meta=comfy_resident_meta
        )
    if MOGE_GEOMETRY_TYPE not in registry:
        registry.register(
            MOGE_GEOMETRY_TYPE,
            encode=encode_latent,
            decode=partial(decode_latent, require_samples=False),
            fingerprint=latent_fingerprint(MOGE_GEOMETRY_TYPE),
            validate_encoded=partial(validate_latent_encoded, require_samples=False),
        )


inference_nodes(NODES)
