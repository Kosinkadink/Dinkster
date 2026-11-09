"""First-party image operations with portable media and layer values."""

from functools import partial

from dinkster_api.v1 import (
    ASSET_TYPE,
    BOUNDING_BOXES_TYPE,
    COLOR_TYPE,
    COMPOSITOR_TYPE,
    IMAGECOMPARE_TYPE,
    LAYERS_TYPE,
    WEBCAM_TYPE,
    TypeRegistry,
    decode_latent,
    encode_latent,
    inference_nodes,
    latent_fingerprint,
    register_asset_type,
    register_box_types,
    register_media_types,
    resolver_from_env,
    validate_latent_encoded,
)

from .bounding_boxes import CreateBoundingBoxes
from .image_compositor import ImageCompositor
from .layer_recipe import coerce_delta, decode_delta, encode_delta, validate_delta_encoded
from .layers_from_bounding_boxes import LayersFromBoundingBoxes
from .ported import NODES as PORTED_NODES
from .webcam import WebcamCapture

NODES = (
    *PORTED_NODES,
    CreateBoundingBoxes,
    LayersFromBoundingBoxes,
    ImageCompositor,
    WebcamCapture,
)


def register_types(registry: TypeRegistry) -> None:
    if ASSET_TYPE not in registry:
        register_asset_type(registry, resolver_from_env())
    if WEBCAM_TYPE not in registry:
        register_asset_type(registry, resolver_from_env(), type_id=WEBCAM_TYPE)
    register_media_types(registry)
    register_box_types(registry)
    if COMPOSITOR_TYPE not in registry:
        registry.register(
            COMPOSITOR_TYPE,
            encode=encode_delta,
            decode=decode_delta,
            coerce=coerce_delta,
            meta=lambda obj: {"version": coerce_delta(obj)["version"]},
            validate_encoded=validate_delta_encoded,
            asset_references_complete=True,
        )
    for type_id in (BOUNDING_BOXES_TYPE, COLOR_TYPE, IMAGECOMPARE_TYPE):
        if type_id not in registry:
            registry.register(type_id)
    if LAYERS_TYPE not in registry:
        registry.register(
            LAYERS_TYPE,
            encode=encode_latent,
            decode=partial(decode_latent, require_samples=False),
            fingerprint=latent_fingerprint(LAYERS_TYPE),
            validate_encoded=partial(validate_latent_encoded, require_samples=False),
        )


__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
