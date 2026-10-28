"""Torch-free public schema checks for latent-upscale model loading."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    AssetWidget,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.latent_upscale_loader import (
    LATENT_UPSCALE_LOADER_NODES,
    LatentUpscaleModelLoader,
)
from dinkster_workers import load_manifest


def test_latent_upscale_loader_public_schema_and_manifest_order() -> None:
    schema = LatentUpscaleModelLoader.schema()
    assert LATENT_UPSCALE_LOADER_NODES == (LatentUpscaleModelLoader,)
    assert schema.node_type == "LatentUpscaleModelLoader"
    assert schema.display_name == "Load Latent Upscale Model"
    assert schema.category == "model/loaders"
    assert tuple((item.id, item.type.types) for item in schema.inputs) == (
        ("model_name", (ASSET_TYPE,)),
    )
    assert schema.inputs[0].widget == AssetWidget(
        accept=("application/octet-stream",), kind="model/latent-upscaler"
    )
    assert tuple((item.id, item.type.types) for item in schema.outputs) == (
        ("LATENT_UPSCALE_MODEL", (LATENT_UPSCALE_MODEL_TYPE,)),
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert LatentUpscaleModelLoader in COMFY_NODES
    assert (
        tuple(node.schema().node_type for node in COMFY_NODES)
        == load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml").namespaces
    )
