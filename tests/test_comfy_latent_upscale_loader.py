"""Torch-free public schema checks for latent-upscale model loading."""

from dinkster_api.v1 import (
    ASSET_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    AssetWidget,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_models import NODES
from dinkster_models.latent_upscale_loader import LatentUpscaleModelLoader
from dinkster_schema import claim_covers
from dinkster_workers import load_manifest


def test_latent_upscale_loader_public_schema_and_manifest_order() -> None:
    schema = LatentUpscaleModelLoader.schema()
    assert NODES.count(LatentUpscaleModelLoader) == 1
    assert schema.node_type == "dinkster.models.latent_upscale_model_loader"
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
    manifest = load_manifest("packages/dinkster-nodes-models/dinkster-pack.toml")
    assert all(
        any(claim_covers(claim, node.schema().node_type) for claim in manifest.namespaces)
        for node in NODES
    )
