"""Torch-free public contracts for native latent upscale consumers."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dinkster_api.v1 import (
    LATENT_TYPE,
    LATENT_UPSCALE_MODEL_TYPE,
    VAE_TYPE,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.latent_upscale import LATENT_UPSCALE_NODES


def test_latent_upscale_public_schemas_are_exact() -> None:
    hunyuan, ltx = (node.schema() for node in LATENT_UPSCALE_NODES)
    assert [(schema.node_type, schema.category) for schema in (hunyuan, ltx)] == [
        ("HunyuanVideo15LatentUpscaleWithModel", "model/latent/hunyuan video"),
        ("LTXVLatentUpsampler", "model/latent/ltxv"),
    ]
    assert [(port.id, port.type.types) for port in hunyuan.inputs] == [
        ("model", (LATENT_UPSCALE_MODEL_TYPE,)),
        ("samples", (LATENT_TYPE,)),
        ("upscale_method", ("COMBO",)),
        ("width", ("INT",)),
        ("height", ("INT",)),
        ("crop", ("COMBO",)),
    ]
    assert [(port.id, port.type.types) for port in ltx.inputs] == [
        ("samples", (LATENT_TYPE,)),
        ("upscale_model", (LATENT_UPSCALE_MODEL_TYPE,)),
        ("vae", (VAE_TYPE,)),
    ]
    for schema in (hunyuan, ltx):
        assert schema_from_wire(schema_to_wire(schema)) == schema


def test_latent_upscale_registration_and_manifest_order_are_complete() -> None:
    manifest = tomllib.loads(
        (Path(__file__).parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml").read_text(
            encoding="ascii"
        )
    )
    registered = tuple(node.schema().node_type for node in COMFY_NODES)
    assert tuple(manifest["pack"]["namespaces"]) == ("devices", *registered)
    assert tuple(node.schema().node_type for node in LATENT_UPSCALE_NODES) == (
        "HunyuanVideo15LatentUpscaleWithModel",
        "LTXVLatentUpsampler",
    )
