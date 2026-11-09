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
from dinkster_hunyuan import NODES as HUNYUAN_NODES
from dinkster_hunyuan.latent_upscale import HunyuanVideo15LatentUpscaleWithModel
from dinkster_ltx import NODES as LTX_NODES
from dinkster_ltx.latent_upscale import LTXVLatentUpsampler
from dinkster_schema import claim_covers


def test_latent_upscale_public_schemas_are_exact() -> None:
    hunyuan, ltx = (HunyuanVideo15LatentUpscaleWithModel.schema(), LTXVLatentUpsampler.schema())
    assert [(schema.node_type, schema.category) for schema in (hunyuan, ltx)] == [
        (
            "dinkster.hunyuan.hunyuan_video15_latent_upscale_with_model",
            "model/latent/hunyuan video",
        ),
        ("dinkster.ltx.ltxv_latent_upsampler", "model/latent/ltxv"),
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
    for pack, node, nodes in (
        ("hunyuan", HunyuanVideo15LatentUpscaleWithModel, HUNYUAN_NODES),
        ("ltx", LTXVLatentUpsampler, LTX_NODES),
    ):
        manifest = tomllib.loads(
            (
                Path(__file__).parents[1] / f"packages/dinkster-nodes-{pack}/dinkster-pack.toml"
            ).read_text(encoding="ascii")
        )
        assert nodes.count(node) == 1
        assert all(
            any(
                claim_covers(claim, entry.schema().node_type)
                for claim in manifest["pack"]["namespaces"]
            )
            for entry in nodes
        )
