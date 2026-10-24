"""Public interfaces of image-edit conditioning nodes, without Torch."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dinkster_api.v1 import CONDITIONING_TYPE, IMAGE_TYPE, LATENT_TYPE, VAE_TYPE, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.reference_conditioning import (
    InstructPixToPixConditioning,
    ReferenceLatent,
)


def test_reference_conditioning_interfaces_and_manifest() -> None:
    manifest = tomllib.loads(
        (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        ).read_text()
    )
    for node in (InstructPixToPixConditioning, ReferenceLatent):
        assert node in COMFY_NODES
        assert node.__name__ in manifest["pack"]["namespaces"]
    image_edit = InstructPixToPixConditioning.define_schema()
    assert [(port.id, port.type) for port in image_edit.inputs] == [
        ("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
        ("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
        ("vae", TypeExpr.concrete(VAE_TYPE)),
        ("pixels", TypeExpr.concrete(IMAGE_TYPE)),
    ]
    assert [(port.id, port.type) for port in image_edit.outputs] == [
        ("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
        ("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
        ("latent", TypeExpr.concrete(LATENT_TYPE)),
    ]
    reference = ReferenceLatent.define_schema()
    assert not reference.inputs[1].required
    conditioning = [["embedding", {"reference_latents": ["first"], "strength": 0.7}]]
    assert ReferenceLatent.execute(conditioning=conditioning)["CONDITIONING"] is conditioning
