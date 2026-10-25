"""Public interfaces of image-edit conditioning nodes, without Torch."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    NumberWidget,
    TypeExpr,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.reference_conditioning import (
    InstructPixToPixConditioning,
    ReferenceLatent,
    ReferenceMultiLatent,
)


def test_reference_conditioning_interfaces_and_manifest() -> None:
    manifest = tomllib.loads(
        (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        ).read_text()
    )
    for node in (InstructPixToPixConditioning, ReferenceLatent, ReferenceMultiLatent):
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


def test_reference_multi_latent_exposes_resize_and_optional_ordered_images() -> None:
    schema = ReferenceMultiLatent.schema()
    assert [(port.id, port.required) for port in schema.inputs] == [
        ("positive", True),
        ("negative", False),
        ("vae", False),
    ]
    assert [(port.id, port.type) for port in schema.outputs] == [
        ("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
        ("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
        ("empty_latent", TypeExpr.concrete(LATENT_TYPE)),
    ]
    combo = schema.combos[0]
    assert combo.id == "resolution" and combo.default == "keep resolution"
    assert [option.key for option in combo.options] == ["keep resolution", "resize"]
    megapixels = combo.options[1].inputs[0]
    assert isinstance(megapixels, InputSpec)
    assert megapixels.id == "megapixels" and megapixels.default == 1.0
    assert isinstance(megapixels.widget, NumberWidget)
    assert megapixels.widget.min == 0.01
    assert megapixels.widget.max == 16.0
    family = schema.input_families[0]
    assert family.id == "images" and not family.required
    assert (family.min_members, family.max_members, family.member_prefix) == (1, 100, "image")
    assert isinstance(family.template[0], InputSpec)
    assert family.template[0].type == TypeExpr.concrete(IMAGE_TYPE)
