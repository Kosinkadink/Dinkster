"""Cosmos image-conditioning schemas are available without Torch."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dinkster_api.v1 import CORE_INT, IMAGE_TYPE, LATENT_TYPE, VAE_TYPE, NumberWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.cosmos_conditioning import (
    COSMOS_CONDITIONING_NODES,
    CosmosImageToVideoLatent,
    CosmosPredict2ImageToVideoLatent,
)


def test_cosmos_conditioning_schemas_match_the_public_contract() -> None:
    manifest = tomllib.loads(
        (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        ).read_text(encoding="ascii")
    )
    integer = TypeExpr.concrete(CORE_INT)
    image = TypeExpr.concrete(IMAGE_TYPE)
    latent = TypeExpr.concrete(LATENT_TYPE)
    vae = TypeExpr.concrete(VAE_TYPE)
    for node, defaults, length_step in (
        (CosmosImageToVideoLatent, (1280, 704, 121), 8),
        (CosmosPredict2ImageToVideoLatent, (848, 480, 93), 4),
    ):
        assert node in COSMOS_CONDITIONING_NODES
        assert node in COMFY_NODES
        schema = node.define_schema()
        assert schema.node_type == node.__name__
        assert schema.node_type in manifest["pack"]["namespaces"]
        assert schema.category == "model/conditioning/cosmos"
        assert [(input.id, input.type) for input in schema.inputs] == [
            ("vae", vae),
            ("width", integer),
            ("height", integer),
            ("length", integer),
            ("batch_size", integer),
            ("start_image", image),
            ("end_image", image),
        ]
        assert [input.required for input in schema.inputs] == [
            True,
            True,
            True,
            True,
            True,
            False,
            False,
        ]
        assert [input.default for input in schema.inputs] == [None, *defaults, 1, None, None]
        for input, minimum, maximum, step in zip(
            schema.inputs[1:5],
            (16, 16, 1, 1),
            (16384, 16384, 16384, 4096),
            (16, 16, length_step, None),
            strict=True,
        ):
            assert isinstance(input.widget, NumberWidget)
            assert (input.widget.min, input.widget.max, input.widget.step) == (
                minimum,
                maximum,
                step,
            )
        assert [(output.id, output.type, output.optional) for output in schema.outputs] == [
            ("LATENT", latent, False)
        ]
