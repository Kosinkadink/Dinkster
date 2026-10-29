"""Torch-free discovery and schema contracts for Wan 2.2 image latents."""

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    NumberWidget,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.wan22_latent import Wan22ImageToVideoLatent
from dinkster_workers import load_manifest


def test_wan22_schema_and_cold_native_discovery() -> None:
    schema = Wan22ImageToVideoLatent.schema()
    assert schema.node_type == "Wan22ImageToVideoLatent"
    assert schema.category == "model/conditioning/wan"
    assert [i.id for i in schema.inputs] == [
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "start_image",
    ]
    assert [i.type for i in schema.inputs] == [
        TypeExpr.concrete(t) for t in (VAE_TYPE, CORE_INT, CORE_INT, CORE_INT, CORE_INT, IMAGE_TYPE)
    ]
    assert [i.required for i in schema.inputs] == [True] * 5 + [False]
    assert [i.default for i in schema.inputs[1:5]] == [1280, 704, 49, 1]
    assert [i.widget for i in schema.inputs[1:5]] == [
        NumberWidget(min=32, max=16384, step=32),
        NumberWidget(min=32, max=16384, step=32),
        NumberWidget(min=1, max=16384, step=4),
        NumberWidget(min=1, max=4096),
    ]
    assert [(o.id, o.type) for o in schema.outputs] == [("LATENT", TypeExpr.concrete(LATENT_TYPE))]
    assert schema_from_wire(schema_to_wire(schema)) == schema
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert ("devices", *(n.schema().node_type for n in COMFY_NODES)) == manifest.namespaces
    assert Wan22ImageToVideoLatent in COMFY_NODES
