"""Torch-free autoregressive start-image conditioning contract."""

import subprocess
import sys

from dinkster_api.v1 import (
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    VAE_TYPE,
    NumberWidget,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.ar_video import ARVideoI2V
from dinkster_workers import load_manifest


def test_ar_video_conditioning_schema_and_cold_discovery() -> None:
    schema = ARVideoI2V.schema()
    assert schema.node_type == "ARVideoI2V"
    assert schema.category == "model/conditioning/autoregressive"
    assert [(i.id, i.type) for i in schema.inputs[:3]] == [
        ("model", TypeExpr.concrete(MODEL_TYPE)),
        ("vae", TypeExpr.concrete(VAE_TYPE)),
        ("start_image", TypeExpr.concrete(IMAGE_TYPE)),
    ]
    assert [(i.id, i.type, i.default, i.widget) for i in schema.inputs[3:]] == [
        ("width", TypeExpr.concrete(CORE_INT), 832, NumberWidget(min=16, max=8192, step=16)),
        ("height", TypeExpr.concrete(CORE_INT), 480, NumberWidget(min=16, max=8192, step=16)),
        ("length", TypeExpr.concrete(CORE_INT), 81, NumberWidget(min=1, max=1024, step=4)),
        ("batch_size", TypeExpr.concrete(CORE_INT), 1, NumberWidget(min=1, max=64)),
    ]
    assert [(o.id, o.type) for o in schema.outputs] == [
        ("MODEL", TypeExpr.concrete(MODEL_TYPE)),
        ("LATENT", TypeExpr.concrete(LATENT_TYPE)),
    ]
    assert all(i.required for i in schema.inputs)
    assert schema_from_wire(schema_to_wire(schema)) == schema
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy import COMFY_NODES; "
            "assert any(n.schema().node_type == 'ARVideoI2V' for n in COMFY_NODES); "
            "assert 'torch' not in sys.modules; assert 'dinkster_inference' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_ar_video_conditioning_complete_registry_matches_manifest() -> None:
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert ARVideoI2V in COMFY_NODES
    assert tuple(node.schema().node_type for node in COMFY_NODES) == manifest.namespaces
