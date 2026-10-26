"""Torch-free native Wan image-conditioning interface contracts."""

import subprocess
import sys

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
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
from dinkster_workers import load_manifest


def test_wan_image_schemas_and_cold_native_discovery() -> None:
    optional_names = {
        "WanImageToVideo": ["clip_vision_output", "start_image", "ref_pad_image"],
        "WanFirstLastFrameToVideo": [
            "clip_vision_start_image",
            "clip_vision_end_image",
            "start_image",
            "end_image",
        ],
        "WanFunInpaintToVideo": ["clip_vision_output", "start_image", "end_image"],
    }
    for name, optional in optional_names.items():
        schema = next(n.schema() for n in COMFY_NODES if n.schema().node_type == name)
        assert schema.category == (
            "model/conditioning/wan/fun inpaint"
            if name == "WanFunInpaintToVideo"
            else "model/conditioning/wan"
        )
        assert [i.id for i in schema.inputs] == [
            "positive",
            "negative",
            "vae",
            "width",
            "height",
            "length",
            "batch_size",
            *optional,
        ]
        assert [i.type for i in schema.inputs[:7]] == [
            TypeExpr.concrete(t)
            for t in (
                CONDITIONING_TYPE,
                CONDITIONING_TYPE,
                VAE_TYPE,
                CORE_INT,
                CORE_INT,
                CORE_INT,
                CORE_INT,
            )
        ]
        assert [i.default for i in schema.inputs[3:7]] == [832, 480, 81, 1]
        assert [i.widget for i in schema.inputs[3:7]] == [
            NumberWidget(min=16, max=16384, step=16),
            NumberWidget(min=16, max=16384, step=16),
            NumberWidget(min=1, max=16384, step=4),
            NumberWidget(min=1, max=4096),
        ]
        assert all(i.required for i in schema.inputs[:7])
        assert all(not i.required for i in schema.inputs[7:])
        assert [i.type for i in schema.inputs[7:]] == [
            TypeExpr.concrete(CLIP_VISION_OUTPUT_TYPE if n.startswith("clip_") else IMAGE_TYPE)
            for n in optional
        ]
        assert [o.id for o in schema.outputs] == ["positive", "negative", "latent"]
        assert [o.type for o in schema.outputs] == [
            TypeExpr.concrete(t) for t in (CONDITIONING_TYPE, CONDITIONING_TYPE, LATENT_TYPE)
        ]
        assert schema_from_wire(schema_to_wire(schema)) == schema
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert tuple(n.schema().node_type for n in COMFY_NODES) == manifest.namespaces
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy import COMFY_NODES; "
            "assert {'WanImageToVideo','WanFirstLastFrameToVideo','WanFunInpaintToVideo'} "
            "<= {n.schema().node_type for n in COMFY_NODES}; "
            "assert 'torch' not in sys.modules; assert 'dinkster_inference' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
