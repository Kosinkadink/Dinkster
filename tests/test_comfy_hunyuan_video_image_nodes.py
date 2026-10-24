"""Torch-free Hunyuan image-to-video schemas."""

import subprocess
import sys
import tomllib
from pathlib import Path

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    ComboWidget,
    NumberWidget,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.hunyuan_video_image import HUNYUAN_VIDEO_IMAGE_NODES


def test_hunyuan_video_image_schemas_match_the_public_contract() -> None:
    manifest = tomllib.loads(
        (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        ).read_text(encoding="ascii")
    )
    expected = (
        (
            "HunyuanImageToVideo",
            (
                ("positive", CONDITIONING_TYPE),
                ("vae", VAE_TYPE),
                ("width", CORE_INT),
                ("height", CORE_INT),
                ("length", CORE_INT),
                ("batch_size", CORE_INT),
                ("guidance_type", CORE_COMBO),
                ("start_image", IMAGE_TYPE),
            ),
            (("positive", CONDITIONING_TYPE), ("latent", LATENT_TYPE)),
            (None, None, 848, 480, 53, 1, "v1 (concat)", None),
            (True, True, True, True, True, True, True, False),
        ),
        (
            "HunyuanVideo15ImageToVideo",
            (
                ("positive", CONDITIONING_TYPE),
                ("negative", CONDITIONING_TYPE),
                ("vae", VAE_TYPE),
                ("width", CORE_INT),
                ("height", CORE_INT),
                ("length", CORE_INT),
                ("batch_size", CORE_INT),
                ("start_image", IMAGE_TYPE),
                ("clip_vision_output", CLIP_VISION_OUTPUT_TYPE),
            ),
            (
                ("positive", CONDITIONING_TYPE),
                ("negative", CONDITIONING_TYPE),
                ("latent", LATENT_TYPE),
            ),
            (None, None, None, 848, 480, 33, 1, None, None),
            (True, True, True, True, True, True, True, False, False),
        ),
    )
    for node, (node_type, inputs, outputs, defaults, required) in zip(
        HUNYUAN_VIDEO_IMAGE_NODES, expected, strict=True
    ):
        schema = node.schema()
        assert node in COMFY_NODES
        assert schema.node_type == node_type
        assert schema.category == "model/conditioning/hunyuan video"
        assert schema.node_type in manifest["pack"]["namespaces"]
        assert tuple((item.id, item.type.types[0]) for item in schema.inputs) == inputs
        assert tuple((item.id, item.type.types[0]) for item in schema.outputs) == outputs
        assert tuple(item.default for item in schema.inputs) == defaults
        assert tuple(item.required for item in schema.inputs) == required
        assert schema_from_wire(schema_to_wire(schema)) == schema
        for item, minimum, step in zip(
            schema.inputs[3:7] if node_type == "HunyuanVideo15ImageToVideo" else schema.inputs[2:6],
            (16, 16, 1, 1),
            (16, 16, 4, None),
            strict=True,
        ):
            assert isinstance(item.widget, NumberWidget)
            assert (item.widget.min, item.widget.max, item.widget.step) == (
                minimum,
                4096 if item.id == "batch_size" else 16384,
                step,
            )
    guidance = HUNYUAN_VIDEO_IMAGE_NODES[0].schema().inputs[6]
    assert guidance.advanced and isinstance(guidance.widget, ComboWidget)
    assert guidance.widget.options == ("v1 (concat)", "v2 (replace)", "custom")


def test_hunyuan_video_image_cold_discovery() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['dinkster_inference'] = None\n"
            "sys.modules['torch'] = None\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "ids = {node.schema().node_type for node in COMFY_NODES}\n"
            "assert {'HunyuanImageToVideo', 'HunyuanVideo15ImageToVideo'} <= ids\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
