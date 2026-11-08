"""Torch-free Hunyuan image-to-video schemas."""

import tomllib
from pathlib import Path

from dinkster_api.v1 import (
    CLIP_VISION_OUTPUT_TYPE,
    CONDITIONING_TYPE,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    ComboWidget,
    NumberWidget,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_hunyuan import NODES
from dinkster_hunyuan.hunyuan_video_image import (
    HunyuanImageToVideo,
    HunyuanVideo15ImageToVideo,
    HunyuanVideo15SuperResolution,
)
from dinkster_schema import claim_covers


def test_hunyuan_video_image_schemas_match_the_public_contract() -> None:
    manifest = tomllib.loads(
        (
            Path(__file__).resolve().parents[1]
            / "packages/dinkster-nodes-hunyuan/dinkster-pack.toml"
        ).read_text(encoding="ascii")
    )
    expected = (
        (
            "dinkster.hunyuan.hunyuan_image_to_video",
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
            "dinkster.hunyuan.hunyuan_video15_image_to_video",
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
        (
            "dinkster.hunyuan.hunyuan_video15_super_resolution",
            (
                ("positive", CONDITIONING_TYPE),
                ("negative", CONDITIONING_TYPE),
                ("vae", VAE_TYPE),
                ("start_image", IMAGE_TYPE),
                ("clip_vision_output", CLIP_VISION_OUTPUT_TYPE),
                ("latent", LATENT_TYPE),
                ("noise_augmentation", CORE_FLOAT),
            ),
            (
                ("positive", CONDITIONING_TYPE),
                ("negative", CONDITIONING_TYPE),
                ("latent", LATENT_TYPE),
            ),
            (None, None, None, None, None, None, 0.70),
            (True, True, False, False, False, True, True),
        ),
    )
    for node, (node_type, inputs, outputs, defaults, required) in zip(
        (HunyuanImageToVideo, HunyuanVideo15ImageToVideo, HunyuanVideo15SuperResolution),
        expected,
        strict=True,
    ):
        schema = node.schema()
        assert node in NODES
        assert schema.node_type == node_type
        assert schema.category == "model/conditioning/hunyuan video"
        assert any(
            claim_covers(namespace, node_type) for namespace in manifest["pack"]["namespaces"]
        )
        assert tuple((item.id, item.type.types[0]) for item in schema.inputs) == inputs
        assert tuple((item.id, item.type.types[0]) for item in schema.outputs) == outputs
        assert tuple(item.default for item in schema.inputs) == defaults
        assert tuple(item.required for item in schema.inputs) == required
        assert schema_from_wire(schema_to_wire(schema)) == schema
        for item, minimum, step in (
            zip(
                schema.inputs[3:7] if node is HunyuanVideo15ImageToVideo else schema.inputs[2:6],
                (16, 16, 1, 1),
                (16, 16, 4, None),
                strict=True,
            )
            if node is not HunyuanVideo15SuperResolution
            else ()
        ):
            assert isinstance(item.widget, NumberWidget)
            assert (item.widget.min, item.widget.max, item.widget.step) == (
                minimum,
                4096 if item.id == "batch_size" else 16384,
                step,
            )
    guidance = HunyuanImageToVideo.schema().inputs[6]
    assert guidance.advanced and isinstance(guidance.widget, ComboWidget)
    assert guidance.widget.options == ("v1 (concat)", "v2 (replace)", "custom")
