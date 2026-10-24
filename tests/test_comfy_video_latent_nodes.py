"""Torch-free video latent node interfaces."""

from typing import cast

from dinkster_api.v1 import NumberWidget, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.video_latent import VIDEO_LATENT_NODES


def test_video_latent_schemas_match_the_fixed_reference_contract() -> None:
    expected = (
        (
            "EmptyHunyuanLatentVideo",
            "Empty Hunyuan Video 1.0 Latent",
            "model/latent/hunyuan video",
            (848, 480, 25, 1),
            (16, 16, 1, 1),
            (16384, 16384, 16384, 4096),
            (16, 16, 4, None),
        ),
        (
            "EmptyHunyuanVideo15Latent",
            "Empty Hunyuan Video 1.5 Latent",
            "model/latent/hunyuan video",
            (848, 480, 25, 1),
            (16, 16, 1, 1),
            (16384, 16384, 16384, 4096),
            (16, 16, 4, None),
        ),
        (
            "EmptyLTXVLatentVideo",
            "Empty LTXV Latent Video",
            "model/latent/ltxv",
            (768, 512, 97, 1),
            (64, 64, 1, 1),
            (16384, 16384, 16384, 4096),
            (32, 32, 8, None),
        ),
        (
            "EmptyCosmosLatentVideo",
            "Empty Cosmos Latent Video",
            "model/latent/cosmos",
            (1280, 704, 121, 1),
            (16, 16, 1, 1),
            (16384, 16384, 16384, 4096),
            (16, 16, 8, None),
        ),
        (
            "EmptyARVideoLatent",
            "",
            "model/latent/autoregressive",
            (832, 480, 81, 1),
            (16, 16, 1, 1),
            (8192, 8192, 1024, 64),
            (16, 16, 4, None),
        ),
    )
    actual = tuple(
        (
            node.schema().node_type,
            node.schema().display_name,
            node.schema().category,
            tuple(input_spec.default for input_spec in node.schema().inputs),
            tuple(cast(NumberWidget, input_spec.widget).min for input_spec in node.schema().inputs),
            tuple(cast(NumberWidget, input_spec.widget).max for input_spec in node.schema().inputs),
            tuple(
                cast(NumberWidget, input_spec.widget).step for input_spec in node.schema().inputs
            ),
        )
        for node in VIDEO_LATENT_NODES
    )
    assert actual == expected
    for node in VIDEO_LATENT_NODES:
        assert node in COMFY_NODES
        assert [output.id for output in node.schema().outputs] == ["LATENT"]
        assert schema_to_wire(node.schema())
