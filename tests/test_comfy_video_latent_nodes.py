"""Torch-free video latent node interfaces."""

from typing import cast

from dinkster_api.v1 import NumberWidget, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.video_latent import VIDEO_LATENT_NODES
from dinkster_nodes_comfy.video_latent_ops import VIDEO_LATENT_OPERATION_NODES


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


def test_video_latent_transform_schemas_match_the_fixed_reference_contract() -> None:
    trim, freeze, normalize = VIDEO_LATENT_OPERATION_NODES
    trim_schema, freeze_schema, normalize_schema = (
        trim.schema(),
        freeze.schema(),
        normalize.schema(),
    )

    assert (trim_schema.node_type, trim_schema.display_name, trim_schema.category) == (
        "TrimVideoLatent",
        "Trim Video Latent",
        "model/latent",
    )
    assert [input_spec.id for input_spec in trim_schema.inputs] == ["samples", "trim_amount"]
    trim_amount = cast(NumberWidget, trim_schema.inputs[1].widget)
    assert (trim_schema.inputs[1].default, trim_amount.min, trim_amount.max, trim_amount.step) == (
        0,
        0,
        99999,
        None,
    )
    assert [output.id for output in trim_schema.outputs] == ["LATENT"]

    assert (freeze_schema.node_type, freeze_schema.display_name, freeze_schema.category) == (
        "LTXVFreezeLatent",
        "LTXV Freeze Latent",
        "model/latent/ltxv",
    )
    assert [input_spec.id for input_spec in freeze_schema.inputs] == ["latent"]
    assert [output.id for output in freeze_schema.outputs] == ["latent"]

    assert (
        normalize_schema.node_type,
        normalize_schema.display_name,
        normalize_schema.category,
    ) == (
        "NormalizeVideoLatentStart",
        "NormalizeVideoLatentStart",
        "model/conditioning",
    )
    assert [input_spec.id for input_spec in normalize_schema.inputs] == [
        "latent",
        "start_frame_count",
        "reference_frame_count",
    ]
    for input_spec, default in zip(normalize_schema.inputs[1:], (4, 5), strict=True):
        widget = cast(NumberWidget, input_spec.widget)
        assert (input_spec.default, widget.min, widget.max, widget.step) == (default, 1, 16384, 1)
    assert [output.id for output in normalize_schema.outputs] == ["latent"]

    for node in VIDEO_LATENT_OPERATION_NODES:
        assert node in COMFY_NODES
        assert schema_to_wire(node.schema())
