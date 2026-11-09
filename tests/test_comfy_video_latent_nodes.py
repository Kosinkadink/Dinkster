"""Torch-free video latent node interfaces."""

from typing import cast

from dinkster_api.v1 import NumberWidget, schema_to_wire
from dinkster_cosmos import NODES as COSMOS_NODES
from dinkster_cosmos.video_latent import EmptyCosmosLatentVideo
from dinkster_hunyuan import NODES as HUNYUAN_NODES
from dinkster_hunyuan.video_latent import EmptyHunyuanLatentVideo, EmptyHunyuanVideo15Latent
from dinkster_latent import NODES as LATENT_NODES
from dinkster_latent.video_latent_ops import NormalizeVideoLatentStart, TrimVideoLatent
from dinkster_ltx import NODES as LTX_NODES
from dinkster_ltx.video_latent import EmptyLTXVLatentVideo
from dinkster_ltx.video_latent_ops import LTXVFreezeLatent
from dinkster_misc_models import NODES as MISC_NODES
from dinkster_misc_models.video_latent import EmptyARVideoLatent


def test_video_latent_schemas_match_the_fixed_reference_contract() -> None:
    nodes = (
        (EmptyHunyuanLatentVideo, HUNYUAN_NODES),
        (EmptyHunyuanVideo15Latent, HUNYUAN_NODES),
        (EmptyLTXVLatentVideo, LTX_NODES),
        (EmptyCosmosLatentVideo, COSMOS_NODES),
        (EmptyARVideoLatent, MISC_NODES),
    )
    expected = (
        (
            "dinkster.hunyuan.empty_hunyuan_latent_video",
            "Empty Hunyuan Video 1.0 Latent",
            "model/latent/hunyuan video",
            (848, 480, 25, 1),
            (16, 16, 1, 1),
            (16384, 16384, 16384, 4096),
            (16, 16, 4, None),
        ),
        (
            "dinkster.hunyuan.empty_hunyuan_video15_latent",
            "Empty Hunyuan Video 1.5 Latent",
            "model/latent/hunyuan video",
            (848, 480, 25, 1),
            (16, 16, 1, 1),
            (16384, 16384, 16384, 4096),
            (16, 16, 4, None),
        ),
        (
            "dinkster.ltx.empty_ltxv_latent_video",
            "Empty LTXV Latent Video",
            "model/latent/ltxv",
            (768, 512, 97, 1),
            (64, 64, 1, 1),
            (16384, 16384, 16384, 4096),
            (32, 32, 8, None),
        ),
        (
            "dinkster.cosmos.empty_cosmos_latent_video",
            "Empty Cosmos Latent Video",
            "model/latent/cosmos",
            (1280, 704, 121, 1),
            (16, 16, 1, 1),
            (16384, 16384, 16384, 4096),
            (16, 16, 8, None),
        ),
        (
            "dinkster.misc.models.empty_ar_video_latent",
            "EmptyARVideoLatent",
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
        for node, _ in nodes
    )
    assert actual == expected
    for node, registered in nodes:
        assert node in registered
        assert [output.id for output in node.schema().outputs] == ["LATENT"]
        assert schema_to_wire(node.schema())


def test_video_latent_transform_schemas_match_the_fixed_reference_contract() -> None:
    trim, freeze, normalize = TrimVideoLatent, LTXVFreezeLatent, NormalizeVideoLatentStart
    trim_schema, freeze_schema, normalize_schema = (
        trim.schema(),
        freeze.schema(),
        normalize.schema(),
    )

    assert (trim_schema.node_type, trim_schema.display_name, trim_schema.category) == (
        "dinkster.latent.trim_video_latent",
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
        "dinkster.ltx.ltxv_freeze_latent",
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
        "dinkster.latent.normalize_video_latent_start",
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

    for node, registered in ((trim, LATENT_NODES), (freeze, LTX_NODES), (normalize, LATENT_NODES)):
        assert node in registered
        assert schema_to_wire(node.schema())
