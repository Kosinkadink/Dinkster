"""Torch-free LTXV AV latent node schemas."""

from dinkster_api.v1 import LATENT_TYPE, schema_from_wire, schema_to_wire
from dinkster_ltx import NODES
from dinkster_ltx.ltx_av_latent import LTXVConcatAVLatent, LTXVSeparateAVLatent


def test_ltx_av_latent_schemas_and_registration() -> None:
    nodes = (LTXVConcatAVLatent, LTXVSeparateAVLatent)
    concat, separate = (node.schema() for node in nodes)
    assert all(NODES.count(node) == 1 for node in nodes)
    assert [(schema.node_type, schema.category) for schema in (concat, separate)] == [
        ("dinkster.ltx.ltxv_concat_av_latent", "model/latent/ltxv"),
        ("dinkster.ltx.ltxv_separate_av_latent", "model/latent/ltxv"),
    ]
    assert [input.id for input in concat.inputs] == ["video_latent", "audio_latent"]
    assert [output.id for output in concat.outputs] == ["LATENT"]
    assert [input.id for input in separate.inputs] == ["av_latent"]
    assert [output.id for output in separate.outputs] == ["VIDEO_LATENT", "AUDIO_LATENT"]
    for schema in (concat, separate):
        assert all(input.required and input.type.types == (LATENT_TYPE,) for input in schema.inputs)
        assert all(output.type.types == (LATENT_TYPE,) for output in schema.outputs)
        assert schema_from_wire(schema_to_wire(schema)) == schema
