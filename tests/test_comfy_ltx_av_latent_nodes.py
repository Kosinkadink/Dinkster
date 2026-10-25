"""Torch-free LTXV AV latent node schemas."""

import subprocess
import sys

from dinkster_api.v1 import LATENT_TYPE, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.ltx_av_latent import LTX_AV_LATENT_NODES


def test_ltx_av_latent_schemas_and_registration() -> None:
    concat, separate = (node.schema() for node in LTX_AV_LATENT_NODES)
    assert all(node in COMFY_NODES for node in LTX_AV_LATENT_NODES)
    assert [(schema.node_type, schema.category) for schema in (concat, separate)] == [
        ("LTXVConcatAVLatent", "model/latent/ltxv"),
        ("LTXVSeparateAVLatent", "model/latent/ltxv"),
    ]
    assert [input.id for input in concat.inputs] == ["video_latent", "audio_latent"]
    assert [output.id for output in concat.outputs] == ["LATENT"]
    assert [input.id for input in separate.inputs] == ["av_latent"]
    assert [output.id for output in separate.outputs] == ["VIDEO_LATENT", "AUDIO_LATENT"]
    for schema in (concat, separate):
        assert all(input.required and input.type.types == (LATENT_TYPE,) for input in schema.inputs)
        assert all(output.type.types == (LATENT_TYPE,) for output in schema.outputs)
        assert schema_from_wire(schema_to_wire(schema)) == schema


def test_ltx_av_latent_cold_discovery() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy.ltx_av_latent import LTX_AV_LATENT_NODES; "
            "[node.schema() for node in LTX_AV_LATENT_NODES]; "
            "assert 'torch' not in sys.modules; assert 'dinkster_inference' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
