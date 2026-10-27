"""Torch-free public interfaces for native attention guidance nodes."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from dinkster_api.v1 import CORE_FLOAT, CORE_STRING, MODEL_TYPE, TypeExpr, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.attention_guidance import ATTENTION_GUIDANCE_NODES


def test_attention_guidance_schemas_are_registered_with_fixed_core_interfaces() -> None:
    manifest = tomllib.loads(
        (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        ).read_text(encoding="ascii")
    )
    expected = (
        ("SelfAttentionGuidance", ("model", "scale", "blur_sigma")),
        ("PerturbedAttentionGuidance", ("model", "scale")),
        (
            "SkipLayerGuidanceDiT",
            (
                "model",
                "double_layers",
                "single_layers",
                "scale",
                "start_percent",
                "end_percent",
                "rescaling_scale",
            ),
        ),
        ("SkipLayerGuidanceSD3", ("model", "layers", "scale", "start_percent", "end_percent")),
        (
            "SkipLayerGuidanceDiTSimple",
            ("model", "double_layers", "single_layers", "start_percent", "end_percent"),
        ),
    )
    assert (
        tuple(
            (node.schema().node_type, tuple(port.id for port in node.schema().inputs))
            for node in ATTENTION_GUIDANCE_NODES
        )
        == expected
    )
    for node in ATTENTION_GUIDANCE_NODES:
        schema = node.schema()
        assert node in COMFY_NODES
        assert schema.node_type in manifest["pack"]["namespaces"]
        assert schema.inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert schema.outputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert schema_to_wire(schema)
    assert ATTENTION_GUIDANCE_NODES[0].schema().inputs[1].type == TypeExpr.concrete(CORE_FLOAT)
    assert ATTENTION_GUIDANCE_NODES[2].schema().inputs[1].type == TypeExpr.concrete(CORE_STRING)


def test_attention_guidance_schema_registration_does_not_import_torch_or_inference() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['dinkster_inference'] = None\n"
            "sys.modules['torch'] = None\n"
            "from dinkster_nodes_comfy.attention_guidance import ATTENTION_GUIDANCE_NODES\n"
            "schemas = [node.define_schema() for node in ATTENTION_GUIDANCE_NODES]\n"
            "assert [schema.node_type for schema in schemas] == [\n"
            "    'SelfAttentionGuidance', 'PerturbedAttentionGuidance',\n"
            "    'SkipLayerGuidanceDiT', 'SkipLayerGuidanceSD3', 'SkipLayerGuidanceDiTSimple'\n"
            "]\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
