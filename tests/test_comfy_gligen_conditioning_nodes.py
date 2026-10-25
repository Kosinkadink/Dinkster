"""Cold public interface for native GLIGEN textbox conditioning."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_INT,
    CORE_STRING,
    GLIGEN_TYPE,
    NumberWidget,
    StringWidget,
    TypeExpr,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.gligen_conditioning import GLIGENTextBoxApply


def test_gligen_textbox_schema_preserves_ports_defaults_and_widgets() -> None:
    schema = GLIGENTextBoxApply.schema()
    assert schema.node_type == "GLIGENTextBoxApply"
    assert schema.category == "model/conditioning/gligen"
    assert tuple(item.id for item in schema.inputs) == (
        "conditioning_to",
        "clip",
        "gligen_textbox_model",
        "text",
        "width",
        "height",
        "x",
        "y",
    )
    assert tuple(item.type for item in schema.inputs) == tuple(
        TypeExpr.concrete(name)
        for name in (CONDITIONING_TYPE, CLIP_TYPE, GLIGEN_TYPE, CORE_STRING, *([CORE_INT] * 4))
    )
    assert all(item.required for item in schema.inputs)
    assert schema.inputs[3].widget == StringWidget(multiline=True, dynamic_prompts=True)
    for item, default, minimum in zip(schema.inputs[4:], (64, 64, 0, 0), (8, 8, 0, 0), strict=True):
        assert item.default == default
        assert item.widget == NumberWidget(min=minimum, max=16384, step=8)
    assert [(item.id, item.type) for item in schema.outputs] == [
        ("CONDITIONING", TypeExpr.concrete(CONDITIONING_TYPE))
    ]


def test_gligen_textbox_is_registered_in_manifest_and_cold_serializes() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads(
        (root / "packages/dinkster-nodes-comfy/dinkster-pack.toml").read_text()
    )
    assert [node.schema().node_type for node in COMFY_NODES] == manifest["pack"]["namespaces"]
    assert GLIGENTextBoxApply in COMFY_NODES
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch'] = None; "
            "sys.modules['dinkster_inference'] = None; "
            "from dinkster_nodes_comfy.gligen_conditioning import GLIGENTextBoxApply; "
            "from dinkster_api.v1 import schema_to_wire; "
            "assert schema_to_wire(GLIGENTextBoxApply.schema())",
        ],
        check=True,
        cwd=root,
    )
