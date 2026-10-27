"""ControlNet public schemas and zero-strength behavior without Torch."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from dinkster_api.v1 import ComboOption, ComboWidget
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.controlnet import (
    CONTROLNET_NODES,
    ControlNetApply,
    ControlNetApplyAdvanced,
    ControlNetApplySD3,
    SetUnionControlNetType,
)


def test_controlnet_schemas_register_the_exact_core_ids() -> None:
    manifest = tomllib.loads(
        (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        ).read_text()
    )
    for node in CONTROLNET_NODES:
        assert node in COMFY_NODES
        schema = node.define_schema()
        assert schema.node_type == node.__name__
        assert schema.node_type in manifest["pack"]["namespaces"]
        assert schema.category == "model/conditioning/controlnet"
    legacy_sd3 = ControlNetApplySD3.define_schema()
    assert legacy_sd3.deprecation is not None
    assert legacy_sd3.search_visibility == "deprecated"
    schema = ControlNetApplyAdvanced.define_schema()
    assert [port.id for port in schema.inputs] == [
        "positive",
        "negative",
        "control_net",
        "image",
        "strength",
        "start_percent",
        "end_percent",
        "vae",
    ]
    assert not schema.inputs[-1].required
    assert [(port.id, port.default) for port in schema.inputs[4:7]] == [
        ("strength", 1.0),
        ("start_percent", 0.0),
        ("end_percent", 1.0),
    ]
    union = SetUnionControlNetType.define_schema().inputs[1].widget
    assert isinstance(union, ComboWidget)
    assert [
        option.value if isinstance(option, ComboOption) else option for option in union.options
    ] == [
        "auto",
        "openpose",
        "depth",
        "hed/pidi/scribble/ted",
        "canny/lineart/anime_lineart/mlsd",
        "normal",
        "segment",
        "tile",
        "repaint",
    ]


def test_native_manifest_schemas_are_available_without_inference_or_torch() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['dinkster_inference'] = None\n"
            "sys.modules['torch'] = None\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "schemas = [node.define_schema() for node in COMFY_NODES]\n"
            "assert any(schema.node_type == 'SetUnionControlNetType' for schema in schemas)\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_zero_strength_returns_sources_without_accessing_control_or_image() -> None:
    positive = [["positive", {"control": object(), "strength": 0.7}]]
    negative = [["negative", {"strength": 0.2}]]
    legacy = ControlNetApply.execute(
        conditioning=positive, control_net=object(), image=object(), strength=0.0
    )
    assert legacy["CONDITIONING"] is positive
    advanced = ControlNetApplyAdvanced.execute(
        positive=positive, negative=negative, control_net=object(), image=object(), strength=0.0
    )
    assert advanced["positive"] is positive
    assert advanced["negative"] is negative
