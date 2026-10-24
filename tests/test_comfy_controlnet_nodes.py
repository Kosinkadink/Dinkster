"""ControlNet public schemas and zero-strength behavior without Torch."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.controlnet import (
    CONTROLNET_NODES,
    ControlNetApply,
    ControlNetApplyAdvanced,
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
