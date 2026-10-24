"""Post-processing nodes keep their source schemas and CPU behavior."""

from __future__ import annotations

from typing import Any, cast

import pytest
from dinkster_api.v1 import schema_to_wire
from dinkster_nodes_comfy.post_processing import (
    POST_PROCESSING_NODES,
    ColorTransfer,
)


def _wire(node: type[Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", schema_to_wire(node.schema()))


def test_post_processing_schemas_use_exact_core_ids_and_dynamic_interfaces() -> None:
    schemas = {node.schema().node_type: _wire(node) for node in POST_PROCESSING_NODES}
    assert tuple(schemas) == (
        "ResizeImageMaskNode",
        "BatchImagesNode",
        "BatchMasksNode",
        "BatchLatentsNode",
        "ColorTransfer",
    )
    resize = schemas["ResizeImageMaskNode"]
    assert [item["id"] for item in resize["interface"]] == [
        "input",
        "resize_type",
        "scale_method",
        "resized",
    ]
    assert resize["interface"][0]["type"] == {
        "kind": "variable",
        "templateId": "input_type",
        "allowed": ["IMAGE", "MASK"],
    }
    assert [option["key"] for option in resize["interface"][1]["options"]] == [
        "scale dimensions",
        "scale by multiplier",
        "scale longer dimension",
        "scale shorter dimension",
        "scale width",
        "scale height",
        "scale total pixels",
        "match size",
        "scale to multiple",
    ]
    for node_type, family, prefix, value_type in (
        ("BatchImagesNode", "images", "image", "IMAGE"),
        ("BatchMasksNode", "masks", "mask", "MASK"),
        ("BatchLatentsNode", "latents", "latent", "LATENT"),
    ):
        interface = schemas[node_type]["interface"]
        assert interface[0]["id"] == family
        assert interface[0]["memberPrefix"] == prefix
        assert interface[0]["minMembers"] == 1 and interface[0]["maxMembers"] == 50
        assert interface[0]["template"][0]["type"]["types"] == [value_type]
        assert interface[1]["type"]["types"] == [value_type]
    color = schemas["ColorTransfer"]
    assert [item["id"] for item in color["interface"]] == [
        "image_target",
        "image_ref",
        "method",
        "source_stats",
        "strength",
        "image",
    ]
    assert [option["key"] for option in color["interface"][3]["options"]] == [
        "per_frame",
        "uniform",
        "target_frame",
    ]


def test_color_transfer_identity_strength_does_not_materialize_the_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = object()
    monkeypatch.setattr(
        "dinkster_nodes_comfy.post_processing.materialize_image", lambda value: value
    )
    assert (
        ColorTransfer.execute(
            image_target=target,
            image_ref=None,
            method="histogram",
            source_stats={"source_stats": "per_frame"},
            strength=0.0,
        )["image"]
        is target
    )
