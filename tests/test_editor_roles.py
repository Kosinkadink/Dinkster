from __future__ import annotations

import dataclasses
from typing import Any, cast

import pytest
from dinkster_native.native import LoadLatent
from dinkster_nodes_foundation.conversion import CurveEditor
from dinkster_nodes_foundation.routing import RouteSwitchByName
from dinkster_nodes_image.compositor import CreateLayeredImage
from dinkster_nodes_image.glsl import GlslShader
from dinkster_nodes_image.layer_document import EditLayers, FlattenLayers
from dinkster_nodes_image.layer_io import LoadLayers
from dinkster_nodes_media_io.audio_ops import ExtractAudioEnvelope
from dinkster_nodes_media_io.image import LoadImage, PaintMask, SaveImage
from dinkster_nodes_media_io.video_ops import CropVideo, TrimVideo
from dinkster_schema import (
    InputSpec,
    Node,
    TypeExpr,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)
from dinkster_values import CustomWidgetDescriptor


@pytest.mark.parametrize(
    ("node", "role"),
    [
        (LoadLatent, "latent-source"),
        (CurveEditor, "curve"),
        (RouteSwitchByName, "named-route-switch"),
        (CreateLayeredImage, "compositor"),
        (GlslShader, "glsl"),
        (FlattenLayers, "layers-flatten"),
        (EditLayers, "layers-edit"),
        (LoadLayers, "layers-load"),
        (ExtractAudioEnvelope, "audio-envelope"),
        (LoadImage, "image-source"),
        (PaintMask, "mask-paint"),
        (SaveImage, "image-save"),
        (TrimVideo, "video-trim"),
        (CropVideo, "video-crop"),
    ],
)
def test_editor_capabilities_are_declared_by_schema(node: type[Node], role: str) -> None:
    assert node.schema().editor_role == role


@pytest.mark.parametrize(
    ("node", "features"),
    [(TrimVideo, ["trim"]), (CropVideo, ["crop"])],
)
def test_video_edit_features_are_declared_by_widget(node: type[Node], features: list[str]) -> None:
    video_edit = next(item for item in node.schema().inputs if item.id == "video_edit")
    assert video_edit.widget == CustomWidgetDescriptor(
        "VIDEO_EDIT", cast("dict[str, Any]", {"features": features})
    )


def test_editor_metadata_roundtrips_through_the_schema_wire() -> None:
    schema = dataclasses.replace(
        TrimVideo.schema(),
        inputs=(
            InputSpec(
                "custom",
                TypeExpr.concrete("extension.value"),
                widget=CustomWidgetDescriptor(
                    "extension.widget",
                    cast("dict[str, Any]", {"nested": {"enabled": True, "values": [1, None]}}),
                ),
            ),
        ),
    )
    wire = schema_to_wire(schema)
    assert wire["editorRole"] == "video-trim"
    assert cast("list[dict[str, object]]", wire["interface"])[0]["widget"] == {
        "type": "extension.widget",
        "nested": {"enabled": True, "values": [1, None]},
    }
    assert schema_from_wire(wire) == schema
    assert schema_signature(schema) == schema_signature(
        dataclasses.replace(
            schema,
            editor_role=None,
            inputs=(dataclasses.replace(schema.inputs[0], widget=None),),
        )
    )


@pytest.mark.parametrize(
    "widget_type",
    [
        "ASSET",
        "BOOLEAN",
        "COLOR",
        "COMBO",
        "COMPOSITOR",
        "CURVE",
        "MULTI_COMBO",
        "NUMBER",
        "REPRESENTATIONS",
        "SAVE_TARGET",
        "STRING",
    ],
)
def test_custom_widget_descriptors_cannot_impersonate_builtins(widget_type: str) -> None:
    with pytest.raises(ValueError, match="is reserved"):
        CustomWidgetDescriptor(widget_type)


def test_custom_widget_descriptor_freezes_nested_json() -> None:
    source = {"features": ["trim"], "options": {"snap": True}}
    descriptor = CustomWidgetDescriptor("VIDEO_EDIT", cast("dict[str, Any]", source))
    cast("list[str]", source["features"]).append("crop")
    cast("dict[str, bool]", source["options"])["snap"] = False
    assert descriptor.params == {"features": ("trim",), "options": {"snap": True}}
