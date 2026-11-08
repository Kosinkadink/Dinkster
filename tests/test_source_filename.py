from __future__ import annotations

import dataclasses
from typing import Any, cast

import pytest
from dinkster_api.v1 import VIDEO_TYPE
from dinkster_schema import (
    AssetWidget,
    InputSpec,
    NodeSchema,
    SourceFilenameSpec,
    TypeExpr,
    WidgetRepresentation,
    WidgetRepresentations,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)


def _schema(*, listed: bool = False) -> NodeSchema:
    asset = TypeExpr.concrete("ASSET")
    return NodeSchema(
        node_type="test.source-filename",
        inputs=(
            InputSpec(
                "source",
                TypeExpr.list_of(asset) if listed else asset,
                source_filename=SourceFilenameSpec("media/image", "input"),
                widget=AssetWidget(
                    accept=("image/png", "image/jpeg", "image/webp"),
                    kind="media/image",
                    allow_upload=True,
                ),
            ),
        ),
    )


def test_source_filename_model_is_frozen_strict_and_signature_significant() -> None:
    source = SourceFilenameSpec("media/image", "input")
    with pytest.raises(dataclasses.FrozenInstanceError):
        source.kind = "media/audio"  # type: ignore[misc]
    for kind in ("image", "model/checkpoint", "media/mesh", "media/animated"):
        with pytest.raises(ValueError, match="source filename kind"):
            SourceFilenameSpec(kind, "input")  # type: ignore[arg-type]
    for category in ("", "inputs", "preview", 1):
        with pytest.raises(ValueError, match="source filename category"):
            SourceFilenameSpec("media/image", category)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="allow_upload must be a bool"):
        AssetWidget(kind="media/image", allow_upload=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires an upload-enabled ASSET widget"):
        InputSpec(
            "source",
            TypeExpr.concrete("ASSET"),
            source_filename=source,
        )
    with pytest.raises(ValueError, match="requires a scalar asset or list of assets"):
        InputSpec(
            "source",
            TypeExpr.concrete("STRING"),
            source_filename=source,
            widget=AssetWidget(kind="media/image", allow_upload=True),
        )
    schema = _schema()
    without_binding = dataclasses.replace(
        schema,
        inputs=(
            dataclasses.replace(
                schema.inputs[0],
                source_filename=None,
                widget=AssetWidget(kind="media/image"),
            ),
        ),
    )
    assert schema_signature(schema) != schema_signature(without_binding)
    presentation_only = dataclasses.replace(
        schema,
        inputs=(
            dataclasses.replace(
                schema.inputs[0],
                widget=AssetWidget(kind="media/image", allow_upload=False),
                source_filename=None,
            ),
        ),
    )
    assert schema_signature(presentation_only) == schema_signature(without_binding)


def _model3d_schema() -> NodeSchema:
    return NodeSchema(
        node_type="test.source-filename-model3d",
        inputs=(
            InputSpec(
                "model_file",
                TypeExpr.asset_of(TypeExpr.concrete("MODEL3D")),
                source_filename=SourceFilenameSpec("media/model3d", "input"),
                widget=AssetWidget(
                    accept=("model/gltf-binary",),
                    kind="media/model3d",
                    allow_upload=True,
                ),
            ),
        ),
        outputs=(),
    )


def test_model3d_source_filename_roundtrip() -> None:
    schema = _model3d_schema()
    wire = schema_to_wire(schema)
    assert wire["schemaVersion"] == 1
    entry = cast("list[dict[str, Any]]", wire["interface"])[0]
    assert entry["sourceFilename"] == {"kind": "media/model3d", "category": "input"}
    assert cast("dict[str, Any]", entry["widget"])["kind"] == "media/model3d"
    assert schema_from_wire(wire) == schema


def test_signature_basis_covers_model3d_and_keeps_prior_signatures() -> None:
    # Pin the signature for the canonical ASSET vocabulary; model3d support
    # must not change schemas that do not use it.
    assert schema_signature(_schema()) == "d4518a7b1ae2855b2ea80088a615af6ecac536a0"
    model3d = schema_signature(_model3d_schema())
    assert model3d != schema_signature(_schema())
    image_bound = dataclasses.replace(
        _model3d_schema(),
        inputs=(
            dataclasses.replace(
                _model3d_schema().inputs[0],
                source_filename=SourceFilenameSpec("media/image", "input"),
                widget=AssetWidget(
                    accept=("model/gltf-binary",),
                    kind="media/image",
                    allow_upload=True,
                ),
            ),
        ),
    )
    assert model3d != schema_signature(image_bound)


def test_list_asset_widget_representations_cannot_bypass_source_binding() -> None:
    asset_list = TypeExpr.list_of(TypeExpr.concrete("ASSET"))

    def representations(*, allow_upload: bool) -> WidgetRepresentations:
        return WidgetRepresentations(
            representations=(
                WidgetRepresentation(
                    "asset",
                    AssetWidget(kind="media/image", allow_upload=allow_upload),
                ),
            ),
            default="asset",
            user_switchable=True,
        )

    with pytest.raises(ValueError, match="asset widget requires a concrete"):
        InputSpec("source", asset_list, widget=representations(allow_upload=False))
    with pytest.raises(ValueError, match="requires a source filename binding"):
        InputSpec("source", asset_list, widget=representations(allow_upload=True))

    malformed = schema_to_wire(_schema(listed=True))
    entry = cast("list[dict[str, Any]]", malformed["interface"])[0]
    entry.pop("sourceFilename")
    entry["widget"] = {
        "type": "REPRESENTATIONS",
        "default": "asset",
        "userSwitchable": True,
        "representations": [
            {
                "id": "asset",
                "widget": {
                    "type": "ASSET",
                    "accept": [],
                    "kind": "media/image",
                    "allowUpload": True,
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="requires a source filename binding"):
        schema_from_wire(malformed)


def test_typed_asset_upload_widget_requires_source_binding() -> None:
    # A typed asset input (asset<T>) with an upload-enabled ASSET widget must
    # declare source_filename like any other upload input: the frontend
    # decoder rejects the schema otherwise, silently dropping the node from
    # the catalog (dinkster.load_video shipped this way).
    typed = TypeExpr.asset_of(TypeExpr.concrete(VIDEO_TYPE))
    widget = AssetWidget(accept=("video/mp4",), kind="media/video", allow_upload=True)
    with pytest.raises(ValueError, match="requires a source filename binding"):
        InputSpec("video", typed, widget=widget)
    InputSpec(
        "video",
        typed,
        widget=widget,
        source_filename=SourceFilenameSpec("media/video", "input"),
    )


def test_source_filename_golden_is_encoder_authored() -> None:
    assert schema_to_wire(_schema(listed=True)) == {
        "schemaVersion": 1,
        "nodeType": "test.source-filename",
        "version": 1,
        "displayName": "",
        "category": "",
        "description": "",
        "idempotent": True,
        "interface": [
            {
                "role": "input",
                "id": "source",
                "type": {
                    "kind": "list",
                    "element": {"kind": "concrete", "types": ["ASSET"]},
                },
                "required": True,
                "widget": {
                    "type": "ASSET",
                    "accept": ["image/png", "image/jpeg", "image/webp"],
                    "kind": "media/image",
                    "allowUpload": True,
                },
                "sourceFilename": {"kind": "media/image", "category": "input"},
            }
        ],
    }
