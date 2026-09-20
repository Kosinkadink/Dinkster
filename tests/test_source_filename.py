from __future__ import annotations

import dataclasses
from typing import Any, cast

import pytest
from dinkster_assets import AssetRef, digest_bytes
from dinkster_compat_comfy import CompatError, CompatTranslation
from dinkster_compat_comfy.translate import translate_node
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
from dinkster_workers import ExecutionContext
from dinkster_workers.execution import use_execution_context


def _schema(*, listed: bool = False) -> NodeSchema:
    asset = TypeExpr.concrete("dinkster.asset")
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
            TypeExpr.concrete("dinkster.asset"),
            source_filename=source,
        )
    with pytest.raises(ValueError, match="requires a scalar asset or list of assets"):
        InputSpec(
            "source",
            TypeExpr.concrete("core.string"),
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
                TypeExpr.asset_of(TypeExpr.concrete("dinkster.model3d")),
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
    # This pre-model3d signature must never move; a change here means cache
    # keys drifted.
    assert schema_signature(_schema()) == "a5f4d7ad940047fe84496c165b176c88b40dead2"
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
    asset_list = TypeExpr.list_of(TypeExpr.concrete("dinkster.asset"))

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
    typed = TypeExpr.asset_of(TypeExpr.concrete("comfy.VIDEO"))
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
                    "element": {"kind": "concrete", "types": ["dinkster.asset"]},
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


def _source_node(config: dict[str, object], *, input_is_list: bool = False) -> type:
    class SourceNode:
        RETURN_TYPES = ()
        FUNCTION = "execute"
        seen: list[object] = []

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"source": (["existing"], config)}}

        def execute(self, source):  # noqa: ANN001, ANN201
            type(self).seen.append(source)
            return ()

    if input_is_list:
        SourceNode.INPUT_IS_LIST = True  # type: ignore[attr-defined]
    return SourceNode


def _ref(name: str) -> AssetRef:
    return AssetRef(digest_bytes(name.encode()), name, len(name))


def test_v1_source_filename_translation_and_execution_use_only_materializer() -> None:
    translation = CompatTranslation()
    source = _source_node({"video_upload": True, "image_folder": "temp", "forceInput": True})
    node = translate_node("SourceVideo", source, translation)
    (spec,) = node.schema().inputs
    assert spec.type == TypeExpr.concrete("dinkster.asset")
    assert spec.source_filename == SourceFilenameSpec("media/video", "temp")
    assert spec.force_input is True
    assert spec.widget == AssetWidget(
        accept=("video/mp4", "video/webm"),
        kind="media/video",
        allow_upload=True,
    )

    asset = _ref("clip.webm")
    calls: list[tuple[AssetRef, str, str]] = []

    def materialize(asset: AssetRef, kind: str, category: str) -> str:
        calls.append((asset, kind, category))
        return "staged/clip.webm [temp]"

    with use_execution_context(ExecutionContext(None, None, materialize_source=materialize)):
        assert node.execute(source=asset) == {}
    assert calls == [(asset, "media/video", "temp")]
    assert source.seen == ["staged/clip.webm [temp]"]

    with pytest.raises(CompatError, match="requires an AssetRef"):
        with use_execution_context(ExecutionContext(None, None, materialize_source=materialize)):
            node.execute(source="ambient/path.webm")
    with pytest.raises(CompatError, match="invocation-scoped source staging"):
        node.execute(source=asset)


def test_v1_declared_source_list_preserves_order_and_duplicates() -> None:
    source = _source_node({"audio_upload": True, "multiselect": True})
    node = translate_node("SourceAudioList", source, CompatTranslation())
    (spec,) = node.schema().inputs
    assert spec.type == TypeExpr.list_of(TypeExpr.concrete("dinkster.asset"))
    first = _ref("one.wav")
    second = _ref("two.wav")
    calls: list[AssetRef] = []

    def materialize(asset: AssetRef, kind: str, category: str) -> str:
        assert (kind, category) == ("media/audio", "input")
        calls.append(asset)
        return asset.name

    with use_execution_context(ExecutionContext(None, None, materialize_source=materialize)):
        assert node.execute(source=[first, second, first]) == {}
    assert calls == [first, second, first]
    assert source.seen == [["one.wav", "two.wav", "one.wav"]]
    with pytest.raises(CompatError, match="declared asset list"):
        with use_execution_context(ExecutionContext(None, None, materialize_source=materialize)):
            node.execute(source=(first, second))
    calls.clear()
    with pytest.raises(CompatError, match="only AssetRef"):
        with use_execution_context(ExecutionContext(None, None, materialize_source=materialize)):
            node.execute(source=[first, "ambient/two.wav"])
    assert calls == []


@pytest.mark.parametrize(
    ("config", "input_is_list", "message"),
    (
        ({"image_upload": "yes"}, False, "must be a Boolean"),
        ({"image_upload": True, "audio_upload": True}, False, "contradictory"),
        ({"file_upload": True}, False, "file_upload/model"),
        ({"animated_upload": True}, False, "animated_upload"),
        ({"mesh_upload": True}, False, "mesh_upload"),
        ({"image_upload": True, "animated": True}, False, "animated"),
        ({"image_upload": True, "multiple": True}, False, "multiple"),
        ({"image_upload": True, "other_upload": False}, False, "unsupported source upload"),
        ({"image_folder": "output"}, False, "no source upload"),
        ({"image_upload": True, "image_folder": "preview"}, False, "image_folder"),
        ({"image_upload": True, "default": "ambient.png"}, False, "ambient default"),
        ({"image_upload": True, "lazy": True}, False, "lazy"),
        ({"image_upload": True, "rawLink": True}, False, "rawLink"),
        ({"image_upload": True, "remote": {}}, False, "remote route"),
        (
            {"image_upload": True, "remote": {"route": "relative"}},
            False,
            "remote route",
        ),
        (
            {"image_upload": True, "remote": {"route": "/files", "other": True}},
            False,
            "unsupported fields",
        ),
        (
            {"image_upload": True, "control_after_generate": "increment"},
            False,
            "control_after_generate",
        ),
        ({"image_upload": True}, True, "INPUT_IS_LIST"),
    ),
)
def test_v1_source_filename_refuses_malformed_unsupported_and_batch_mismatch(
    config: dict[str, object],
    input_is_list: bool,
    message: str,
) -> None:
    with pytest.raises(CompatError, match=message):
        translate_node(
            "RefusedSource",
            _source_node(config, input_is_list=input_is_list),
            CompatTranslation(),
        )


def test_v1_source_filename_accepts_bounded_remote_refresh_metadata() -> None:
    source = _source_node(
        {
            "image_upload": True,
            "image_folder": "output",
            "remote": {
                "route": "/internal/files/output",
                "refresh_button": True,
                "control_after_refresh": "first",
            },
        }
    )
    (spec,) = translate_node("RemoteImage", source, CompatTranslation()).schema().inputs
    assert spec.source_filename == SourceFilenameSpec("media/image", "output")


def test_legacy_combo_without_source_declaration_stays_dormant() -> None:
    source = _source_node({})
    node = translate_node("LegacyCombo", source, CompatTranslation())
    (spec,) = node.schema().inputs
    assert spec.type == TypeExpr.concrete("core.combo")
    assert spec.source_filename is None
    assert node.execute(source="existing") == {}
    assert source.seen == ["existing"]
