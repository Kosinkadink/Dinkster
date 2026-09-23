"""Native execution of VIDEO alias graphs, including trusted template widgets.

WORKFLOW_TEMPLATES_ROOT enables the full tracked corpus replay. This tests the
backend alias contract after source-widget decoding, not the frontend importer.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from av.container import InputContainer
from dinkster_assets import AssetRef
from dinkster_assets.save_target import coerce_save_target
from dinkster_nodes_foundation.combo import StringToCombo
from dinkster_nodes_foundation.conversion import ValueConvert
from dinkster_nodes_foundation.primitives import FloatPrimitive
from dinkster_nodes_media_io import MEDIA_IO_NODES
from dinkster_nodes_media_io.video import SaveVideo
from dinkster_schema import schema_from_wire, schema_to_wire
from dinkster_schema.replace import MappingSource, ReplacementPredicate, rule_from_wire
from dinkster_video import assemble_video
from PIL import Image

from tests.test_media_comfy_aliases import _registry
from tests.test_video_io import _bound, _frames, _mount
from tools.gen_video_media_comfy_aliases import (
    VIDEO_CLASSES,
    _save_source_schema,
    merge_video_registry,
)

NODES = {
    node.schema().node_type: node
    for node in (*MEDIA_IO_NODES, ValueConvert, StringToCombo, FloatPrimitive)
}
RECORDS = {item["source"]["nodeClass"]: item for item in _registry()["records"]}


def test_video_registry_merge_is_independent_of_existing_video_positions() -> None:
    registry = _registry()
    video = {
        "records": [r for r in registry["records"] if r["source"]["nodeClass"] in VIDEO_CLASSES],
        "sourceSchemas": [
            s
            for s in registry["sourceSchemas"]
            if s["nodeType"].removeprefix("comfy.") in VIDEO_CLASSES
        ],
    }
    other = {key: [item for item in registry[key] if item not in video[key]] for key in video}
    expected = {**registry, **{key: [*other[key], *video[key]] for key in video}}
    existing = {
        **registry,
        **{key: [*reversed(video[key]), *other[key]] for key in video},
    }
    assert merge_video_registry({**registry, **other}, video) == expected
    assert merge_video_registry(existing, video) == expected
    assert merge_video_registry(existing, video) == expected


def test_save_video_source_schema_preserves_positional_branches() -> None:
    wire = next(s for s in _registry()["sourceSchemas"] if s["nodeType"] == "comfy.SaveVideo")
    interface = wire["interface"]
    assert [(entry["role"], entry["id"]) for entry in interface] == [
        ("input", "video"),
        ("input", "filename_prefix"),
        ("input", "encoding"),
        ("input", "encoding.crf"),
        ("input", "crf"),
        ("input", "metadata"),
        ("dynamicCombo", "format"),
        ("dynamicCombo", "codec"),
        ("output", "video"),
    ]
    for entry, kind in zip(
        interface[2:6], ("core.combo", "core.float", "core.float", "core.string"), strict=True
    ):
        assert entry == {
            "role": "input",
            "id": entry["id"],
            "type": {"kind": "concrete", "types": [kind]},
            "required": False,
            "forceInput": True,
            **(
                {"widget": {"type": "COMBO", "options": ["auto", "re-encode"]}}
                if entry["id"] == "encoding"
                else {}
            ),
        }
    assert [
        entry["id"] for entry in interface if "widget" in entry and not entry.get("forceInput")
    ] == ["filename_prefix"]
    format_combo, legacy_codec = interface[6:8]
    assert format_combo["required"] is True
    assert [option["key"] for option in format_combo["options"]] == ["auto", "mp4", "mkv", "webm"]
    assert legacy_codec["required"] is False

    def check_codec(combo, keys, *, required):
        assert combo["role"] == "dynamicCombo" and combo["id"] == "codec"
        assert combo["required"] is required
        assert combo.get("default", combo["options"][0]["key"]) == "auto"
        assert [option["key"] for option in combo["options"]] == keys
        assert combo["options"][0]["inputs"] == []
        for option in combo["options"][1:]:
            (encoding,) = option["inputs"]
            assert encoding["role"] == "dynamicCombo" and encoding["id"] == "encoding"
            assert encoding["required"] is False
            assert [mode["key"] for mode in encoding["options"]] == interface[2]["widget"][
                "options"
            ]
            assert encoding["options"][0]["inputs"] == []
            (crf,) = encoding["options"][1]["inputs"]
            default, maximum = (23.0, 51.0) if option["key"] == "h264" else (30.0, 63.0)
            assert crf["role"] == "input" and crf["id"] == "crf"
            assert crf["type"] == {"kind": "concrete", "types": ["core.float"]}
            assert crf["default"] == default
            assert crf["widget"] == {"type": "NUMBER", "min": 0.0, "max": maximum, "step": 1.0}

    for option in format_combo["options"]:
        (codec,) = option["inputs"]
        keys = ["auto", "av1"] if option["key"] == "webm" else ["auto", "h264", "av1"]
        check_codec(codec, keys, required=True)
    check_codec(legacy_codec, ["auto", "h264", "av1"], required=False)
    schema = schema_from_wire(wire)
    assert schema_to_wire(schema) == wire
    original = replace(schema, inputs=schema.inputs[:2])
    before = schema_to_wire(original)
    adapted = _save_source_schema(original)
    assert adapted == schema
    assert adapted.combos is original.combos
    assert schema_to_wire(original) == before


def _matches(predicate: ReplacementPredicate | None, values: dict, links: dict) -> bool:
    if predicate is None or predicate.kind == "always":
        return True
    if predicate.kind == "valueEquals":
        return values.get(predicate.input) == predicate.value
    if predicate.kind == "valuePresent":
        return predicate.input in values
    if predicate.kind == "inputConnected":
        return predicate.input in links
    if predicate.kind == "not":
        return not _matches(predicate.of[0], values, links)
    children = (_matches(child, values, links) for child in predicate.of)
    return all(children) if predicate.kind == "all" else any(children)


def _plan(name: str, values: dict, links: dict | None = None) -> tuple[Any, dict, dict]:
    links = links or {}
    rule = rule_from_wire(RECORDS[name]["replacement"])
    case = next(case for case in rule.cases if _matches(case.when, values, links))
    nodes = {"": case.to, **{key: node.type for key, node in case.nodes or ()}}
    inputs = {
        key: {
            spec.id: spec.default
            for spec in NODES[kind].schema().inputs
            if spec.default is not None
        }
        for key, kind in nodes.items()
    }
    for key, node in case.nodes or ():
        inputs[key].update(dict(node.values))

    def mapped(source: MappingSource) -> tuple[bool, object]:
        if source.kind == "constant":
            return True, source.value
        if source.kind in ("copy", "link") and source.input in links:
            return True, links[source.input]
        if source.kind == "link" or source.input not in values:
            return False, None
        value = values[source.input]
        if source.transform:
            assert source.transform.kind == "enumRename"
            value = dict(source.transform.map)[str(value)]
        return True, value

    for address, source in case.inputs:
        present, value = mapped(source)
        if present:
            node, port = address.split(":") if ":" in address else ("", address)
            inputs[node][port] = value
    for address, family in case.input_families:
        source_members = links.get(family.source_family, {})
        assert isinstance(source_members, dict)
        members = {}
        for suffix, source_inputs in source_members.items():
            assert isinstance(source_inputs, dict)
            mapped_inputs = {}
            for target_input, source in family.inputs:
                if source.kind == "copy" and source.input in source_inputs:
                    mapped_inputs[target_input] = source_inputs[source.input]
            assert set(mapped_inputs) == {"value"}
            members[suffix] = mapped_inputs["value"]
        node, port = address.split(":") if ":" in address else ("", address)
        inputs[node][port] = members
    return case, nodes, inputs


def _execute(name: str, values: dict, links: dict | None = None) -> tuple[dict, dict]:
    case, nodes, inputs = _plan(name, values, links)
    results = {}
    while len(results) != len(nodes):
        before = len(results)
        for key, kind in nodes.items():
            if key in results:
                continue
            incoming = []
            for link in case.links:
                target, port = link.to.split(":") if ":" in link.to else ("", link.to)
                if target == key:
                    producer, output = (
                        link.from_address.split(":")
                        if ":" in link.from_address
                        else ("", link.from_address)
                    )
                    incoming.append((producer, output, port))
            if any(producer not in results for producer, _, _ in incoming):
                continue
            for producer, output, port in incoming:
                inputs[key][port] = results[producer][output]
            if kind == "dinkster.set_save_target_prefix":
                inputs[key]["target"] = coerce_save_target(inputs[key]["target"])
            results[key] = NODES[kind].execute(**inputs[key])
        assert len(results) > before, "alias helper graph is not executable"
    outputs = {}
    for address, source in case.outputs:
        key, port = address.split(":") if ":" in address else ("", address)
        outputs[source] = results[key][port]
    return outputs, results


@pytest.fixture
def mounted_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, snapshot = _mount(tmp_path, monkeypatch)
    images = _frames(4)
    video = assemble_video(images, fps=4)
    saved = SaveVideo.execute(video=video, container="mp4", codec="h264")
    asset = saved["asset"]
    assert isinstance(asset, AssetRef)
    ref = _bound(asset, snapshot)
    return root, ref, images, video


@pytest.mark.parametrize(
    "name",
    [
        "LoadVideo",
        "ConcatenateVideo",
        "CreateVideo",
        "GetVideoComponents",
        "Video Slice",
        "VideoTrim",
        "VideoCrop",
        "SaveVideo",
        "SaveWEBM",
    ],
)
def test_video_alias_branches_execute(name: str, mounted_video) -> None:
    root, ref, images, video = mounted_video
    widget = {
        "trim": {"start_time": 0.25, "duration": 0.5},
        "crop": {"x": 2, "y": 2, "width": 32, "height": 32},
        "features": ["trim"],
    }
    values = {
        "filename_prefix": "alias",
        "format": "mp4",
        "codec": "h264",
        "fps": 4.0,
        "start_time": 0.25,
        "duration": 0.5,
        "strict_duration": False,
        "bit_depth": "10",
        "color_space": "HDR PQ",
        "trim": widget,
        "crop": widget,
    }
    if name == "SaveWEBM":
        values.update(codec="vp9", crf=32.75)
    links = {"video": video, "images": images, "file": ref}
    if name == "ConcatenateVideo":
        values.update(codec=["h264"])
        links["videos"] = {
            "video0": {"video": [video]},
            "video1": {"video": [video]},
        }
    outputs, results = _execute(name, values, links)
    if name in ("SaveVideo", "SaveWEBM"):
        asset = results[""]["asset"]
        assert results[""]["video"] is (
            video if name == "SaveVideo" else results["assemble"]["video"]
        )
        with av.open(root / asset.name, mode="r") as container:
            assert isinstance(container, InputContainer)
            assert len(list(container.decode(video=0))) == 4
        if name == "SaveWEBM":
            np.testing.assert_array_equal(next(iter(outputs.values())), images)
            assert results["crf"]["int"] == 32
    if name == "CreateVideo":
        assert results[""]["video"]["components"]["bit_depth"] == 10
        assert results[""]["video"]["components"]["color_space"] == "HDR PQ"
    if name == "ConcatenateVideo":
        assert len(results[""]["video"]["edits"][0]["concat"]) == 1
        assert results[""]["video"]["preferred_codec"] == "h264"
    if name == "GetVideoComponents":
        assert len(outputs) == 5
        np.testing.assert_array_equal(outputs["_0_IMAGE_"], images)
    if name in ("VideoTrim", "VideoCrop"):
        assert _plan(name, values, {"video": video})[2][""]["video_edit"] is widget
        assert widget["features"] == ["trim"]


@pytest.mark.parametrize("layout", ["flat", "codec", "format"])
@pytest.mark.parametrize("format", [None, "auto", "mp4", "mkv", "webm"])
@pytest.mark.parametrize("codec", ["auto", "h264", "av1"])
@pytest.mark.parametrize("encoding", ["auto", "re-encode"])
def test_save_video_all_source_branches(layout, format, codec, encoding, mounted_video):
    root, _ref, _images, video = mounted_video
    codec_path = "format.codec" if layout == "format" else "codec"
    encoding_path = codec_path + ".encoding" if layout != "flat" else "encoding"
    crf_path = encoding_path + ".crf" if layout != "flat" else "crf"
    values = {
        "filename_prefix": "branch",
        codec_path: codec,
        encoding_path: encoding,
        "metadata": '{"title":"alias conformance"}',
    }
    if format is not None:
        values["format"] = format
    if encoding == "re-encode" and codec != "auto":
        values[crf_path] = 31.75
    case, _nodes, planned = _plan("SaveVideo", values, {"video": video})
    assert dict(case.outputs) == {"video": "video"}
    assert (
        planned[""]["container"] == ("webm" if codec == "av1" else "mp4")
        if format in (None, "auto")
        else planned[""]["container"] == format
    )
    outputs, results = _execute("SaveVideo", values, {"video": video})
    assert outputs["video"] is video
    if crf_path in values:
        assert results["crf"]["int"] == 31
    if format == "webm" and codec == "h264":
        assert results[""]["asset"].name.endswith(".mkv")
    with av.open(root / results[""]["asset"].name, mode="r") as container:
        assert isinstance(container, InputContainer)
        expected_codec = codec if codec != "auto" else ("av1" if format == "webm" else "h264")
        assert container.streams.video[0].codec_context.codec.canonical_name == expected_codec
        assert len(list(container.decode(video=0))) == 4
        assert container.metadata["title"] == "alias conformance"


@pytest.mark.parametrize(
    "codec_values,expected_codec",
    [
        ({}, "vp9"),
        ({"codec": "auto"}, "vp9"),
        ({"codec": "h264"}, "h264"),
        ({"codec": "av1"}, "av1"),
        ({"format.codec": "auto"}, "vp9"),
        ({"format.codec": "h264"}, "h264"),
        ({"format.codec": "av1"}, "av1"),
        ({"format.codec": "auto", "codec": "av1"}, "vp9"),
        ({"format.codec": "av1", "codec": "h264"}, "av1"),
    ],
)
def test_save_video_omitted_format_uses_source_default(
    codec_values, expected_codec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, snapshot = _mount(tmp_path, monkeypatch)
    source = SaveVideo.execute(
        video=assemble_video(_frames(4), fps=4),
        container="webm",
        codec="vp9",
        target={"mount": "comfy-output", "prefix": "source"},
    )
    asset = source["asset"]
    assert isinstance(asset, AssetRef)
    loaded, _ = _execute("LoadVideo", {}, {"file": _bound(asset, snapshot)})
    video = next(iter(loaded.values()))
    native = SaveVideo.execute(video=video, target={"mount": "comfy-output", "prefix": "native"})
    native_asset = native["asset"]
    assert isinstance(native_asset, AssetRef)
    assert native_asset.name.endswith(".webm")
    assert (root / native_asset.name).read_bytes() == (root / asset.name).read_bytes()

    values = {"filename_prefix": "omitted-format", **codec_values}
    codec_path = "format.codec" if "format.codec" in codec_values else "codec"
    explicit = {**values, "format": "auto", codec_path: codec_values.get(codec_path, "auto")}
    assert _plan("SaveVideo", values)[2] == _plan("SaveVideo", explicit)[2]
    outputs, results = _execute("SaveVideo", values, {"video": video})
    assert outputs["video"] is video
    saved = results[""]["asset"]
    extension = "webm" if expected_codec == "av1" else "mp4"
    assert saved.name.endswith("." + extension)
    with av.open(root / saved.name, mode="r") as container:
        assert isinstance(container, InputContainer)
        expected_format = "matroska,webm" if extension == "webm" else "mov,mp4,m4a,3gp,3g2,mj2"
        assert container.format.name == expected_format
        assert container.streams.video[0].codec_context.codec.canonical_name == expected_codec
        assert len(list(container.decode(video=0))) == 4


@pytest.mark.parametrize("codec,depth", [("vp9", 8), ("av1", 10)])
@pytest.mark.parametrize("linked", [False, True])
def test_webm_encoding_and_passthrough(codec, depth, linked, mounted_video):
    root, _ref, images, _video = mounted_video
    outputs, results = _execute(
        "SaveWEBM",
        {
            "filename_prefix": "webm",
            "codec": codec,
            "fps": 4.0,
            **({} if linked else {"crf": 31.75}),
        },
        {"images": images, **({"crf": 31.75} if linked else {})},
    )
    assert results["crf_value"]["value"] == 31.75
    assert results["crf"]["int"] == 31
    np.testing.assert_array_equal(outputs["_0_IMAGE_"], images)
    assert results["assemble"]["video"]["components"]["bit_depth"] == depth
    with av.open(root / results[""]["asset"].name, mode="r") as container:
        assert isinstance(container, InputContainer)
        assert container.streams.video[0].codec_context.codec.canonical_name == codec
        pixel_format = container.streams.video[0].codec_context.format
        assert pixel_format is not None
        assert pixel_format.name == ("yuv420p10le" if depth == 10 else "yuv420p")
        assert len(list(container.decode(video=0))) == 4


def test_video_alias_helper_edges_and_combo_outputs_keep_types() -> None:
    sources = {
        schema.node_type: schema for schema in map(schema_from_wire, _registry()["sourceSchemas"])
    }
    for name in ("SaveVideo", "SaveWEBM", "GetVideoComponents"):
        record = RECORDS[name]
        source = sources[record["source"]["nodeType"]]
        for case in rule_from_wire(record["replacement"]).cases:
            schemas = {
                "": NODES[case.to].schema(),
                **{key: NODES[node.type].schema() for key, node in case.nodes or ()},
            }

            def port(address, role, schemas=schemas):
                key, name = address.split(":") if ":" in address else ("", address)
                specs = schemas[key].outputs if role == "output" else schemas[key].inputs
                return next(spec for spec in specs if spec.id == name)

            for link in case.links:
                output, input = port(link.from_address, "output"), port(link.to, "input")
                assert output.type.kind == "concrete"
                assert input.type.accepts_concrete(output.type.types[0])
            if name in ("SaveVideo", "SaveWEBM") and "crf" in schemas:
                assert dict(case.nodes or ())["crf_value"].type == "dinkster.float"
                assert "crf_value:value" in dict(case.inputs)
                assert "crf:value" not in dict(case.inputs)
                crf_input = next(spec for spec in schemas["crf_value"].inputs if spec.id == "value")
                assert crf_input.widget is not None and not crf_input.force_input
                assert {(link.from_address, link.to) for link in case.links} >= {
                    ("crf_value:value", "crf:value"),
                    ("crf:int", "crf"),
                }
                assert dict(dict(case.nodes or ())["crf"].values) == {
                    "target": "int",
                    "force_lossy": True,
                }
            for target, source_id in case.outputs:
                expected = next(spec.type for spec in source.outputs if spec.id == source_id)
                if expected.types == ("core.combo",):
                    assert port(target, "output").type == expected


@pytest.mark.parametrize("depth", ["auto", "8", "10"])
@pytest.mark.parametrize("color", ["sRGB", "HDR", "HDR PQ"])
def test_components_alias_depth_and_color_links(depth, color, mounted_video):
    _root, _ref, images, _video = mounted_video
    created, _ = _execute(
        "CreateVideo", {"bit_depth": depth, "color_space": color}, {"images": images}
    )
    video = created["_0_VIDEO_"]
    outputs, _ = _execute("GetVideoComponents", {}, {"video": video})
    expected = ("8" if color == "sRGB" else "10") if depth == "auto" else depth
    assert outputs["_3_COMBO_"] == expected
    assert outputs["_4_COMBO_"] == color
    assert outputs["_2_FLOAT_"] == 30.0
    roundtrip, _ = _execute(
        "CreateVideo",
        {},
        {
            "images": outputs["_0_IMAGE_"],
            "fps": outputs["_2_FLOAT_"],
            "bit_depth": outputs["_3_COMBO_"],
            "color_space": outputs["_4_COMBO_"],
        },
    )
    assert roundtrip["_0_VIDEO_"]["components"]["bit_depth"] == int(expected)


@pytest.mark.parametrize(
    "path", ["crf", "encoding.crf", "codec.encoding.crf", "format.codec.encoding.crf"]
)
@pytest.mark.parametrize("linked", [False, True])
def test_crf_link_uses_explicit_integer_conversion(path, linked, mounted_video):
    _root, _ref, _images, video = mounted_video
    codec_path = "format.codec" if path.startswith("format.") else "codec"
    _, results = _execute(
        "SaveVideo",
        {
            "format": "auto",
            codec_path: "h264",
            "filename_prefix": "crf",
            **({} if linked else {path: 31.75}),
        },
        {"video": video, **({path: 31.75} if linked else {})},
    )
    assert results["crf_value"]["value"] == 31.75
    assert results["crf"]["int"] == 31


@pytest.mark.parametrize("codec,crf", [("h264", 23), ("av1", 30)])
@pytest.mark.parametrize("format", [None, "auto"])
def test_reencode_default_crf_is_codec_specific(codec, crf, format, mounted_video):
    _root, _ref, _images, video = mounted_video
    values = {"format.codec": codec, "format.codec.encoding": "re-encode"}
    if format is not None:
        values["format"] = format
    case, _, inputs = _plan(
        "SaveVideo",
        values,
        {"video": video},
    )
    assert inputs[""]["crf"] == crf
    assert dict(case.inputs)["crf"].kind == "constant"


@pytest.mark.parametrize(
    "widget",
    [
        {},
        {"crop": {"x": 2, "y": 2, "width": 32, "height": 32}},
        {"trim": {"start_time": -0.5, "duration": 0}},
    ],
)
@pytest.mark.parametrize("name,port", [("VideoTrim", "trim"), ("VideoCrop", "crop")])
def test_video_edit_widgets_are_lossless(name, port, widget, mounted_video):
    _root, _ref, _images, video = mounted_video
    serialized = json.dumps(widget)
    _, results = _execute(name, {port: widget}, {"video": video})
    assert json.dumps(widget) == serialized
    assert len(results[""]["video"]["edits"]) == (1 if port in widget else 0)


def _template_values(name: str, widgets: list | dict | None) -> dict:
    """Simulate post-import values, including bit-depth strings; not importer evidence."""
    if isinstance(widgets, dict):
        return widgets
    widgets = widgets or []
    if name == "SaveVideo":
        assert 3 <= len(widgets) <= 4
        return dict(zip(("filename_prefix", "format", "codec", "encoding"), widgets, strict=False))
    if name == "CreateVideo":
        assert 1 <= len(widgets) <= 3
        values = dict(zip(("fps", "bit_depth", "color_space"), widgets, strict=False))
        if "bit_depth" in values:
            values["bit_depth"] = str(values["bit_depth"])
        return values
    if name == "Video Slice":
        assert len(widgets) == 3
        return dict(zip(("start_time", "duration", "strict_duration"), widgets, strict=True))
    assert name == "GetVideoComponents" and not widgets
    return {}


def test_tracked_template_video_shapes(mounted_video):
    configured = os.environ.get("WORKFLOW_TEMPLATES_ROOT")
    if not configured:
        pytest.skip("set WORKFLOW_TEMPLATES_ROOT to a trusted tracked corpus")
    corpus = Path(configured)
    sha = subprocess.check_output(
        ["git", "-C", str(corpus), "rev-parse", "HEAD"], text=True
    ).strip()
    paths = subprocess.check_output(
        ["git", "-C", str(corpus), "ls-files", "templates/*.json"], text=True
    ).splitlines()
    counts = Counter()
    counts_by_directory = defaultdict(Counter)
    shapes = Counter()
    root, _ref, images, video = mounted_video

    def visit(data, directory):
        if isinstance(data, dict):
            name = data.get("type")
            if name in ("SaveVideo", "CreateVideo", "GetVideoComponents", "Video Slice"):
                counts[name] += 1
                counts_by_directory[directory][name] += 1
                widgets = data.get("widgets_values")
                values = _template_values(name, widgets)
                # Source prefixes remain authored graph data, but corpus execution uses
                # a bounded portable target to avoid interpreting template path syntax.
                case, _nodes, inputs = _plan(name, values, {"video": video, "images": images})
                if name == "SaveVideo":
                    assert inputs["save_target"]["prefix"] == values["filename_prefix"]
                    values["filename_prefix"] = "template"
                shape = (name, json.dumps(values, sort_keys=True))
                shapes[shape] += 1
                if shapes[shape] == 1:
                    outputs, results = _execute(name, values, {"video": video, "images": images})
                    assert len(outputs) >= len(data.get("outputs", []))
                    if name == "SaveVideo":
                        with av.open(root / results[""]["asset"].name, mode="r") as container:
                            assert isinstance(container, InputContainer)
                            assert len(list(container.decode(video=0))) == 4
            for item in data.values():
                visit(item, directory)
        elif isinstance(data, list):
            for item in data:
                visit(item, directory)

    for path in paths:
        source = corpus / path
        if source.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            with Image.open(source) as image:
                visit(json.loads(image.info["workflow"]), Path(path).parts[0])
        else:
            visit(json.loads(source.read_text(encoding="utf-8")), Path(path).parts[0])
    templates = counts_by_directory["templates"]
    assert templates["SaveVideo"] >= 311
    assert templates["CreateVideo"] >= 167
    assert templates["GetVideoComponents"] >= 105
    assert templates["Video Slice"] >= 12
    print(
        json.dumps(
            {
                "corpus_sha": sha,
                "counts": counts,
                "by_directory": counts_by_directory,
                "executed_shapes": len(shapes),
            },
            sort_keys=True,
        )
    )
