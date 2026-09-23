"""The media I/O pack's maintained ComfyUI alias registry."""

from __future__ import annotations

import json
import math
import tomllib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from dinkster_nodes_foundation import FOUNDATION_NODES
from dinkster_nodes_foundation.combo import StringToCombo
from dinkster_nodes_foundation.conversion import ValueConvert
from dinkster_nodes_foundation.primitives import FloatPrimitive
from dinkster_nodes_media_io import MEDIA_IO_NODES
from dinkster_schema import (
    comfy_alias_registry_from_wire,
    schema_from_wire,
    schema_to_wire,
    validate_replacement_references,
)
from dinkster_schema.replace import rule_from_wire, rule_to_wire

ROOT = Path(__file__).parent.parent
ALIAS_PATH = ROOT / "packages" / "dinkster-nodes-media-io" / "comfy-aliases.json"

AUDIO_ALIAS_CLASSES = {
    "LoadAudio": "dinkster.load_audio",
    "EmptyAudio": "dinkster.empty_audio",
    "SaveAudio": "dinkster.save_audio",
    "SaveAudioMP3": "dinkster.save_audio_mp3",
    "SaveAudioOpus": "dinkster.save_audio_opus",
    "PreviewAudio": "dinkster.preview_audio",
    "SaveAudioAdvanced": "dinkster.save_audio",
    "VHS_LoadAudio": "dinkster.load_audio",
    "VHS_LoadAudioUpload": "dinkster.load_audio",
}
TEXT_ALIAS_CLASSES = {
    "SaveText": "dinkster.save_text",
}
CAPTURE_ALIAS_CLASSES = {
    "RecordAudio": "dinkster.record_audio",
    "WebcamCapture": "dinkster.webcam_capture",
}
AUDIO_OPS_ALIAS_CLASSES = {
    "TrimAudioDuration": "dinkster.audio.trim",
    "AudioCrop": "dinkster.audio.trim",
    "SplitAudioChannels": "dinkster.audio.split_channels",
    "JoinAudioChannels": "dinkster.audio.join_channels",
    "AudioConcat": "dinkster.audio.concat",
    "AudioMerge": "dinkster.audio.merge",
    "AudioAdjustVolume": "dinkster.audio.volume",
    "AudioEqualizer3Band": "dinkster.audio.equalizer",
}
VIDEO_OPS_ALIAS_CLASSES = {
    "ConcatenateVideo": "dinkster.video.concatenate",
    "CreateVideo": "dinkster.video.assemble",
    "GetVideoComponents": "dinkster.video.disassemble",
    "VideoTrim": "dinkster.video.trim",
    "VideoCrop": "dinkster.video.crop",
    "SaveWEBM": "dinkster.save_video",
}
CURRENT_SEEDVR2_ALIAS_CLASSES = {
    "LoadVideo": "dinkster.load_video_value",
    "SaveVideo": "dinkster.save_video",
    "Video Slice": "dinkster.video.trim",
}
IMAGE_ALIAS_CLASSES = {
    "LoadImage": "dinkster.load_image",
    "LoadImageMask": "dinkster.load_mask",
    "LoadImageOutput": "dinkster.load_image_output",
    "SaveImage": "dinkster.save_image",
    "PreviewImage": "dinkster.preview_image",
    "SaveAnimatedPNG": "dinkster.save_animated_image",
    "SaveAnimatedWEBP": "dinkster.save_animated_image",
    "SaveImageAdvanced": "dinkster.save_avif",
}
MODEL3D_ALIAS_CLASSES = {
    "Preview3DAdvanced": "dinkster.preview_model3d",
    "Save3DAdvanced": "dinkster.save_model3d",
}
MEDIA_ALIAS_CLASSES = {
    **AUDIO_ALIAS_CLASSES,
    **TEXT_ALIAS_CLASSES,
    **CAPTURE_ALIAS_CLASSES,
    **AUDIO_OPS_ALIAS_CLASSES,
    **VIDEO_OPS_ALIAS_CLASSES,
    **IMAGE_ALIAS_CLASSES,
    **CURRENT_SEEDVR2_ALIAS_CLASSES,
    **MODEL3D_ALIAS_CLASSES,
}


def _registry() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(ALIAS_PATH.read_text(encoding="utf-8")))


def test_media_comfy_aliases_use_the_canonical_wire_contract() -> None:
    registry = _registry()
    assert registry["format"] == "dinkster-comfy-alias/1"
    assert set(registry) == {"format", "sourceSchemas", "records"}
    comfy_alias_registry_from_wire(registry)

    source_schemas = [schema_from_wire(wire) for wire in registry["sourceSchemas"]]
    assert len(source_schemas) == len({schema.node_type for schema in source_schemas})
    assert all(not schema.replacements for schema in source_schemas)
    assert [schema_to_wire(schema) for schema in source_schemas] == registry["sourceSchemas"]

    records = registry["records"]
    assert {record["source"]["nodeClass"]: record["carrier"] for record in records} == (
        MEDIA_ALIAS_CLASSES
    )
    source_types = {schema.node_type for schema in source_schemas}
    assert {record["source"]["nodeType"] for record in records} == source_types
    for record in records:
        assert record["mappingKind"] == "op"
        assert record["source"]["nodeType"] == record["replacement"]["from"]
        rule = rule_from_wire(record["replacement"])
        assert rule_to_wire(rule) == record["replacement"]
        assert record["carrier"] in {case.to for case in rule.cases}


def test_media_comfy_alias_rules_reference_valid_schema_surface() -> None:
    registry = _registry()
    carriers = set(MEDIA_ALIAS_CLASSES.values())
    native_schemas = {
        node.schema().node_type: node.schema() for node in (*MEDIA_IO_NODES, *FOUNDATION_NODES)
    }
    records_by_carrier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in registry["records"]:
        records_by_carrier[record["carrier"]].append(record)
    assert set(records_by_carrier) == carriers

    source_schemas = [schema_from_wire(wire) for wire in registry["sourceSchemas"]]
    schemas = {schema.node_type: schema for schema in source_schemas}
    for carrier, schema in native_schemas.items():
        rules = tuple(
            rule_from_wire(record["replacement"]) for record in records_by_carrier[carrier]
        )
        schemas[carrier] = replace(schema, replacements=rules)
    for helper in (ValueConvert, StringToCombo, FloatPrimitive):
        schemas[helper.schema().node_type] = helper.schema()
    assert validate_replacement_references(schemas) == ()


def test_trellis2_3d_output_aliases_preserve_models() -> None:
    registry = _registry()
    records = {record["source"]["nodeClass"]: record for record in registry["records"]}
    source_schemas = {schema["nodeType"]: schema for schema in registry["sourceSchemas"]}

    preview = records["Preview3DAdvanced"]
    assert preview["source"]["revision"] == "8a33128f"
    preview_case = preview["replacement"]["cases"][0]
    assert preview_case["inputs"] == {"model": {"kind": "copy", "input": "model_3d"}}
    assert preview_case["outputs"] == {"model": "_0_FILE_3D_"}

    save = records["Save3DAdvanced"]
    assert save["source"]["revision"] == "8a33128f"
    save_case = save["replacement"]["cases"][0]
    assert save_case["inputs"] == {
        "model": {"kind": "copy", "input": "model_3d"},
        "save_target:prefix": {"kind": "copy", "input": "filename_prefix"},
    }
    assert save_case["outputs"] == {"model": "_0_FILE_3D_"}

    for record in (preview, save):
        source = source_schemas[record["source"]["nodeType"]]
        inputs = {item["id"]: item for item in source["interface"] if item["role"] == "input"}
        assert inputs["model_3d"]["type"] == {
            "kind": "concrete",
            "types": ["dinkster.model3d"],
        }
        assert inputs["viewport_state"]["widget"]["multiline"] is False
        assert source["interface"][-1]["type"] == {
            "kind": "concrete",
            "types": ["dinkster.model3d"],
        }

    save_source = source_schemas[save["source"]["nodeType"]]
    save_inputs = {item["id"]: item for item in save_source["interface"] if item["role"] == "input"}
    assert save_inputs["filename_prefix"]["widget"]["multiline"] is False


def test_media_comfy_aliases_preserve_pinned_source_semantics() -> None:
    registry = _registry()
    source_schemas = {schema["nodeType"]: schema for schema in registry["sourceSchemas"]}

    load = source_schemas["comfy.LoadAudio"]
    load_input = next(item for item in load["interface"] if item["id"] == "audio")
    assert load_input["widget"]["allowUpload"] is True
    assert load.get("outputNode") is None

    for node_type, quality in [
        ("comfy.SaveAudio", None),
        ("comfy.SaveAudioMP3", ("V0", ["V0", "128k", "320k"])),
        ("comfy.SaveAudioOpus", ("128k", ["64k", "96k", "128k", "192k", "320k"])),
    ]:
        schema = source_schemas[node_type]
        assert schema["outputNode"] is True
        assert "deprecation" in schema
        interface = {item["id"]: item for item in schema["interface"]}
        # Hidden prompt/extra_pnginfo inputs are excluded from translation.
        assert "prompt" not in interface and "extra_pnginfo" not in interface
        assert interface["filename_prefix"]["default"] == "audio/ComfyUI"
        if quality is not None:
            default, options = quality
            assert interface["quality"]["default"] == default
            assert interface["quality"]["widget"]["options"] == options

    empty = {item["id"]: item for item in source_schemas["comfy.EmptyAudio"]["interface"]}
    assert empty["duration"]["default"] == 60.0
    assert empty["sample_rate"]["default"] == 44100
    assert empty["channels"]["default"] == 2

    preview = source_schemas["comfy.PreviewAudio"]
    assert preview["outputNode"] is True
    assert "deprecation" not in preview

    save_text = source_schemas["comfy.SaveText"]
    assert save_text["outputNode"] is True
    assert "deprecation" not in save_text
    text_interface = {item["id"]: item for item in save_text["interface"]}
    assert text_interface["text"]["forceInput"] is True
    assert text_interface["filename_prefix"]["default"] == "ComfyUI"
    assert text_interface["format"]["default"] == "txt"
    assert text_interface["format"]["widget"]["options"] == ["txt", "csv", "md", "json"]


def test_media_comfy_aliases_preserve_pinned_capture_semantics() -> None:
    registry = _registry()
    source_schemas = {schema["nodeType"]: schema for schema in registry["sourceSchemas"]}
    records = {record["source"]["nodeClass"]: record for record in registry["records"]}

    # The pinned RecordAudio's sole input is the browser-recording handle:
    # an opaque custom socket with no widget. It cannot carry over to a
    # server-side recording, so the case maps no inputs and declares the gap.
    record_audio = source_schemas["comfy.RecordAudio"]
    inputs = {item["id"]: item for item in record_audio["interface"] if item["role"] == "input"}
    outputs = {item["id"]: item for item in record_audio["interface"] if item["role"] == "output"}
    assert set(inputs) == {"audio"}
    assert inputs["audio"]["type"] == {"kind": "concrete", "types": ["comfy.AUDIO_RECORD"]}
    assert "widget" not in inputs["audio"]
    assert set(outputs) == {"_0_AUDIO_"}
    assert outputs["_0_AUDIO_"]["type"] == {"kind": "concrete", "types": ["comfy.AUDIO"]}
    record = records["RecordAudio"]
    assert record["confidence"]["tier"] == "parametric"
    case = record["replacement"]["cases"][0]
    assert "inputs" not in case
    assert case["outputs"] == {"audio": "_0_AUDIO_"}
    note = record["replacement"]["note"]
    assert "browser" in note and "server-side" in note

    # The pinned WebcamCapture captures client-side and its server execution
    # ignores width/height/capture_on_queue; the native node makes width and
    # height binding, so only those two inputs carry over.
    webcam = source_schemas["comfy.WebcamCapture"]
    assert webcam["idempotent"] is False
    inputs = {item["id"]: item for item in webcam["interface"] if item["role"] == "input"}
    outputs = {item["id"]: item for item in webcam["interface"] if item["role"] == "output"}
    assert set(inputs) == {"image", "width", "height", "capture_on_queue"}
    assert inputs["image"]["type"] == {"kind": "concrete", "types": ["comfy.WEBCAM"]}
    assert "widget" not in inputs["image"]
    for name in ("width", "height"):
        assert inputs[name]["default"] == 0
        assert inputs[name]["widget"]["max"] == 16384
    assert inputs["capture_on_queue"]["default"] is True
    assert set(outputs) == {"image"}
    assert outputs["image"]["type"] == {"kind": "concrete", "types": ["comfy.IMAGE"]}
    record = records["WebcamCapture"]
    assert record["confidence"]["tier"] == "parametric"
    case = record["replacement"]["cases"][0]
    assert case["inputs"] == {
        "width": {"kind": "copy", "input": "width"},
        "height": {"kind": "copy", "input": "height"},
    }
    assert case["outputs"] == {"image": "image"}
    assert "capture_on_queue" in record["replacement"]["note"]


def test_media_comfy_aliases_preserve_pinned_video_ops_semantics() -> None:
    registry = _registry()
    source_schemas = {schema["nodeType"]: schema for schema in registry["sourceSchemas"]}
    records = {record["source"]["nodeClass"]: record for record in registry["records"]}

    create = source_schemas["comfy.CreateVideo"]
    inputs = {item["id"]: item for item in create["interface"] if item["role"] == "input"}
    assert set(inputs) == {"images", "fps", "audio", "bit_depth", "color_space", "codec"}
    assert inputs["fps"]["default"] == 30.0
    assert inputs["bit_depth"]["default"] == "auto"
    assert inputs["bit_depth"]["widget"]["options"] == ["auto", "8", "10"]
    assert inputs["color_space"]["default"] == "sRGB"
    record = records["CreateVideo"]
    assert record["confidence"]["tier"] == "parametric"
    case = record["replacement"]["cases"][0]
    assert case["inputs"] == {
        "images": {"kind": "copy", "input": "images"},
        "fps": {"kind": "copy", "input": "fps"},
        "audio": {"kind": "copy", "input": "audio"},
        "bit_depth": {"kind": "copy", "input": "bit_depth"},
        "color_space": {"kind": "copy", "input": "color_space"},
        "codec": {"kind": "copy", "input": "codec"},
    }
    assert set(case["outputs"]) == {"video"}
    assert "bit_depth" in record["replacement"]["note"]
    assert "color_space" in record["replacement"]["note"]

    concatenate = source_schemas["comfy.ConcatenateVideo"]
    family = next(item for item in concatenate["interface"] if item["role"] == "inputFamily")
    assert family["id"] == "videos"
    assert family["memberPrefix"] == "video"
    assert family["minMembers"] == 1
    assert family["maxMembers"] == 100
    record = records["ConcatenateVideo"]
    case = record["replacement"]["cases"][0]
    assert case["inputs"] == {
        "codec": {"kind": "copy", "input": "codec"},
        "complete_audio": {"kind": "copy", "input": "complete_audio"},
    }
    assert case["inputFamilies"] == {
        "videos": {
            "kind": "copy",
            "sourceFamily": "videos",
            "inputs": {"value": {"kind": "copy", "input": "video"}},
        }
    }
    assert case["outputs"] == {"video": "_0_VIDEO_"}

    components = source_schemas["comfy.GetVideoComponents"]
    outputs = {item["id"]: item for item in components["interface"] if item["role"] == "output"}
    assert len(outputs) == 5
    record = records["GetVideoComponents"]
    assert record["confidence"]["tier"] == "parametric"
    case = record["replacement"]["cases"][0]
    assert case["inputs"] == {"video": {"kind": "copy", "input": "video"}}
    assert set(case["outputs"]) == {"images", "audio", "fps", "depth:choice", "color:choice"}
    assert "bit_depth" in record["replacement"]["note"]
    assert "color_space" in record["replacement"]["note"]


def test_current_seedvr2_media_aliases_preserve_supported_paths() -> None:
    registry = _registry()
    source_schemas = {schema["nodeType"]: schema for schema in registry["sourceSchemas"]}
    records = {record["source"]["nodeClass"]: record for record in registry["records"]}

    load = records["LoadVideo"]
    assert load["source"]["revision"] == "95539f56"
    assert load["confidence"]["tier"] == "exact"
    assert load["replacement"]["cases"][0] == {
        "to": "dinkster.load_video_value",
        "inputs": {"video": {"kind": "copy", "input": "file"}},
        "outputs": {"video": "_0_VIDEO_"},
    }
    load_input = next(
        item for item in source_schemas["comfy.LoadVideo"]["interface"] if item["id"] == "file"
    )
    assert load_input["widget"]["accept"] == ["video/mp4", "video/webm"]

    save = records["SaveVideo"]
    assert save["source"]["revision"] == "95539f56"
    assert save["confidence"]["tier"] == "parametric"
    for save_case in save["replacement"]["cases"]:
        assert save_case["to"] == "dinkster.save_video"
        assert save_case["inputs"]["video"] == {"kind": "copy", "input": "video"}
        assert save_case["outputs"] == {"video": "video"}
        assert save_case["inputs"]["save_target:prefix"] == {
            "kind": "copy",
            "input": "filename_prefix",
        }

    trim = records["Video Slice"]
    assert trim["source"]["revision"] == "95539f56"
    assert trim["confidence"]["tier"] == "parametric"
    trim_case = trim["replacement"]["cases"][0]
    assert trim_case["inputs"] == {
        name: {"kind": "copy", "input": name}
        for name in ("video", "start_time", "duration", "strict_duration")
    }
    assert trim_case["outputs"] == {"video": "_0_VIDEO_"}
    assert "intersect" in trim["replacement"]["note"]

    assert "ImageCompare" not in records

    advanced = records["SaveImageAdvanced"]
    assert advanced["source"]["revision"] == "95539f56"
    assert advanced["confidence"]["tier"] == "parametric"
    advanced_schema = source_schemas["comfy.SaveImageAdvanced"]
    advanced_inputs = {
        item["id"]: item for item in advanced_schema["interface"] if item["role"] == "input"
    }
    assert advanced_inputs["bit_depth"]["default"] == "auto"
    assert advanced_inputs["bit_depth"]["widget"]["options"] == [
        "auto",
        "8-bit YUV420",
        "10-bit YUV420",
    ]
    assert advanced_inputs["input_color_space"]["widget"]["options"] == [
        "sRGB",
        "HDR",
        "HDR PQ",
    ]
    still, animated, refusal = advanced["replacement"]["cases"]
    assert still["when"]["of"] == [
        {"kind": "valueEquals", "input": "format", "value": "avif"},
        {"kind": "valueEquals", "input": "save_mode", "value": "still images"},
    ]
    assert still["inputs"]["animated"] == {"kind": "constant", "value": False}
    assert animated["when"]["of"][1]["value"] == "animated"
    assert animated["inputs"]["animated"] == {"kind": "constant", "value": True}
    assert animated["inputs"]["fps"] == {"kind": "copy", "input": "fps"}
    assert animated["inputs"]["loop"] == {"kind": "copy", "input": "loop_count"}
    assert refusal["inputs"]["images"]["transform"] == {"kind": "enumRename", "map": {}}


def test_media_comfy_aliases_preserve_pinned_audio_ops_semantics() -> None:
    registry = _registry()
    source_schemas = {schema["nodeType"]: schema for schema in registry["sourceSchemas"]}
    records = {record["source"]["nodeClass"]: record for record in registry["records"]}

    # Every pinned audio op accepts a None audio passthrough that the native
    # required-input nodes drop; the shared gap is declared on every note.
    for name in AUDIO_OPS_ALIAS_CLASSES.keys() - {"AudioCrop"}:
        assert "passthrough" in records[name]["replacement"]["note"]

    trim = {item["id"]: item for item in source_schemas["comfy.TrimAudioDuration"]["interface"]}
    assert trim["start_index"]["default"] == 0.0
    assert trim["duration"]["default"] == 60.0
    assert trim["duration"]["widget"]["min"] == 0.0
    record = records["TrimAudioDuration"]
    assert record["confidence"]["tier"] == "exact"
    case = record["replacement"]["cases"][0]
    assert case["inputs"] == {
        "audio": {"kind": "copy", "input": "audio"},
        "start": {"kind": "copy", "input": "start_index"},
        "duration": {"kind": "copy", "input": "duration"},
    }
    assert case["outputs"] == {"audio": "_0_AUDIO_"}

    split_outputs = [
        item["id"]
        for item in source_schemas["comfy.SplitAudioChannels"]["interface"]
        if item["role"] == "output"
    ]
    assert split_outputs == ["_0_AUDIO_", "_1_AUDIO_"]
    record = records["SplitAudioChannels"]
    assert record["confidence"]["tier"] == "exact"
    case = record["replacement"]["cases"][0]
    assert case["inputs"] == {"audio": {"kind": "copy", "input": "audio"}}
    assert case["outputs"] == {"left": "_0_AUDIO_", "right": "_1_AUDIO_"}

    # Mixed-rate inputs go through libswresample rather than torchaudio's
    # sinc resampler, so join/concat/merge are parametric with the gap noted.
    for name in ("JoinAudioChannels", "AudioConcat", "AudioMerge"):
        record = records[name]
        assert record["confidence"]["tier"] == "parametric"
        assert "libswresample" in record["replacement"]["note"]

    concat = {item["id"]: item for item in source_schemas["comfy.AudioConcat"]["interface"]}
    assert concat["direction"]["default"] == "after"
    assert concat["direction"]["widget"]["options"] == ["after", "before"]

    merge = {item["id"]: item for item in source_schemas["comfy.AudioMerge"]["interface"]}
    assert merge["merge_method"]["default"] == "add"
    assert merge["merge_method"]["widget"]["options"] == ["add", "mean", "subtract", "multiply"]

    # The pinned integer decibel volume widens to the native float gain_db.
    volume = {item["id"]: item for item in source_schemas["comfy.AudioAdjustVolume"]["interface"]}
    assert volume["volume"]["type"] == {"kind": "concrete", "types": ["core.int"]}
    assert volume["volume"]["default"] == 1
    assert volume["volume"]["widget"]["min"] == -100
    assert volume["volume"]["widget"]["max"] == 100
    record = records["AudioAdjustVolume"]
    assert record["confidence"]["tier"] == "exact"
    case = record["replacement"]["cases"][0]
    assert case["inputs"] == {
        "audio": {"kind": "copy", "input": "audio"},
        "gain_db": {"kind": "copy", "input": "volume"},
    }
    assert "gain_db" in record["replacement"]["note"]

    # The native equalizer is a float64 port of the pinned float32 torchaudio
    # biquad chain, so it is equivalent with a declared waveform tolerance.
    equalizer = {
        item["id"]: item for item in source_schemas["comfy.AudioEqualizer3Band"]["interface"]
    }
    for name, default in [("low_gain_dB", 0.0), ("mid_gain_dB", 0.0), ("high_gain_dB", 0.0)]:
        assert equalizer[name]["default"] == default
        assert equalizer[name]["widget"]["min"] == -24.0
        assert equalizer[name]["widget"]["max"] == 24.0
    assert equalizer["low_freq"]["default"] == 100
    assert equalizer["mid_freq"]["default"] == 1000
    assert equalizer["mid_q"]["default"] == 0.707
    assert equalizer["high_freq"]["default"] == 5000
    record = records["AudioEqualizer3Band"]
    assert record["confidence"]["tier"] == "equivalent"
    assert record["confidence"]["tolerances"] == [
        {"metric": "waveform_max_abs_diff", "operator": "<=", "value": 0.04}
    ]
    case = record["replacement"]["cases"][0]
    assert case["inputs"] == {
        "audio": {"kind": "copy", "input": "audio"},
        "low_gain_db": {"kind": "copy", "input": "low_gain_dB"},
        "low_freq": {"kind": "copy", "input": "low_freq"},
        "mid_gain_db": {"kind": "copy", "input": "mid_gain_dB"},
        "mid_freq": {"kind": "copy", "input": "mid_freq"},
        "mid_q": {"kind": "copy", "input": "mid_q"},
        "high_gain_db": {"kind": "copy", "input": "high_gain_dB"},
        "high_freq": {"kind": "copy", "input": "high_freq"},
    }
    assert "torchaudio" in record["replacement"]["note"]

    # FadeAudio is native-only: the pinned pack has no fade node, so no
    # alias record exists for it.
    assert "FadeAudio" not in records


def test_media_comfy_aliases_declare_save_target_and_passthrough_gaps() -> None:
    records = {record["source"]["nodeClass"]: record for record in _registry()["records"]}

    for name in ("SaveAudio", "SaveAudioMP3", "SaveAudioOpus", "SaveText"):
        record = records[name]
        case = record["replacement"]["cases"][0]
        # filename_prefix has no mapping (the mounted target keeps its declared
        # default), the source passthrough output is not preserved, and the
        # pinned savers' embedded metadata (audio) or inline display (text) is
        # not replicated; every gap is declared on the rule note for reviewers.
        assert "target" not in case["inputs"]
        assert "outputs" not in case
        assert "filename_prefix" in record["replacement"]["note"]
        assert record["confidence"]["tier"] in {"parametric", "equivalent"}

    for name in ("SaveAudio", "SaveAudioMP3", "SaveAudioOpus"):
        assert "prompt/extra_pnginfo" in records[name]["replacement"]["note"]

    save_text_note = records["SaveText"]["replacement"]["note"]
    assert "dinkster.preview_any" in save_text_note
    assert "newline" in save_text_note
    assert "64 MiB" in save_text_note

    opus = records["SaveAudioOpus"]["confidence"]
    assert opus["tier"] == "equivalent"
    assert opus["tolerances"] == [
        {"metric": "decoded_waveform_max_abs_diff", "operator": "<=", "value": 0.05}
    ]

    assert records["LoadAudio"]["confidence"]["tier"] == "exact"
    load_case = records["LoadAudio"]["replacement"]["cases"][0]
    assert load_case["inputs"] == {"audio": {"kind": "copy", "input": "audio"}}
    assert load_case["outputs"] == {"audio": "_0_AUDIO_"}

    assert records["PreviewAudio"]["confidence"]["tier"] == "exact"
    assert records["PreviewAudio"]["replacement"]["cases"][0]["outputs"] == {"audio": "audio"}

    empty = records["EmptyAudio"]
    assert empty["confidence"]["tier"] == "parametric"
    assert "86400" in empty["replacement"]["note"]


def test_media_comfy_alias_confidence_has_pinned_evidence() -> None:
    for record in _registry()["records"]:
        assert (
            record["source"]["revision"]
            in {
                "comfy-core": {"b78cec87", "8a33128f", "15eb748b", "95539f56"},
                "comfyui-videohelpersuite": {"4d907bee61e92c2e65af3bd6383a4e4d356126d1"},
                "audio-separation-nodes-comfyui": {"ac339561973f0c1e56db2f9d40f11b0fddda6763"},
            }[record["source"]["pack"]]
        )
        confidence = record["confidence"]
        assert confidence["evidence"]
        for selector in confidence["evidence"]:
            path_text, _, test_name = selector.partition("::")
            path = ROOT / path_text
            assert path.is_file()
            assert test_name and f"def {test_name}(" in path.read_text(encoding="utf-8")
        tolerances = confidence.get("tolerances")
        if confidence["tier"] == "equivalent":
            assert tolerances
        if confidence["tier"] == "exact":
            assert tolerances is None
        for tolerance in tolerances or ():
            assert tolerance["operator"] in {"<=", ">="}
            assert math.isfinite(tolerance["value"])


def test_media_pack_bundles_alias_registry() -> None:
    configuration = cast(
        "dict[str, Any]",
        tomllib.loads(
            (ROOT / "packages" / "dinkster-nodes-media-io" / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        ),
    )
    force_include = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include["comfy-aliases.json"] == "dinkster_nodes_media_io_pack/comfy-aliases.json"
