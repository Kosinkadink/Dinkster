"""Generate the media I/O pack's maintained ComfyUI alias registry.

Usage from the Dinkster repository root with a torch interpreter containing
ComfyUI's declared dependencies:

    COMFYUI_ROOT=/path/to/ComfyUI \
      PYTHONPATH=<all local package src paths> \
      /path/to/python tools/gen_media_comfy_aliases.py

The ComfyUI checkout must be clean and pinned to the commit below. Every
source schema is translated from the pinned V3 class; behavior claims cite
tests that replay the pinned goldens (tests/goldens/audio_io_e20d433a.json).

Audio additions also require COMFY_AUDIO_ROOT, VHS_ROOT and AUDIO_CROP_ROOT
at the pins in gen_audio_alias_receipts.py. Use .venv-torch/bin/python and
--update-audio-only to update those entries without regenerating other media.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from dinkster_api.v1 import CORE_COMBO, ComboWidget, InputSpec, TypeExpr
from dinkster_compat_comfy import CompatTranslation, translate_node, translate_v3_schema
from dinkster_schema import (
    InputFamilyMapping,
    InputFamilyMember,
    MappingSource,
    ReplacementCase,
    ReplacementLink,
    ReplacementNode,
    ReplacementPredicate,
    ReplacementRule,
    ValueTransform,
)
from dinkster_schema.model import DynamicComboSpec, DynamicEntry, NodeSchema
from dinkster_schema.replace import rule_to_wire
from dinkster_schema.wire import schema_to_wire
from gen_image_media_comfy_aliases import build_registry as build_image_registry
from gen_video_media_comfy_aliases import merge_video_registry

COMFY_BASELINE = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
CURRENT_COMFY_BASELINE = "95539f56344958339e39b7582a476267d489b0ee"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages" / "dinkster-nodes-media-io" / "comfy-aliases.json"

SAVE_NOTE = (
    "filename_prefix does not carry over; the mounted save target keeps its "
    "declared default. The source audio passthrough output is not preserved: "
    "the native node outputs the saved audio assets instead. The pinned "
    "saver's embedded prompt/extra_pnginfo container metadata is not written."
)

TEXT_SAVE_NOTE = (
    "filename_prefix does not carry over; the mounted save target keeps its "
    "declared default. The source text passthrough output and inline text "
    "display are not preserved: the native node outputs the saved text "
    "assets, and inline value display is the named dinkster.preview_any "
    "disposition. Bytes are written as UTF-8 without platform newline "
    "translation, matching the pinned saver's POSIX output exactly. "
    "Rendered output above 64 MiB fails closed; the pinned saver has no "
    "size bound."
)

RECORD_NOTE = (
    "The pinned node records in the browser at queue time and uploads the "
    "take; the server side only loads that uploaded file. The native node "
    "records on a server-side device through the configured audio capture "
    "provider: the recorded-upload widget value does not carry over, and "
    "recording duration, sample rate, and channels become explicit bounded "
    "inputs."
)

WEBCAM_NOTE = (
    "The pinned node captures in the browser at queue time and uploads the "
    "frame; the server side only loads that uploaded image and ignores "
    "width, height, and capture_on_queue. The native node captures on a "
    "server-side device through the configured video capture provider: "
    "width and height carry over as the requested capture resolution, and "
    "capture_on_queue and the client-captured image value do not carry over."
)

ASSEMBLE_NOTE = (
    "The pinned node wraps lazy components that a downstream saver encodes; "
    "the native node encodes immediately to mp4/webm container bytes with "
    "explicit format, crf, and bit_depth controls (defaults mp4_h264, crf "
    "23, 8-bit). The pinned bit_depth (auto/8/10) and color_space inputs do "
    "not carry over: the native combo is 8/10 with no auto, and HDR color "
    "spaces are not supported, so both sides default to 8-bit sRGB."
)

DISASSEMBLE_NOTE = (
    "The pinned node reads lazy components without re-encoding; the native "
    "node decodes encoded container bytes, so frames and audio reflect one "
    "codec round trip. The pinned bit_depth and color_space outputs are not "
    "preserved; the native node adds frame_count and duration outputs."
)

AUDIO_OPS_GAP = (
    "The pinned None-audio passthrough is not carried: native audio inputs "
    "are required and validated as [B,C,T] float32 mono or stereo within a "
    "256 MiB budget."
)

RESAMPLE_GAP = (
    " Mixed sample rates are matched to the higher rate through "
    "libswresample rather than torchaudio's sinc resampler, so resampled "
    "samples differ numerically; equal-rate results are bit-exact."
)

EQUALIZER_NOTE = (
    AUDIO_OPS_GAP + " The native equalizer is a float64 port of the pinned "
    "float32 torchaudio biquad chain (RBJ shelf and peaking coefficients, "
    "per-stage [-1, 1] clamp, gain-zero stages skipped), so outputs are "
    "value-close rather than bit-exact; the declared tolerance bounds the "
    "worst measured case, a +24 dB shelf at 20 Hz."
)


def _git(comfy_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(comfy_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _record(
    *,
    node_class: str,
    carrier: str,
    rule: ReplacementRule,
    tier: str,
    evidence: list[str],
    tolerances: list[dict[str, object]] | None = None,
    revision: str = "b78cec87",
) -> dict[str, object]:
    confidence: dict[str, object] = {"tier": tier, "evidence": evidence}
    if tolerances is not None:
        confidence["tolerances"] = tolerances
    return {
        "id": f"comfy_alias:comfy-core/{node_class}",
        "mappingKind": "op",
        "carrier": carrier,
        "source": {
            "pack": "comfy-core",
            "nodeClass": node_class,
            "nodeType": rule.from_type,
            "revision": revision,
        },
        "replacement": rule_to_wire(rule),
        "confidence": confidence,
    }


def _core_v3_schema(node_class: type[Any]) -> NodeSchema:
    return translate_v3_schema(node_class.GET_SCHEMA(), CompatTranslation())


def _only_output(schema: NodeSchema) -> str:
    if len(schema.outputs) != 1:
        raise RuntimeError(f"{schema.node_type} must have exactly one output")
    return schema.outputs[0].id


def _static_schema(
    node_class: str,
    *,
    required: dict[str, object],
    returns: tuple[str, ...],
    return_names: tuple[str, ...],
) -> NodeSchema:
    def input_types(_cls: type[Any]) -> dict[str, object]:
        return {"required": required}

    def execute(_self: object) -> None:
        raise RuntimeError("static schema shim")

    shim = type(
        f"_{node_class}",
        (),
        {
            "INPUT_TYPES": classmethod(input_types),
            "RETURN_TYPES": returns,
            "RETURN_NAMES": return_names,
            "FUNCTION": "execute",
            "CATEGORY": "3d",
            "execute": execute,
        },
    )
    return translate_node(node_class, shim, CompatTranslation(), namespace="").schema()


def _model3d_schema(schema: NodeSchema) -> NodeSchema:
    model3d = TypeExpr.concrete("dinkster.model3d")
    return replace(
        schema,
        inputs=tuple(
            replace(input_spec, type=model3d) if input_spec.id == "model_3d" else input_spec
            for input_spec in schema.inputs
        ),
        outputs=tuple(replace(output_spec, type=model3d) for output_spec in schema.outputs),
    )


def _build_model3d_registry() -> dict[str, object]:
    preview = _model3d_schema(
        _static_schema(
            "Preview3DAdvanced",
            required={
                "model_3d": ("FILE_3D_GLB",),
                "viewport_state": ("STRING", {"default": "", "multiline": False}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            },
            returns=("FILE_3D",),
            return_names=("_0_FILE_3D_",),
        )
    )
    save = _model3d_schema(
        _static_schema(
            "Save3DAdvanced",
            required={
                "model_3d": ("FILE_3D_GLB",),
                "filename_prefix": (
                    "STRING",
                    {"default": "3d/ComfyUI", "multiline": False},
                ),
                "viewport_state": ("STRING", {"default": "", "multiline": False}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            },
            returns=("FILE_3D",),
            return_names=("_0_FILE_3D_",),
        )
    )
    evidence = [
        "tests/test_media_comfy_aliases.py::test_trellis2_3d_output_aliases_preserve_models"
    ]
    records = [
        _record(
            node_class="Preview3DAdvanced",
            carrier="dinkster.preview_model3d",
            rule=ReplacementRule(
                from_type=preview.node_type,
                note="Viewport state and explicit viewport dimensions do not carry over.",
                cases=(
                    ReplacementCase.build(
                        "dinkster.preview_model3d",
                        inputs={"model": MappingSource.copy("model_3d")},
                        outputs={"model": "_0_FILE_3D_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=evidence,
            revision="8a33128f",
        ),
        _record(
            node_class="Save3DAdvanced",
            carrier="dinkster.save_model3d",
            rule=ReplacementRule(
                from_type=save.node_type,
                note="Viewport state and explicit viewport dimensions do not carry over.",
                cases=(
                    ReplacementCase.build(
                        "dinkster.save_model3d",
                        nodes={
                            "save_target": ReplacementNode.build(
                                "dinkster.set_save_target_prefix",
                                values={
                                    "target": {
                                        "mount": "comfy-output",
                                        "prefix": "3d/ComfyUI",
                                    }
                                },
                            )
                        },
                        inputs={
                            "model": MappingSource.copy("model_3d"),
                            "save_target:prefix": MappingSource.copy("filename_prefix"),
                        },
                        links=(ReplacementLink("save_target:save_target", "target"),),
                        outputs={"model": "_0_FILE_3D_"},
                    ),
                ),
            ),
            tier="exact",
            evidence=evidence,
            revision="8a33128f",
        ),
    ]
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [schema_to_wire(preview), schema_to_wire(save)],
        "records": records,
    }


def _refusal_case(target_type: str, target_input: str, source_input: str) -> ReplacementCase:
    return ReplacementCase.build(
        target_type,
        inputs={
            target_input: MappingSource.from_value(
                source_input,
                ValueTransform.enum_rename({}),
            )
        },
    )


def _flatten_dynamic_combos(
    schema: NodeSchema,
    *,
    selected_options: dict[str, str] | None = None,
) -> NodeSchema:
    selectors: dict[str, InputSpec] = {}
    child_inputs: dict[str, InputSpec] = {}

    def walk(entries: tuple[DynamicEntry, ...]) -> None:
        for entry in entries:
            if isinstance(entry, InputSpec):
                child_inputs.setdefault(entry.id, replace(entry, required=False))
                continue
            if not isinstance(entry, DynamicComboSpec):
                continue
            options = tuple(option.key for option in entry.options)
            existing = selectors.get(entry.id)
            if existing is None:
                selectors[entry.id] = InputSpec(
                    entry.id,
                    TypeExpr.concrete(CORE_COMBO),
                    required=False,
                    default=entry.default or options[0],
                    widget=ComboWidget(options=options),
                )
            selected = None if selected_options is None else selected_options.get(entry.id)
            for option in entry.options:
                if selected is not None and option.key != selected:
                    continue
                walk(option.inputs)

    walk(schema.combos)
    return replace(
        schema,
        inputs=(*schema.inputs, *selectors.values(), *child_inputs.values()),
        combos=(),
    )


def _build_current_seedvr2_registry(comfy_root: Path) -> dict[str, object]:
    if _git(comfy_root, "rev-parse", "HEAD") != CURRENT_COMFY_BASELINE:
        raise RuntimeError(f"Current ComfyUI must be pinned to {CURRENT_COMFY_BASELINE}")
    if _git(comfy_root, "status", "--porcelain"):
        raise RuntimeError("Current ComfyUI checkout must be clean")
    sys.path.insert(0, str(comfy_root))

    from comfy.cli_args import args as comfy_args  # pyright: ignore[reportMissingImports]

    comfy_args.cpu = True
    from comfy_extras import nodes_images, nodes_video  # pyright: ignore[reportMissingImports]

    load_video = _core_v3_schema(nodes_video.LoadVideo)
    save_video = _flatten_dynamic_combos(_core_v3_schema(nodes_video.SaveVideo))
    video_slice = _core_v3_schema(nodes_video.VideoSlice)
    save_image_advanced = _flatten_dynamic_combos(
        _core_v3_schema(nodes_images.SaveImageAdvanced),
        selected_options={"format": "avif"},
    )
    source_schemas = [load_video, save_video, video_slice, save_image_advanced]
    revision = "95539f56"
    alias_evidence = [
        "tests/test_media_comfy_aliases.py::"
        "test_current_seedvr2_media_aliases_preserve_supported_paths"
    ]
    records = [
        _record(
            node_class="LoadVideo",
            carrier="dinkster.load_video_value",
            rule=ReplacementRule(
                from_type=load_video.node_type,
                cases=(
                    ReplacementCase.build(
                        "dinkster.load_video_value",
                        inputs={"video": MappingSource.copy("file")},
                        outputs={"video": _only_output(load_video)},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                *alias_evidence,
                "tests/test_video_io.py::test_video_value_load_and_save_preserve_container_bytes",
            ],
            revision=revision,
        ),
        _record(
            node_class="SaveVideo",
            carrier="dinkster.save_video_value",
            rule=ReplacementRule(
                from_type=save_video.node_type,
                note=(
                    "Automatic container and codec selection preserve MP4 or WebM bytes. "
                    "Explicit format or codec selection fails closed. The source VIDEO "
                    "passthrough output and embedded prompt metadata are not preserved."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.save_video_value",
                        when=ReplacementPredicate.all_of(
                            ReplacementPredicate.value_equals("format", "auto"),
                            ReplacementPredicate.value_equals("codec", "auto"),
                        ),
                        nodes={
                            "save_target": ReplacementNode.build(
                                "dinkster.set_save_target_prefix",
                                values={
                                    "target": {
                                        "mount": "comfy-output",
                                        "prefix": "video/ComfyUI",
                                    }
                                },
                            )
                        },
                        inputs={
                            "video": MappingSource.copy("video"),
                            "save_target:prefix": MappingSource.copy("filename_prefix"),
                        },
                        links=(ReplacementLink("save_target:save_target", "target"),),
                    ),
                    _refusal_case("dinkster.save_video_value", "video", "format"),
                ),
            ),
            tier="parametric",
            evidence=[
                *alias_evidence,
                "tests/test_video_io.py::test_video_value_load_and_save_preserve_container_bytes",
            ],
            revision=revision,
        ),
        _record(
            node_class="Video Slice",
            carrier="dinkster.video.trim",
            rule=ReplacementRule(
                from_type=video_slice.node_type,
                note=(
                    "A zero start and unlimited duration preserve container bytes. Nonzero "
                    "windows decode and re-encode instead of preserving ComfyUI's lazy "
                    "trim, so codec bytes and decoded samples can differ."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.video.trim",
                        inputs={
                            "video": MappingSource.copy("video"),
                            "start_time": MappingSource.copy("start_time"),
                            "duration": MappingSource.copy("duration"),
                            "strict_duration": MappingSource.copy("strict_duration"),
                        },
                        outputs={"video": _only_output(video_slice)},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                *alias_evidence,
                "tests/test_video_ops.py::test_trim_zero_window_preserves_container_bytes",
                "tests/test_video_ops.py::test_trim_selects_requested_frame_window",
            ],
            revision=revision,
        ),
        _record(
            node_class="SaveImageAdvanced",
            carrier="dinkster.save_avif",
            rule=ReplacementRule(
                from_type=save_image_advanced.node_type,
                note=(
                    "Only the AVIF branches carry over. filename_prefix becomes a literal "
                    "mounted relative path without ComfyUI substitutions. Hidden prompt and "
                    "extra_pnginfo metadata are not available during document migration. "
                    "Native encoding fails closed above the 1 GiB output bound."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.save_avif",
                        when=ReplacementPredicate.all_of(
                            ReplacementPredicate.value_equals("format", "avif"),
                            ReplacementPredicate.value_equals("save_mode", "still images"),
                        ),
                        nodes={
                            "save_target": ReplacementNode.build(
                                "dinkster.set_save_target_prefix",
                                values={
                                    "target": {
                                        "mount": "comfy-output",
                                        "prefix": "ComfyUI",
                                    }
                                },
                            )
                        },
                        inputs={
                            "images": MappingSource.copy("images"),
                            "bit_depth": MappingSource.copy("bit_depth"),
                            "input_color_space": MappingSource.copy("input_color_space"),
                            "crf": MappingSource.copy("crf"),
                            "animated": MappingSource.constant(False),
                            "save_target:prefix": MappingSource.copy("filename_prefix"),
                        },
                        links=(ReplacementLink("save_target:save_target", "target"),),
                        outputs={"images": _only_output(save_image_advanced)},
                    ),
                    ReplacementCase.build(
                        "dinkster.save_avif",
                        when=ReplacementPredicate.all_of(
                            ReplacementPredicate.value_equals("format", "avif"),
                            ReplacementPredicate.value_equals("save_mode", "animated"),
                        ),
                        nodes={
                            "save_target": ReplacementNode.build(
                                "dinkster.set_save_target_prefix",
                                values={
                                    "target": {
                                        "mount": "comfy-output",
                                        "prefix": "ComfyUI",
                                    }
                                },
                            )
                        },
                        inputs={
                            "images": MappingSource.copy("images"),
                            "bit_depth": MappingSource.copy("bit_depth"),
                            "input_color_space": MappingSource.copy("input_color_space"),
                            "crf": MappingSource.copy("crf"),
                            "animated": MappingSource.constant(True),
                            "fps": MappingSource.copy("fps"),
                            "loop": MappingSource.copy("loop_count"),
                            "save_target:prefix": MappingSource.copy("filename_prefix"),
                        },
                        links=(ReplacementLink("save_target:save_target", "target"),),
                        outputs={"images": _only_output(save_image_advanced)},
                    ),
                    _refusal_case("dinkster.save_avif", "images", "format"),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_avif_alias.py::test_save_image_advanced_avif_alias_executes_still_and_animated"
            ],
            revision=revision,
        ),
    ]
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [schema_to_wire(schema) for schema in source_schemas],
        "records": records,
    }


def _build_audio_additions() -> dict[str, object]:
    from gen_audio_alias_receipts import COMFY_PIN, CROP_PIN, VHS_PIN, load_sources

    nodes_audio, vhs, crop, _provenance = load_sources()
    # Dynamic upload choices come from an empty directory, not a developer's files.
    import tempfile

    import folder_paths  # pyright: ignore[reportMissingImports]

    with tempfile.TemporaryDirectory() as directory:
        folder_paths.set_input_directory(directory)
        load = translate_node("VHS_LoadAudio", vhs["LoadAudio"], CompatTranslation()).schema()
        upload = translate_node(
            "VHS_LoadAudioUpload", vhs["LoadAudioUpload"], CompatTranslation()
        ).schema()
    crop_schema = translate_node("AudioCrop", crop, CompatTranslation()).schema()
    dynamic_save = _core_v3_schema(nodes_audio.SaveAudioAdvanced)
    qualities = tuple(
        dict.fromkeys(
            quality
            for combo in dynamic_save.combos
            if isinstance(combo, DynamicComboSpec)
            for option in combo.options
            for spec in option.inputs
            if isinstance(spec, InputSpec) and isinstance(spec.widget, ComboWidget)
            for quality in spec.widget.options
        )
    )
    save = _flatten_dynamic_combos(dynamic_save)
    save = replace(
        save,
        inputs=tuple(
            replace(
                spec,
                default=None,
                widget=ComboWidget(options=qualities),
            )
            if spec.id == "quality"
            else spec
            for spec in save.inputs
        ),
    )
    records = []
    for schema, source_input, start_input in (
        (load, "audio_file", "seek_seconds"),
        (upload, "audio", "start_time"),
    ):
        node_class = schema.node_type.removeprefix("comfy.")
        record = _record(
            node_class=node_class,
            carrier="dinkster.load_audio",
            revision=VHS_PIN,
            tier="parametric",
            evidence=["tests/test_audio_alias_replay.py::test_vhs_load_replay"],
            rule=ReplacementRule(
                from_type=schema.node_type,
                note=(
                    "Select the source as a portable library asset rather than a host path/URL. "
                    "Start and duration are lazy seconds-based trims; duration=0 reads to end. "
                    "The source duration output is not mapped: a native audio facts output is "
                    "still required. Native channel support follows the installed decoder, "
                    "not the pinned VHS mono/stereo parser."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.load_audio",
                        inputs={
                            "audio": MappingSource.copy(source_input),
                            "start_time": MappingSource.copy(start_input),
                            "duration": MappingSource.copy("duration"),
                        },
                        outputs={"audio": schema.outputs[0].id},
                    ),
                ),
            ),
        )
        record["id"] = f"comfy_alias:comfyui-videohelpersuite/{node_class}"
        cast("dict[str, object]", record["source"])["pack"] = "comfyui-videohelpersuite"
        records.append(record)

    # Expression families require a typed connection even for unused operands.
    crop_nodes = {"zero": ReplacementNode.build("dinkster.int", values={"value": 0})}
    crop_inputs = {"audio": MappingSource.copy("audio")}
    crop_links = []
    for name in ("start", "end"):
        crop_nodes[name + "_decimal"] = ReplacementNode.build(
            "dinkster.string.regex",
            values={"operation": "replace", "pattern": r"\b0+(?=\d)", "replacement": ""},
        )
        crop_nodes[name + "_text"] = ReplacementNode.build(
            "dinkster.string.transform",
            values={"operation": "replace_literal", "value": ":", "replacement": "*60+"},
        )
        crop_nodes[name + "_seconds"] = ReplacementNode.build("dinkster.math.expression")
        crop_inputs[name + "_decimal:text"] = MappingSource.copy(name + "_time")
        crop_links.append(ReplacementLink(name + "_decimal:text", name + "_text:text"))
        crop_links.append(ReplacementLink(name + "_text:text", name + "_seconds:expression"))
        crop_links.append(ReplacementLink("zero:value", name + "_seconds:values.a"))
    crop_nodes["start_clamp"] = ReplacementNode.build(
        "dinkster.math.expression", values={"expression": "max(0,a)"}
    )
    crop_nodes["duration"] = ReplacementNode.build(
        "dinkster.math.expression", values={"expression": "max(0,b)-max(0,a)"}
    )
    crop_links.extend(
        (
            ReplacementLink("start_seconds:float", "start_clamp:values.a"),
            ReplacementLink("start_seconds:float", "duration:values.a"),
            ReplacementLink("end_seconds:float", "duration:values.b"),
            ReplacementLink("start_clamp:float", "start"),
            ReplacementLink("duration:float", "duration"),
        )
    )
    crop_record = _record(
        node_class="AudioCrop",
        carrier="dinkster.audio.trim",
        revision=CROP_PIN,
        tier="parametric",
        evidence=[
            "tests/test_audio_alias_replay.py::test_crop_replay",
            "tests/test_audio_alias_replay.py::test_crop_template_import",
        ],
        rule=ReplacementRule(
            from_type=crop_schema.node_type,
            note=(
                "MM:SS and seconds strings become explicit seconds expressions and a lazy trim. "
                "Negative endpoints clamp to zero. Native trim retains the last sample at EOF "
                "where the source excludes it, and rejects empty ranges which the source accepts. "
                "Expression nodes also accept numeric expressions beyond the source integer syntax."
            ),
            cases=(
                ReplacementCase.build(
                    "dinkster.audio.trim",
                    nodes=crop_nodes,
                    inputs=crop_inputs,
                    input_families={
                        name + ":values": InputFamilyMapping.from_members(
                            *(
                                InputFamilyMember.build(
                                    member, inputs={"value": MappingSource.constant(0)}
                                )
                                for member in members
                            )
                        )
                        for name, members in (
                            ("start_seconds", "a"),
                            ("end_seconds", "a"),
                            ("start_clamp", "a"),
                            ("duration", "ab"),
                        )
                    },
                    links=crop_links,
                    outputs={"audio": _only_output(crop_schema)},
                ),
            ),
        ),
    )
    crop_record["id"] = "comfy_alias:audio-separation-nodes-comfyui/AudioCrop"
    cast("dict[str, object]", crop_record["source"])["pack"] = "audio-separation-nodes-comfyui"
    records.append(crop_record)

    save_cases = []
    for format, target in (
        ("mp3", "dinkster.save_audio_mp3"),
        ("opus", "dinkster.save_audio_opus"),
        ("flac", "dinkster.save_audio"),
    ):
        inputs = {
            "passthrough:audio": MappingSource.copy("audio"),
            "save_target:prefix": MappingSource.copy("filename_prefix"),
        }
        if format != "flac":
            inputs["quality"] = MappingSource.copy("quality")
        save_cases.append(
            ReplacementCase.build(
                target,
                when=(
                    ReplacementPredicate.always()
                    if format == "flac"
                    else ReplacementPredicate.value_equals("format", format)
                ),
                nodes={
                    "save_target": ReplacementNode.build(
                        "dinkster.set_save_target_prefix",
                        values={
                            "target": {"mount": "comfy-output", "prefix": "audio/ComfyUI"},
                        },
                    ),
                    "passthrough": ReplacementNode.build("dinkster.preview_audio"),
                },
                inputs=inputs,
                links=(
                    ReplacementLink("save_target:save_target", "target"),
                    ReplacementLink("passthrough:audio", "audio"),
                ),
                outputs={"passthrough:audio": _only_output(save)},
            )
        )
    records.append(
        _record(
            node_class="SaveAudioAdvanced",
            carrier="dinkster.save_audio",
            revision=COMFY_PIN[:8],
            tier="parametric",
            evidence=["tests/test_audio_alias_replay.py::test_advanced_save_replay"],
            rule=ReplacementRule(
                from_type=save.node_type,
                note=(
                    "Format/quality select native FLAC, MP3 or Opus savers; filename_prefix "
                    "configures the mounted save target. An audio preview preserves passthrough. "
                    "An unset or unrecognized format falls back to FLAC. "
                    "Missing quality uses native MP3 V0/Opus 128k defaults; a direct source "
                    "call without quality uses 128k for both. "
                    "ComfyUI filename token expansion and embedded prompt/extra_pnginfo metadata "
                    "do not carry over. Opus re-encodes follow installed codecs and libswresample "
                    "rather than torchaudio at coerced rates (existing decoded tolerance 0.05). "
                    "sampler_rate is annotation-only in ComfyUI AudioDict; "
                    "runtime uses sample_rate."
                ),
                cases=tuple(save_cases),
            ),
        )
    )
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [schema_to_wire(schema) for schema in (load, upload, crop_schema, save)],
        "records": records,
    }


def _audio_additions_in_subprocess() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--audio-additions"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _build_audio_registry(comfy_root: Path) -> dict[str, object]:
    if _git(comfy_root, "rev-parse", "HEAD") != COMFY_BASELINE:
        raise RuntimeError(f"ComfyUI must be pinned to {COMFY_BASELINE}")
    if _git(comfy_root, "status", "--porcelain"):
        raise RuntimeError("ComfyUI checkout must be clean")
    sys.path.insert(0, str(comfy_root))

    from comfy.cli_args import args as comfy_args  # pyright: ignore[reportMissingImports]

    comfy_args.cpu = True
    from comfy_extras import (  # pyright: ignore[reportMissingImports]
        nodes_audio,
        nodes_text,
        nodes_video,
        nodes_webcam,
    )

    load_audio = _core_v3_schema(nodes_audio.LoadAudio)
    empty_audio = _core_v3_schema(nodes_audio.EmptyAudio)
    save_audio = _core_v3_schema(nodes_audio.SaveAudio)
    save_audio_mp3 = _core_v3_schema(nodes_audio.SaveAudioMP3)
    save_audio_opus = _core_v3_schema(nodes_audio.SaveAudioOpus)
    preview_audio = _core_v3_schema(nodes_audio.PreviewAudio)
    record_audio = _core_v3_schema(nodes_audio.RecordAudio)
    trim_audio = _core_v3_schema(nodes_audio.TrimAudioDuration)
    split_audio = _core_v3_schema(nodes_audio.SplitAudioChannels)
    join_audio = _core_v3_schema(nodes_audio.JoinAudioChannels)
    concat_audio = _core_v3_schema(nodes_audio.AudioConcat)
    merge_audio = _core_v3_schema(nodes_audio.AudioMerge)
    volume_audio = _core_v3_schema(nodes_audio.AudioAdjustVolume)
    equalizer_audio = _core_v3_schema(nodes_audio.AudioEqualizer3Band)
    save_text = _core_v3_schema(nodes_text.SaveTextNode)
    create_video = _core_v3_schema(nodes_video.CreateVideo)
    get_video_components = _core_v3_schema(nodes_video.GetVideoComponents)
    # WebcamCapture is a V1 class, so its schema comes through the same
    # translation the compat layer applies at runtime.
    webcam_capture = translate_node(
        "WebcamCapture", nodes_webcam.WebcamCapture, CompatTranslation()
    ).schema()
    source_schemas = [
        load_audio,
        empty_audio,
        save_audio,
        save_audio_mp3,
        save_audio_opus,
        preview_audio,
        record_audio,
        trim_audio,
        split_audio,
        join_audio,
        concat_audio,
        merge_audio,
        volume_audio,
        equalizer_audio,
        save_text,
        create_video,
        get_video_components,
        webcam_capture,
    ]

    flac_mp3_evidence = [
        "tests/test_audio_comfy_goldens.py::"
        "test_flac_and_mp3_saves_decode_identically_to_pinned_encodes"
    ]
    records = [
        _record(
            node_class="LoadAudio",
            carrier="dinkster.load_audio",
            rule=ReplacementRule(
                from_type=load_audio.node_type,
                cases=(
                    ReplacementCase.build(
                        "dinkster.load_audio",
                        inputs={"audio": MappingSource.copy("audio")},
                        outputs={"audio": _only_output(load_audio)},
                    ),
                ),
            ),
            tier="exact",
            evidence=[
                "tests/test_audio_comfy_goldens.py::"
                "test_load_wav_decodes_bit_exactly_to_pinned_loader"
            ],
        ),
        _record(
            node_class="EmptyAudio",
            carrier="dinkster.empty_audio",
            rule=ReplacementRule(
                from_type=empty_audio.node_type,
                note=(
                    "Duration is bounded at 86400 seconds and a 256 MiB waveform "
                    "budget; longer requests fail closed instead of allocating."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.empty_audio",
                        inputs={
                            "duration": MappingSource.copy("duration"),
                            "sample_rate": MappingSource.copy("sample_rate"),
                            "channels": MappingSource.copy("channels"),
                        },
                        outputs={"audio": _only_output(empty_audio)},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_audio_comfy_goldens.py::"
                "test_empty_audio_matches_pinned_shapes_and_rates",
                "tests/test_audio_io.py::test_empty_audio_rejects_out_of_domain_requests",
            ],
        ),
        _record(
            node_class="SaveAudio",
            carrier="dinkster.save_audio",
            rule=ReplacementRule(
                from_type=save_audio.node_type,
                note=SAVE_NOTE,
                cases=(
                    ReplacementCase.build(
                        "dinkster.save_audio",
                        inputs={"audio": MappingSource.copy("audio")},
                    ),
                ),
            ),
            tier="parametric",
            evidence=flac_mp3_evidence,
        ),
        _record(
            node_class="SaveAudioMP3",
            carrier="dinkster.save_audio_mp3",
            rule=ReplacementRule(
                from_type=save_audio_mp3.node_type,
                note=SAVE_NOTE,
                cases=(
                    ReplacementCase.build(
                        "dinkster.save_audio_mp3",
                        inputs={
                            "audio": MappingSource.copy("audio"),
                            "quality": MappingSource.copy("quality"),
                        },
                    ),
                ),
            ),
            tier="parametric",
            evidence=flac_mp3_evidence,
        ),
        _record(
            node_class="SaveAudioOpus",
            carrier="dinkster.save_audio_opus",
            rule=ReplacementRule(
                from_type=save_audio_opus.node_type,
                note=(
                    SAVE_NOTE + " Decoded Opus differs from the pinned encoder "
                    "through the libopus version delta and the resampler used "
                    "for coerced rates; shapes and the 48 kHz decode rate match."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.save_audio_opus",
                        inputs={
                            "audio": MappingSource.copy("audio"),
                            "quality": MappingSource.copy("quality"),
                        },
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_audio_comfy_goldens.py::"
                "test_opus_saves_decode_within_measured_tolerance"
            ],
            tolerances=[
                {"metric": "decoded_waveform_max_abs_diff", "operator": "<=", "value": 0.05}
            ],
        ),
        _record(
            node_class="PreviewAudio",
            carrier="dinkster.preview_audio",
            rule=ReplacementRule(
                from_type=preview_audio.node_type,
                note=(
                    "The native preview displays the typed audio value without "
                    "writing a temporary FLAC file."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.preview_audio",
                        inputs={"audio": MappingSource.copy("audio")},
                        outputs={"audio": _only_output(preview_audio)},
                    ),
                ),
            ),
            tier="exact",
            evidence=["tests/test_audio_io.py::test_preview_validates_and_passes_audio_through"],
        ),
        _record(
            node_class="TrimAudioDuration",
            carrier="dinkster.audio.trim",
            rule=ReplacementRule(
                from_type=trim_audio.node_type,
                note=AUDIO_OPS_GAP,
                cases=(
                    ReplacementCase.build(
                        "dinkster.audio.trim",
                        inputs={
                            "audio": MappingSource.copy("audio"),
                            "start": MappingSource.copy("start_index"),
                            "duration": MappingSource.copy("duration"),
                        },
                        outputs={"audio": _only_output(trim_audio)},
                    ),
                ),
            ),
            tier="exact",
            evidence=["tests/test_audio_ops_goldens.py::test_trim_matches_pinned_bytes_exactly"],
        ),
        _record(
            node_class="SplitAudioChannels",
            carrier="dinkster.audio.split_channels",
            rule=ReplacementRule(
                from_type=split_audio.node_type,
                note=AUDIO_OPS_GAP,
                cases=(
                    ReplacementCase.build(
                        "dinkster.audio.split_channels",
                        inputs={"audio": MappingSource.copy("audio")},
                        outputs={
                            "left": split_audio.outputs[0].id,
                            "right": split_audio.outputs[1].id,
                        },
                    ),
                ),
            ),
            tier="exact",
            evidence=["tests/test_audio_ops_goldens.py::test_split_matches_pinned_bytes_exactly"],
        ),
        _record(
            node_class="JoinAudioChannels",
            carrier="dinkster.audio.join_channels",
            rule=ReplacementRule(
                from_type=join_audio.node_type,
                note=AUDIO_OPS_GAP + RESAMPLE_GAP,
                cases=(
                    ReplacementCase.build(
                        "dinkster.audio.join_channels",
                        inputs={
                            "audio_left": MappingSource.copy("audio_left"),
                            "audio_right": MappingSource.copy("audio_right"),
                        },
                        outputs={"audio": _only_output(join_audio)},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_audio_ops_goldens.py::test_join_matches_pinned_bytes_exactly",
                "tests/test_audio_ops.py::"
                "test_join_resamples_the_lower_rate_input_to_the_higher_rate",
            ],
        ),
        _record(
            node_class="AudioConcat",
            carrier="dinkster.audio.concat",
            rule=ReplacementRule(
                from_type=concat_audio.node_type,
                note=AUDIO_OPS_GAP + RESAMPLE_GAP,
                cases=(
                    ReplacementCase.build(
                        "dinkster.audio.concat",
                        inputs={
                            "audio1": MappingSource.copy("audio1"),
                            "audio2": MappingSource.copy("audio2"),
                            "direction": MappingSource.copy("direction"),
                        },
                        outputs={"audio": _only_output(concat_audio)},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_audio_ops_goldens.py::test_concat_matches_pinned_bytes_exactly",
                "tests/test_audio_ops.py::test_concat_always_duplicates_mono_to_stereo_as_pinned",
            ],
        ),
        _record(
            node_class="AudioMerge",
            carrier="dinkster.audio.merge",
            rule=ReplacementRule(
                from_type=merge_audio.node_type,
                note=AUDIO_OPS_GAP + RESAMPLE_GAP,
                cases=(
                    ReplacementCase.build(
                        "dinkster.audio.merge",
                        inputs={
                            "audio1": MappingSource.copy("audio1"),
                            "audio2": MappingSource.copy("audio2"),
                            "merge_method": MappingSource.copy("merge_method"),
                        },
                        outputs={"audio": _only_output(merge_audio)},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_audio_ops_goldens.py::test_merge_matches_pinned_bytes_exactly",
                "tests/test_audio_ops.py::test_merge_peak_normalizes_only_above_unity",
            ],
        ),
        _record(
            node_class="AudioAdjustVolume",
            carrier="dinkster.audio.volume",
            rule=ReplacementRule(
                from_type=volume_audio.node_type,
                note=(
                    AUDIO_OPS_GAP + " The pinned integer decibel volume input "
                    "widens to the float gain_db input; integer values apply "
                    "bit-exactly."
                ),
                cases=(
                    ReplacementCase.build(
                        "dinkster.audio.volume",
                        inputs={
                            "audio": MappingSource.copy("audio"),
                            "gain_db": MappingSource.copy("volume"),
                        },
                        outputs={"audio": _only_output(volume_audio)},
                    ),
                ),
            ),
            tier="exact",
            evidence=["tests/test_audio_ops_goldens.py::test_volume_matches_pinned_bytes_exactly"],
        ),
        _record(
            node_class="AudioEqualizer3Band",
            carrier="dinkster.audio.equalizer",
            rule=ReplacementRule(
                from_type=equalizer_audio.node_type,
                note=EQUALIZER_NOTE,
                cases=(
                    ReplacementCase.build(
                        "dinkster.audio.equalizer",
                        inputs={
                            "audio": MappingSource.copy("audio"),
                            "low_gain_db": MappingSource.copy("low_gain_dB"),
                            "low_freq": MappingSource.copy("low_freq"),
                            "mid_gain_db": MappingSource.copy("mid_gain_dB"),
                            "mid_freq": MappingSource.copy("mid_freq"),
                            "mid_q": MappingSource.copy("mid_q"),
                            "high_gain_db": MappingSource.copy("high_gain_dB"),
                            "high_freq": MappingSource.copy("high_freq"),
                        },
                        outputs={"audio": _only_output(equalizer_audio)},
                    ),
                ),
            ),
            tier="equivalent",
            evidence=[
                "tests/test_audio_ops_goldens.py::test_equalizer_is_value_close_to_the_pinned_chain"
            ],
            tolerances=[{"metric": "waveform_max_abs_diff", "operator": "<=", "value": 0.04}],
        ),
        _record(
            node_class="SaveText",
            carrier="dinkster.save_text",
            rule=ReplacementRule(
                from_type=save_text.node_type,
                note=TEXT_SAVE_NOTE,
                cases=(
                    ReplacementCase.build(
                        "dinkster.save_text",
                        inputs={
                            "text": MappingSource.copy("text"),
                            "format": MappingSource.copy("format"),
                        },
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_text_comfy_goldens.py::test_rendered_bytes_match_pinned_save_exactly"
            ],
        ),
        _record(
            node_class="CreateVideo",
            carrier="dinkster.video.assemble",
            rule=ReplacementRule(
                from_type=create_video.node_type,
                note=ASSEMBLE_NOTE,
                cases=(
                    ReplacementCase.build(
                        "dinkster.video.assemble",
                        inputs={
                            "images": MappingSource.copy("images"),
                            "fps": MappingSource.copy("fps"),
                            "audio": MappingSource.copy("audio"),
                        },
                        outputs={"video": _only_output(create_video)},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_video_ops.py::"
                "test_assemble_disassemble_round_trip_preserves_frames_fps_and_audio"
            ],
        ),
        _record(
            node_class="GetVideoComponents",
            carrier="dinkster.video.disassemble",
            rule=ReplacementRule(
                from_type=get_video_components.node_type,
                note=DISASSEMBLE_NOTE,
                cases=(
                    ReplacementCase.build(
                        "dinkster.video.disassemble",
                        inputs={"video": MappingSource.copy("video")},
                        outputs={
                            "images": get_video_components.outputs[0].id,
                            "audio": get_video_components.outputs[1].id,
                            "fps": get_video_components.outputs[2].id,
                        },
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_video_ops.py::"
                "test_assemble_disassemble_round_trip_preserves_frames_fps_and_audio",
                "tests/test_video_ops.py::test_rate_matches_load_video_force_rate_decode",
            ],
        ),
        _record(
            node_class="RecordAudio",
            carrier="dinkster.record_audio",
            rule=ReplacementRule(
                from_type=record_audio.node_type,
                note=RECORD_NOTE,
                cases=(
                    ReplacementCase.build(
                        "dinkster.record_audio",
                        outputs={"audio": _only_output(record_audio)},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_capture_io.py::"
                "test_record_audio_returns_typed_audio_from_injected_provider",
                "tests/test_capture_io.py::test_schema_discovery_never_touches_a_provider",
            ],
        ),
        _record(
            node_class="WebcamCapture",
            carrier="dinkster.webcam_capture",
            rule=ReplacementRule(
                from_type=webcam_capture.node_type,
                note=WEBCAM_NOTE,
                cases=(
                    ReplacementCase.build(
                        "dinkster.webcam_capture",
                        inputs={
                            "width": MappingSource.copy("width"),
                            "height": MappingSource.copy("height"),
                        },
                        outputs={"image": _only_output(webcam_capture)},
                    ),
                ),
            ),
            tier="parametric",
            evidence=[
                "tests/test_capture_io.py::"
                "test_webcam_capture_returns_bhwc_image_from_injected_provider",
                "tests/test_capture_io.py::test_webcam_capture_requested_resolution_is_binding",
            ],
        ),
    ]
    additions = _audio_additions_in_subprocess()
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [
            *[schema_to_wire(schema) for schema in source_schemas],
            *additions["sourceSchemas"],
        ],
        "records": [*records, *additions["records"]],
    }


def build_registry(comfy_root: Path) -> dict[str, object]:
    audio = _build_audio_registry(comfy_root)
    image = build_image_registry(comfy_root)
    model3d = _build_model3d_registry()
    audio_schemas = cast("list[object]", audio["sourceSchemas"])
    image_schemas = cast("list[object]", image["sourceSchemas"])
    model3d_schemas = cast("list[object]", model3d["sourceSchemas"])
    audio_records = cast("list[object]", audio["records"])
    image_records = cast("list[object]", image["records"])
    model3d_records = cast("list[object]", model3d["records"])
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [*audio_schemas, *image_schemas, *model3d_schemas],
        "records": [*audio_records, *image_records, *model3d_records],
    }


def _current_registry_in_subprocess(comfy_root: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--current-seedvr2", str(comfy_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    return cast("dict[str, object]", json.loads(completed.stdout))


def main() -> None:
    if sys.argv[1:] == ["--audio-additions"]:
        print(json.dumps(_build_audio_additions()))
        return
    if sys.argv[1:] == ["--update-audio-only"]:
        registry = json.loads(OUT.read_text("utf-8"))
        additions = _audio_additions_in_subprocess()
        for key, identity in (("sourceSchemas", "nodeType"), ("records", "id")):
            replacing = {item[identity] for item in additions[key]}
            registry[key] = [item for item in registry[key] if item[identity] not in replacing]
            registry[key].extend(additions[key])
        OUT.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--current-seedvr2":
        print(json.dumps(_build_current_seedvr2_registry(Path(sys.argv[2]).resolve())))
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--update-current":
        registry = json.loads(OUT.read_text("utf-8"))
        current = _build_current_seedvr2_registry(Path(sys.argv[2]).resolve())
        current["sourceSchemas"] = [
            item
            for item in current["sourceSchemas"]
            if item["nodeType"] == "comfy.SaveImageAdvanced"
        ]
        current["records"] = [
            item
            for item in current["records"]
            if item["source"]["nodeClass"] == "SaveImageAdvanced"
        ]
        for key, identity in (("sourceSchemas", "nodeType"), ("records", "id")):
            replacing = {item[identity] for item in current[key]}
            registry[key] = [item for item in registry[key] if item[identity] not in replacing]
            registry[key].extend(current[key])
        OUT.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        return
    configured = os.environ.get("COMFYUI_ROOT")
    comfy_root = Path(configured) if configured else REPO.parent / "ComfyUI"
    current_configured = os.environ.get("CURRENT_COMFYUI_ROOT")
    if not current_configured:
        raise RuntimeError("CURRENT_COMFYUI_ROOT must point to the pinned current ComfyUI checkout")
    registry = build_registry(comfy_root.resolve())
    current = _current_registry_in_subprocess(Path(current_configured).resolve())
    registry["sourceSchemas"] = [
        *cast("list[object]", registry["sourceSchemas"]),
        *cast("list[object]", current["sourceSchemas"]),
    ]
    registry["records"] = [
        *cast("list[object]", registry["records"]),
        *cast("list[object]", current["records"]),
    ]
    video_root = os.environ.get("VIDEO_COMFYUI_ROOT")
    if not video_root:
        raise RuntimeError("VIDEO_COMFYUI_ROOT must point to the pinned VIDEO ComfyUI checkout")
    video = subprocess.run(
        [sys.executable, str(REPO / "tools/gen_video_media_comfy_aliases.py"), "--stdout"],
        env={**os.environ, "COMFYUI_ROOT": video_root},
        check=True,
        capture_output=True,
        text=True,
    )
    registry = merge_video_registry(registry, json.loads(video.stdout))
    content = (json.dumps(registry, indent=2) + "\n").encode()
    OUT.write_bytes(content)
    print(f"{OUT}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
