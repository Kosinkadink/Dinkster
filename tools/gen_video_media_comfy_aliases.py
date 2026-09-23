"""Pinned VIDEO aliases. Run with the reference interpreter and COMFYUI_ROOT.

Only VIDEO records are replaced in the maintained media registry; unrelated
source schemas retain their own reference pins and byte representation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from dinkster_api.v1 import ComboWidget, InputSpec, TypeExpr
from dinkster_compat_comfy import CompatTranslation, translate_v3_schema
from dinkster_schema import (
    InputFamilyMapping,
    ReplacementRule,
    ValueTransform,
)
from dinkster_schema import (
    MappingSource as M,
)
from dinkster_schema import (
    ReplacementCase as C,
)
from dinkster_schema import (
    ReplacementLink as L,
)
from dinkster_schema import (
    ReplacementNode as N,
)
from dinkster_schema import (
    ReplacementPredicate as P,
)
from dinkster_schema.model import DynamicComboSpec, NodeSchema
from dinkster_schema.replace import rule_to_wire
from dinkster_schema.wire import schema_to_wire

BASELINE = "95539f56344958339e39b7582a476267d489b0ee"
VIDEO_CLASSES = {
    "ConcatenateVideo",
    "LoadVideo",
    "SaveVideo",
    "SaveWEBM",
    "CreateVideo",
    "GetVideoComponents",
    "Video Slice",
    "VideoTrim",
    "VideoCrop",
}
OUT = Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-media-io/comfy-aliases.json"
TRIM_NOTE = (
    "Edits stay lazy. Native successive trims intersect the current clip; ComfyUI "
    "file trims add starts but replace duration. Negative starts are relative to "
    "the effective native clip, whereas ComfyUI resolves its composed file start "
    "against the original duration and clamps at zero. Non-strict out-of-bounds "
    "native windows intersect available time; ComfyUI components reject overruns "
    "even when non-strict. Strict native overruns fail. Safe packet-copy saving "
    "exceeds ComfyUI, not parity. VIDEO_EDIT is copied verbatim, including unused "
    "sections. The advanced node's preview publication is not reproduced."
)
SAVE_NOTE = (
    "Auto container lowers to WebM for AV1, otherwise MP4, not native source-preserving "
    "auto. All container/codec/encoding selections carry over, including historical "
    "flat controls. CRF passes through an explicit Convert Value node to integer, "
    "truncating fractional values toward zero (force_lossy=true); integer CRFs are "
    "unchanged. Specifying CRF requests encoding. Invalid codec/container combinations "
    "fail at execution, not translation. VIDEO passthrough and filename_prefix survive. "
    "Explicit metadata JSON carries over; hidden prompt/extra_pnginfo are not source "
    "ports and must be supplied as metadata by the importer. Mount authorization and "
    "the encoded-output bound apply. Safe stream copy exceeds ComfyUI, not parity."
)


def _present(name: str) -> P:
    return P.any_of(P.value_present(name), P.input_connected(name))


def _save_source_schema(schema: NodeSchema) -> NodeSchema:
    """Force named historical inputs to skip positional widget inference."""
    codec = next(combo for combo in schema.combos if combo.id == "codec")
    (encoding_options,) = {
        tuple(option.key for option in entry.options)
        for codec_option in codec.options
        for entry in codec_option.inputs
        if isinstance(entry, DynamicComboSpec) and entry.id == "encoding"
    }
    return replace(
        schema,
        inputs=(
            *schema.inputs,
            InputSpec(
                "encoding",
                TypeExpr.concrete("core.combo"),
                required=False,
                force_input=True,
                widget=ComboWidget(options=encoding_options),
            ),
            InputSpec(
                "encoding.crf", TypeExpr.concrete("core.float"), required=False, force_input=True
            ),
            InputSpec("crf", TypeExpr.concrete("core.float"), required=False, force_input=True),
            InputSpec(
                "metadata", TypeExpr.concrete("core.string"), required=False, force_input=True
            ),
        ),
    )


def _save_case(
    schema: NodeSchema,
    codec_path: str,
    crf_path: str | None,
    *,
    auto_av1: bool = False,
    when: P | None = None,
    default_crf: int | None = None,
) -> tuple[C, C]:
    nodes = {
        "save_target": N.build(
            "dinkster.set_save_target_prefix",
            values={
                "target": {"mount": "comfy-output", "prefix": "video/ComfyUI"},
            },
        )
    }
    inputs = {
        "video": M.copy("video"),
        "save_target:prefix": M.copy("filename_prefix"),
        "codec": M.copy(codec_path),
        "metadata": M.copy("metadata"),
    }
    # Explicit branches avoid changing native auto's source-preserving meaning.
    containers = {
        "auto": "webm" if auto_av1 else "mp4",
        "mp4": "mp4",
        "mkv": "mkv",
        "webm": "webm",
    }
    inputs["container"] = M.from_value("format", ValueTransform.enum_rename(containers))
    links = [L("save_target:save_target", "target")]
    if crf_path is not None:
        nodes["crf_value"] = N.build("dinkster.float")
        nodes["crf"] = N.build(
            "dinkster.value.convert", values={"target": "int", "force_lossy": True}
        )
        inputs["crf_value:value"] = M.copy(crf_path)
        links.extend((L("crf_value:value", "crf:value"), L("crf:int", "crf")))
    elif default_crf is not None:
        inputs["crf"] = M.constant(default_crf)
    case = C.build(
        "dinkster.save_video",
        when=P.all_of(P.value_present("format"), when or P.always()),
        nodes=nodes,
        inputs=inputs,
        links=links,
        outputs={"video": schema.outputs[0].id},
    )
    # Missing stored widgets use the source default, not the native default.
    format_combo = next(spec for spec in schema.combos if spec.id == "format")
    source_default = format_combo.default or format_combo.options[0].key
    return case, replace(
        case,
        when=when,
        inputs=tuple({**inputs, "container": M.constant(containers[str(source_default)])}.items()),
    )


def _save_cases(schema: NodeSchema) -> tuple[C, ...]:
    cases = []
    for codec_path in ("format.codec", "codec"):
        for crf_path in (codec_path + ".encoding.crf", "encoding.crf", "crf"):
            conditions = [_present(codec_path)]
            conditions.append(_present(crf_path))
            for av1 in (True, False):
                guard = (
                    P.all_of(*conditions, P.value_equals(codec_path, "av1"))
                    if av1
                    else P.all_of(*conditions)
                )
                cases.extend(_save_case(schema, codec_path, crf_path, auto_av1=av1, when=guard))
        for codec, crf in (("av1", 30), ("h264", 23)):
            cases.extend(
                _save_case(
                    schema,
                    codec_path,
                    None,
                    auto_av1=codec == "av1",
                    default_crf=crf,
                    when=P.all_of(
                        P.value_equals(codec_path, codec),
                        P.any_of(
                            P.value_equals(codec_path + ".encoding", "re-encode"),
                            P.value_equals("encoding", "re-encode"),
                        ),
                    ),
                )
            )
        cases.extend(
            _save_case(
                schema, codec_path, None, auto_av1=True, when=P.value_equals(codec_path, "av1")
            )
        )
        cases.extend(_save_case(schema, codec_path, None, when=_present(codec_path)))
    cases.extend(_save_case(schema, "codec", None))
    return tuple(cases)


def _webm_cases(schema: NodeSchema) -> tuple[C, ...]:
    cases = []
    for codec, depth in (("av1", "10"), ("vp9", "8")):
        cases.append(
            C.build(
                "dinkster.save_video",
                when=P.value_equals("codec", "av1") if codec == "av1" else None,
                nodes={
                    "images": N.build("dinkster.video.window"),
                    "assemble": N.build(
                        "dinkster.video.assemble",
                        values={"bit_depth": depth, "color_space": "sRGB"},
                    ),
                    "crf_value": N.build("dinkster.float"),
                    "crf": N.build(
                        "dinkster.value.convert", values={"target": "int", "force_lossy": True}
                    ),
                    "save_target": N.build(
                        "dinkster.set_save_target_prefix",
                        values={
                            "target": {"mount": "comfy-output", "prefix": "ComfyUI"},
                        },
                    ),
                },
                inputs={
                    "images:images": M.copy("images"),
                    "assemble:fps": M.copy("fps"),
                    "crf_value:value": M.copy("crf"),
                    "codec": M.copy("codec"),
                    "container": M.constant("webm"),
                    "save_target:prefix": M.copy("filename_prefix"),
                },
                links=(
                    L("images:images", "assemble:images"),
                    L("assemble:video", "video"),
                    L("crf_value:value", "crf:value"),
                    L("crf:int", "crf"),
                    L("save_target:save_target", "target"),
                ),
                outputs={"images:images": schema.outputs[0].id},
            )
        )
    return tuple(cases)


def build_registry(comfy_root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(comfy_root), *args], text=True).strip()

    if git("rev-parse", "HEAD") != BASELINE or git("status", "--porcelain"):
        raise RuntimeError(f"VIDEO reference must be clean and pinned to {BASELINE}")
    sys.path.insert(0, str(comfy_root))
    from comfy.cli_args import args

    args.cpu = True
    from comfy_extras import nodes_video

    schemas = {}
    for name in sorted(VIDEO_CLASSES):
        cls = getattr(nodes_video, "VideoSlice" if name == "Video Slice" else name)
        schema = translate_v3_schema(cls.GET_SCHEMA(), CompatTranslation())
        if name == "CreateVideo":
            # Mixed integer/string upstream choices use the canonical string combo codec.
            schema = replace(
                schema,
                inputs=tuple(
                    replace(spec, widget=ComboWidget(options=("auto", "8", "10")))
                    if spec.id == "bit_depth"
                    else spec
                    for spec in schema.inputs
                ),
            )
        schemas[name] = _save_source_schema(schema) if name == "SaveVideo" else schema

    records = []
    for name, schema in schemas.items():
        note = ""
        if name == "SaveVideo":
            cases, note = _save_cases(schema), SAVE_NOTE
        elif name == "SaveWEBM":
            cases = _webm_cases(schema)
            note = (
                "Deferred assembly then WebM saving preserves IMAGE passthrough through a "
                "zero-window frame helper. AV1 uses 10-bit, VP9 uses 8-bit. CRF is explicitly "
                "converted to integer with truncation toward zero. Native precise samples "
                "avoid ComfyUI's preliminary uint8 quantization. Native refuses AV1 alpha "
                "loss rather than silently dropping alpha. Hidden metadata is not mapped."
            )
        else:
            target, mapping, outputs = {
                "LoadVideo": (
                    "load_video_value",
                    {"video": "file"},
                    {"video": schema.outputs[0].id},
                ),
                "CreateVideo": (
                    "video.assemble",
                    {
                        key: key
                        for key in ("images", "fps", "audio", "bit_depth", "color_space", "codec")
                    },
                    {"video": schema.outputs[0].id},
                ),
                "ConcatenateVideo": (
                    "video.concatenate",
                    {"codec": "codec", "complete_audio": "complete_audio"},
                    {"video": schema.outputs[0].id},
                ),
                "GetVideoComponents": (
                    "video.disassemble",
                    {"video": "video"},
                    dict(
                        zip(
                            ("images", "audio", "fps", "bit_depth", "color_space"),
                            (out.id for out in schema.outputs),
                            strict=False,
                        )
                    ),
                ),
                "Video Slice": (
                    "video.trim",
                    {key: key for key in ("video", "start_time", "duration", "strict_duration")},
                    {"video": schema.outputs[0].id},
                ),
                "VideoTrim": (
                    "video.trim",
                    {"video": "video", "video_edit": "trim", "strict_duration": "strict_duration"},
                    {"video": schema.outputs[0].id},
                ),
                "VideoCrop": (
                    "video.crop",
                    {"video": "video", "video_edit": "crop"},
                    {"video": schema.outputs[0].id},
                ),
            }[name]
            cases = (
                C.build(
                    "dinkster." + target,
                    inputs={key: M.copy(value) for key, value in mapping.items()},
                    outputs=outputs,
                ),
            )
            if name == "ConcatenateVideo":
                cases = (
                    C.build(
                        "dinkster.video.concatenate",
                        inputs={
                            "codec": M.copy("codec"),
                            "complete_audio": M.copy("complete_audio"),
                        },
                        input_families={
                            "videos": InputFamilyMapping.copy(
                                "videos", inputs={"value": M.copy("video")}
                            )
                        },
                        outputs=outputs,
                    ),
                )
                note = (
                    "Segments stay lazy until export. Compatible encoded inputs use bounded "
                    "packet copy; incompatible or component-backed inputs share one encoding. "
                    "Codec preference and complete soundtrack override are preserved."
                )
            if name == "CreateVideo":
                cases = (
                    replace(cases[0], when=_present("fps")),
                    C.build(
                        "dinkster.video.assemble",
                        inputs={
                            **{key: M.copy(value) for key, value in mapping.items()},
                            "fps": M.constant(30.0),
                        },
                        outputs=outputs,
                    ),
                )
            if name == "GetVideoComponents":
                cases = (
                    C.build(
                        "dinkster.video.disassemble",
                        nodes={
                            "depth_text": N.build(
                                "dinkster.value.convert",
                                values={
                                    "target": "string",
                                    "force_lossy": False,
                                },
                            ),
                            "depth": N.build("dinkster.string_to_combo"),
                            "color": N.build("dinkster.string_to_combo"),
                        },
                        inputs={"video": M.copy("video")},
                        links=(
                            L("bit_depth", "depth_text:value"),
                            L("depth_text:string", "depth:string"),
                            L("color_space", "color:string"),
                        ),
                        outputs={
                            **{key: outputs[key] for key in ("images", "audio", "fps")},
                            "depth:choice": outputs["bit_depth"],
                            "color:choice": outputs["color_space"],
                        },
                    ),
                )
            if name in ("Video Slice", "VideoTrim"):
                note = TRIM_NOTE
            elif name == "VideoCrop":
                note = (
                    "VIDEO_EDIT is copied verbatim; native crops compose lazily with "
                    "ComfyUI rectangle normalization. Preview publication is not reproduced."
                )
            elif name in ("CreateVideo", "GetVideoComponents"):
                note = (
                    "Deferred components preserve bit_depth and color_space. Native RGBA "
                    "is an alpha superset, not a ComfyUI golden. Native auto can inherit "
                    "IMAGE depth provenance; untagged SDR defaults to 8-bit."
                )
        rule = ReplacementRule(from_type=schema.node_type, cases=cases, note=note)
        evidence = ["test_video_alias_branches_execute"]
        if name == "SaveVideo":
            evidence += [
                "test_save_video_all_source_branches",
                "test_save_video_omitted_format_uses_source_default",
                "test_crf_link_uses_explicit_integer_conversion",
            ]
        elif name == "SaveWEBM":
            evidence.append("test_webm_encoding_and_passthrough")
        elif name in ("CreateVideo", "GetVideoComponents"):
            evidence.append("test_components_alias_depth_and_color_links")
        elif name in ("VideoTrim", "VideoCrop"):
            evidence.append("test_video_edit_widgets_are_lossless")
        records.append(
            {
                "id": f"comfy_alias:comfy-core/{name}",
                "mappingKind": "op",
                "carrier": cases[0].to,
                "source": {
                    "pack": "comfy-core",
                    "nodeClass": name,
                    "nodeType": schema.node_type,
                    "revision": BASELINE[:8],
                },
                "replacement": rule_to_wire(rule),
                "confidence": {
                    "tier": "exact" if name == "LoadVideo" else "parametric",
                    "evidence": [
                        "tests/test_video_alias_templates.py::" + test for test in evidence
                    ],
                },
            }
        )
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [schema_to_wire(s) for s in schemas.values()],
        "records": records,
    }


def merge_video_registry(registry: dict, video: dict) -> dict:
    for key, identify in (
        ("records", lambda item: item["source"]["nodeClass"]),
        ("sourceSchemas", lambda item: item["nodeType"].removeprefix("comfy.")),
    ):
        replacements = {identify(item): item for item in video[key]}
        result = [item for item in registry[key] if identify(item) not in replacements]
        registry[key] = [*result, *replacements.values()]
    return registry


if __name__ == "__main__":
    video = build_registry(Path(os.environ["COMFYUI_ROOT"]).resolve())
    if "--stdout" in sys.argv:
        print(json.dumps(video))
    else:
        registry = merge_video_registry(json.loads(OUT.read_text()), video)
        OUT.write_text(json.dumps(registry, indent=2) + "\n")
