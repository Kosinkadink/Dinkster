"""Layer maintained current MiniMax H3 aliases onto the compatibility registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dinkster_schema import (
    InputFamilyMapping,
    MappingSource,
    ReplacementCase,
    ReplacementRule,
)
from dinkster_schema.replace import rule_to_wire

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "packages/dinkster-compat-comfy/src/dinkster_compat_comfy/core_schemas.json"
OUTPUT = ROOT / "packages/dinkster-compat-comfy/comfy-aliases.json"
COMFYUI_REVISION = "b5cc8830279eae909a59de030af1e50761c36751"
SOURCE_SHA256 = "e985a4262205a218f78c9653170d208e85cab9f960a4c53ca2353ca613432a61"
SOURCE_NODE_TYPES = (
    "comfy.CLIPLoader",
    "comfy.MiniMaxH3ImageToVideo",
    "comfy.MiniMaxH3ReferenceToVideo",
    "comfy.MiniMaxH3AddGuide",
    "comfy.ResolutionSelector",
)
EVIDENCE = [
    "tests/test_native_h3_comfy_aliases.py::test_native_h3_aliases_are_canonical_and_pinned",
    "tests/test_native_h3_comfy_aliases.py::test_native_h3_aliases_preserve_source_inputs_and_outputs",
]


def _record(node_class: str, carrier: str, rule: ReplacementRule) -> dict[str, object]:
    return {
        "id": f"comfy_alias:comfy-core/{node_class}",
        "mappingKind": "op",
        "carrier": carrier,
        "source": {
            "pack": "comfy-core",
            "nodeClass": node_class,
            "nodeType": rule.from_type,
            "revision": COMFYUI_REVISION,
        },
        "replacement": rule_to_wire(rule),
        "confidence": {"tier": "parametric", "evidence": EVIDENCE},
    }


def add_current_h3_aliases(registry: dict[str, Any]) -> dict[str, Any]:
    source = SOURCE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(
            f"pinned source schema hash mismatch: expected {SOURCE_SHA256}, got {digest}"
        )
    snapshot = json.loads(source)
    schemas = {
        schema["nodeType"]: schema
        for schema in snapshot["schemas"].values()
        if schema["nodeType"] in SOURCE_NODE_TYPES
    }
    missing = sorted(set(SOURCE_NODE_TYPES) - set(schemas))
    if missing:
        raise RuntimeError(f"missing pinned source schemas: {', '.join(missing)}")
    clip_schema = schemas["comfy.CLIPLoader"]
    schemas["comfy.CLIPLoader"] = {
        **clip_schema,
        "interface": [
            {
                **entry,
                "type": {"kind": "concrete", "types": ["dinkster.asset"]},
                "widget": {
                    "accept": ["application/octet-stream"],
                    "kind": "model/text-encoder",
                    "type": "ASSET",
                },
            }
            if entry["role"] == "input" and entry["id"] == "clip_name"
            else entry
            for entry in clip_schema["interface"]
        ],
    }

    image_inputs = {
        name: MappingSource.copy(name)
        for name in (
            "clip",
            "vae",
            "prompt",
            "width",
            "height",
            "length",
            "first_frame",
            "last_frame",
        )
    }
    reference_inputs = {
        name: MappingSource.copy(name)
        for name in (
            "clip",
            "vae",
            "audio_vae",
            "prompt",
            "width",
            "height",
            "length",
            "ref_image_size",
        )
    }
    reference_families = {
        name: InputFamilyMapping.copy(
            name,
            inputs={"value": MappingSource.copy("value")},
        )
        for name in ("ref_images", "ref_videos", "ref_video_audios", "ref_audios")
    }
    records = [
        _record(
            "CLIPLoader",
            "dinkster.load_clip",
            ReplacementRule(
                from_type="comfy.CLIPLoader",
                cases=(
                    ReplacementCase.build(
                        "dinkster.load_clip",
                        inputs={
                            "text_encoder": MappingSource.copy("clip_name"),
                            "type": MappingSource.copy("type"),
                            "device": MappingSource.copy("device"),
                        },
                        outputs={"clip": "clip"},
                    ),
                ),
            ),
        ),
        _record(
            "MiniMaxH3ImageToVideo",
            "dinkster.minimax_h3_image_to_video",
            ReplacementRule(
                from_type="comfy.MiniMaxH3ImageToVideo",
                cases=(
                    ReplacementCase.build(
                        "dinkster.minimax_h3_image_to_video",
                        inputs=image_inputs,
                        outputs={"positive": "positive", "latent": "LATENT"},
                    ),
                ),
            ),
        ),
        _record(
            "MiniMaxH3ReferenceToVideo",
            "dinkster.minimax_h3_reference_to_video",
            ReplacementRule(
                from_type="comfy.MiniMaxH3ReferenceToVideo",
                cases=(
                    ReplacementCase.build(
                        "dinkster.minimax_h3_reference_to_video",
                        inputs=reference_inputs,
                        input_families=reference_families,
                        outputs={"positive": "positive", "latent": "LATENT"},
                    ),
                ),
            ),
        ),
        _record(
            "MiniMaxH3AddGuide",
            "dinkster.minimax_h3_add_guide",
            ReplacementRule(
                from_type="comfy.MiniMaxH3AddGuide",
                cases=(
                    ReplacementCase.build(
                        "dinkster.minimax_h3_add_guide",
                        inputs={
                            name: MappingSource.copy(name)
                            for name in (
                                "positive",
                                "latent",
                                "frame_idx",
                                "vae",
                                "audio_vae",
                                "image",
                                "audio",
                            )
                        },
                        outputs={"positive": "positive"},
                    ),
                ),
            ),
        ),
        _record(
            "ResolutionSelector",
            "dinkster.resolution_selector",
            ReplacementRule(
                from_type="comfy.ResolutionSelector",
                cases=(
                    ReplacementCase.build(
                        "dinkster.resolution_selector",
                        inputs={
                            name: MappingSource.copy(name)
                            for name in ("aspect_ratio", "megapixels", "multiple")
                        },
                        outputs={"width": "width", "height": "height"},
                    ),
                ),
            ),
        ),
    ]
    source_types = set(SOURCE_NODE_TYPES)
    current_schemas = [
        schema for schema in registry["sourceSchemas"] if schema["nodeType"] not in source_types
    ]
    current_records = [
        record for record in registry["records"] if record["source"]["nodeType"] not in source_types
    ]
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [
            *current_schemas,
            *(schemas[node_type] for node_type in SOURCE_NODE_TYPES),
        ],
        "records": [*current_records, *records],
    }


def main() -> None:
    registry = json.loads(OUTPUT.read_text(encoding="utf-8"))
    OUTPUT.write_text(
        json.dumps(
            add_current_h3_aliases(registry),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
