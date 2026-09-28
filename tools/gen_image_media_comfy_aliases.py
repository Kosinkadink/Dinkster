"""Build image-side records for the media I/O ComfyUI alias registry."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from dinkster_compat_comfy import CompatTranslation, translate_node, translate_v3_schema
from dinkster_schema import (
    MappingSource,
    ReplacementCase,
    ReplacementLink,
    ReplacementNode,
    ReplacementRule,
    ValueTransform,
)
from dinkster_schema.model import NodeSchema
from dinkster_schema.replace import rule_to_wire
from dinkster_schema.wire import schema_to_wire

COMFY_BASELINE = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent


def _git(comfy_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(comfy_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_schema(node_class: type[Any]) -> NodeSchema:
    get_schema = getattr(node_class, "GET_SCHEMA", None)
    if get_schema is not None:
        return translate_v3_schema(get_schema(), CompatTranslation())
    return translate_node(node_class.__name__, node_class, CompatTranslation()).schema()


def _record(
    node_class: str,
    carrier: str,
    rule: ReplacementRule,
    evidence: list[str],
) -> dict[str, object]:
    return {
        "id": f"comfy_alias:comfy-core/{node_class}",
        "mappingKind": "op",
        "carrier": carrier,
        "source": {
            "pack": "comfy-core",
            "nodeClass": node_class,
            "nodeType": rule.from_type,
            "revision": "b78cec87",
        },
        "replacement": rule_to_wire(rule),
        "confidence": {"tier": "parametric", "evidence": evidence},
    }


def _target_with_prefix(
    target: str,
    source_output: str,
    inputs: dict[str, MappingSource],
    *,
    target_prefix: str,
    note: str,
) -> ReplacementRule:
    return ReplacementRule(
        from_type=f"comfy.{target_prefix}",
        cases=(
            ReplacementCase.build(
                target,
                nodes={
                    "save_target": ReplacementNode.build(
                        "dinkster.set_save_target_prefix",
                        values={"target": {"mount": "comfy-output", "prefix": "ComfyUI"}},
                    )
                },
                inputs={
                    **inputs,
                    "save_target:prefix": MappingSource.copy("filename_prefix"),
                },
                links=(ReplacementLink("save_target:save_target", "target"),),
                outputs={"images": source_output},
            ),
        ),
        note=note,
    )


def build_registry(comfy_root: Path) -> dict[str, object]:
    if _git(comfy_root, "rev-parse", "HEAD") != COMFY_BASELINE:
        raise RuntimeError(f"ComfyUI must be pinned to {COMFY_BASELINE}")
    if _git(comfy_root, "status", "--porcelain"):
        raise RuntimeError("ComfyUI checkout must be clean")
    sys.path.insert(0, str(comfy_root))

    from comfy_extras.nodes_images import (  # pyright: ignore[reportMissingImports]
        SaveAnimatedPNG,
        SaveAnimatedWEBP,
    )
    from nodes import (  # pyright: ignore[reportMissingImports]
        LoadImage,
        LoadImageMask,
        LoadImageOutput,
        PreviewImage,
        SaveImage,
    )

    source_classes = (
        LoadImage,
        LoadImageMask,
        LoadImageOutput,
        SaveImage,
        PreviewImage,
        SaveAnimatedPNG,
        SaveAnimatedWEBP,
    )
    source_schemas = [_source_schema(node_class) for node_class in source_classes]
    schemas = {schema.node_type: schema for schema in source_schemas}

    def output(node_type: str, index: int = 0) -> str:
        return schemas[node_type].outputs[index].id

    load_evidence = [
        "tests/test_image_io.py::test_load_image_applies_orientation_alpha_polarity_and_opaque_fallback"
    ]
    mask_evidence = [
        "tests/test_image_io.py::test_load_mask_channels_and_polarity_match_core_contract"
    ]
    save_evidence = [
        "tests/test_image_io.py::test_save_image_writes_one_ordered_asset_per_batch_element",
        "tests/test_image_io.py::test_save_image_matches_core_uint8_truncation",
    ]
    animation_evidence = [
        "tests/test_image_io.py::test_save_animated_image_preserves_frame_order_and_timing"
    ]
    records = [
        _record(
            "LoadImage",
            "dinkster.load_image",
            ReplacementRule(
                from_type="comfy.LoadImage",
                cases=(
                    ReplacementCase.build(
                        "dinkster.load_image",
                        inputs={"image": MappingSource.copy("image")},
                        outputs={
                            "image": output("comfy.LoadImage", 0),
                            "mask": output("comfy.LoadImage", 1),
                        },
                    ),
                ),
                note=(
                    "Animated and multipage assets produce an ordered image and mask batch. "
                    "Pillow preserves exact GIF palette/alpha samples and all same-size TIFF "
                    "pages; the pinned source's PyAV path can alter GIF samples, reject tiny "
                    "GIFs, or return only the first TIFF page."
                ),
            ),
            load_evidence,
        ),
        _record(
            "LoadImageMask",
            "dinkster.load_mask",
            ReplacementRule(
                from_type="comfy.LoadImageMask",
                cases=(
                    ReplacementCase.build(
                        "dinkster.load_mask",
                        inputs={
                            "mask": MappingSource.copy("image"),
                            "channel": MappingSource.copy("channel"),
                            "mask_polarity": MappingSource.from_value(
                                "channel",
                                ValueTransform.enum_rename(
                                    {
                                        "alpha": "transparency",
                                        "red": "coverage",
                                        "green": "coverage",
                                        "blue": "coverage",
                                    }
                                ),
                            ),
                        },
                        outputs={"mask": output("comfy.LoadImageMask")},
                    ),
                ),
                note="Animated and multipage assets are refused by the native still-mask loader.",
            ),
            mask_evidence,
        ),
        _record(
            "LoadImageOutput",
            "dinkster.load_image_output",
            ReplacementRule(
                from_type="comfy.LoadImageOutput",
                cases=(
                    ReplacementCase.build(
                        "dinkster.load_image_output",
                        inputs={"image": MappingSource.copy("image")},
                        outputs={
                            "image": output("comfy.LoadImageOutput", 0),
                            "mask": output("comfy.LoadImageOutput", 1),
                        },
                    ),
                ),
                note=(
                    "The source output-folder picker becomes a typed output-category asset picker."
                ),
            ),
            load_evidence,
        ),
        _record(
            "SaveImage",
            "dinkster.save_image",
            _target_with_prefix(
                "dinkster.save_image",
                output("comfy.SaveImage"),
                {
                    "images": MappingSource.copy("images"),
                    "format": MappingSource.constant("png"),
                    "compression": MappingSource.constant(4),
                },
                target_prefix="SaveImage",
                note=(
                    "Filename prefixes are literal mounted relative paths and use native "
                    "collision suffixes; ComfyUI substitutions and hidden prompt metadata "
                    "are not inferred during document migration."
                ),
            ),
            save_evidence,
        ),
        _record(
            "PreviewImage",
            "dinkster.preview_image",
            ReplacementRule(
                from_type="comfy.PreviewImage",
                cases=(
                    ReplacementCase.build(
                        "dinkster.preview_image",
                        inputs={"images": MappingSource.copy("images")},
                        outputs={"images": output("comfy.PreviewImage")},
                    ),
                ),
                note="Preview renditions replace ComfyUI temporary-file publication.",
            ),
            ["tests/test_image_io.py::test_image_io_schemas_use_typed_assets_and_mounted_targets"],
        ),
        _record(
            "SaveAnimatedPNG",
            "dinkster.save_animated_image",
            _target_with_prefix(
                "dinkster.save_animated_image",
                output("comfy.SaveAnimatedPNG"),
                {
                    "images": MappingSource.copy("images"),
                    "format": MappingSource.constant("png"),
                    "fps": MappingSource.copy("fps"),
                    "compression": MappingSource.copy("compress_level"),
                    "loop": MappingSource.constant(0),
                },
                target_prefix="SaveAnimatedPNG",
                note=(
                    "Frame timing uses the source millisecond floor; hidden prompt metadata "
                    "is not inferred."
                ),
            ),
            animation_evidence,
        ),
        _record(
            "SaveAnimatedWEBP",
            "dinkster.save_animated_image",
            _target_with_prefix(
                "dinkster.save_animated_image",
                output("comfy.SaveAnimatedWEBP"),
                {
                    "images": MappingSource.copy("images"),
                    "format": MappingSource.constant("webp"),
                    "fps": MappingSource.copy("fps"),
                    "lossless": MappingSource.copy("lossless"),
                    "quality": MappingSource.copy("quality"),
                    "method": MappingSource.copy("method"),
                    "loop": MappingSource.constant(0),
                },
                target_prefix="SaveAnimatedWEBP",
                note=(
                    "Frame timing uses the source millisecond floor; hidden prompt metadata "
                    "is not inferred."
                ),
            ),
            animation_evidence,
        ),
    ]
    return {
        "format": "dinkster-comfy-alias/1",
        "sourceSchemas": [schema_to_wire(schema) for schema in source_schemas],
        "records": records,
    }
