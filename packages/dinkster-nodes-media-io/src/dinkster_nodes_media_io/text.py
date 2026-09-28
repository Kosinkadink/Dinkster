"""Bounded UTF-8 text saving to mounted targets.

Newline policy: content bytes are written verbatim as UTF-8 with "\\n"
newlines preserved and no platform translation, so the saved bytes are
identical on every OS. The pinned ComfyUI SaveText writes through text
mode (platform newline translation); on POSIX the bytes match exactly.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Mapping
from typing import BinaryIO, cast

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_STRING,
    MEBIBYTE,
    SAVE_TARGET_TYPE,
    AssetError,
    AssetRef,
    AssetWriter,
    ComboWidget,
    InputSpec,
    MountSnapshotWriter,
    Node,
    NodeSchema,
    OutputSpec,
    SaveTargetWidget,
    TypeExpr,
)

STRING = TypeExpr.concrete(CORE_STRING)
TEXT_ASSET = TypeExpr.asset_of(STRING)
TEXT_ASSET_LIST = TypeExpr.list_of(TEXT_ASSET)
COMBO = TypeExpr.concrete(CORE_COMBO)
SAVE_TARGET = TypeExpr.concrete(SAVE_TARGET_TYPE)

MAX_TEXT_BYTES = 64 * MEBIBYTE

DEFAULT_TEXT_TARGET = {"mount": "comfy-output", "prefix": "text/ComfyUI"}

# Combo values are untrusted; anything used as a suffix is resolved through
# this fixed mapping and unknown formats fail closed.
_TEXT_FORMATS: dict[str, tuple[str, str]] = {
    "txt": (".txt", "text/plain"),
    "csv": (".csv", "text/csv"),
    "md": (".md", "text/markdown"),
    "json": (".json", "application/json"),
}


def _mount_writer() -> AssetWriter:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError(
            "saving requires filesystem mounts, but this process has no "
            "DINKSTER_MOUNTS_SNAPSHOT configured"
        )
    return AssetWriter(MountSnapshotWriter(snapshot))


def render_text(text: str, format: str) -> bytes:
    """The exact bytes a save produces: UTF-8, no BOM, no newline
    translation. The json format pretty-prints valid JSON exactly like the
    pinned ComfyUI SaveText (indent=2, ensure_ascii=False) and falls back
    to the raw text when the content is not JSON."""
    if format not in _TEXT_FORMATS:
        raise ValueError(f"format must be one of {tuple(_TEXT_FORMATS)}, got {format!r}")
    rendered = text
    if format == "json":
        try:
            rendered = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            rendered = text
    data = rendered.encode("utf-8")
    if len(data) > MAX_TEXT_BYTES:
        raise ValueError(
            f"rendered text is {len(data)} bytes, above the {MAX_TEXT_BYTES} byte save limit"
        )
    return data


class SaveText(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.save_text",
            display_name="Save Text",
            category="text",
            description="Save text content to a mounted target as UTF-8.",
            inputs=(
                InputSpec("text", STRING, on_absent="fail"),
                InputSpec(
                    "format",
                    COMBO,
                    required=False,
                    default="txt",
                    widget=ComboWidget(options=tuple(_TEXT_FORMATS)),
                ),
                InputSpec(
                    "target",
                    SAVE_TARGET,
                    required=False,
                    default=dict(DEFAULT_TEXT_TARGET),
                    widget=SaveTargetWidget(),
                ),
            ),
            outputs=(OutputSpec("texts", TEXT_ASSET_LIST, preview=True),),
            idempotent=False,
            output_node=True,
            search_terms=("save text", "write text", "export text", "txt"),
        )

    @classmethod
    def execute(
        cls, *, text: object, format: str = "txt", target: object = None
    ) -> Mapping[str, object]:
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text).__name__}")
        data = render_text(text, format)
        suffix, media_type = _TEXT_FORMATS[format]
        ref: AssetRef = _mount_writer().save_stream(
            target or DEFAULT_TEXT_TARGET,
            cast(BinaryIO, io.BytesIO(data)),
            suffix=suffix,
            media_type=media_type,
            limit=MAX_TEXT_BYTES,
        )
        return cls.outputs(texts=[ref])


TEXT_IO_NODES: tuple[type[Node], ...] = (SaveText,)

__all__ = [
    "DEFAULT_TEXT_TARGET",
    "MAX_TEXT_BYTES",
    "TEXT_IO_NODES",
    "SaveText",
    "render_text",
]
