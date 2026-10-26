"""ComfyUI-compatible image file nodes backed by mounted assets."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    AssetError,
    AssetWriter,
    InputSpec,
    MountSnapshotWriter,
    MountWidget,
    Node,
    NodeSchema,
    OutputSpec,
    StringWidget,
    TypeExpr,
    materialize_image,
    render_image_png,
)

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)


def _writer() -> AssetWriter:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
    return AssetWriter(MountSnapshotWriter(snapshot))


class SaveImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SaveImage",
            display_name="Save Image",
            category="image",
            description="Saves the input images to your ComfyUI output directory.",
            inputs=(
                InputSpec("images", _IMAGE),
                InputSpec(
                    "filename_prefix",
                    _STRING,
                    default="ComfyUI",
                    widget=StringWidget(multiline=False),
                ),
                InputSpec(
                    "mount",
                    _STRING,
                    required=False,
                    default=None,
                    widget=MountWidget(),
                    adjacent_to="filename_prefix",
                ),
            ),
            outputs=(OutputSpec("images", _IMAGE),),
            idempotent=False,
            output_node=True,
            search_terms=(
                "save",
                "save image",
                "export image",
                "output image",
                "write image",
                "download",
            ),
        )

    @classmethod
    def execute(
        cls, *, images: Any, filename_prefix: str = "ComfyUI", mount: str | None = None
    ) -> Mapping[str, object]:
        batch = materialize_image(images)
        if batch.ndim != 4 or batch.shape[0] < 1:
            raise ValueError("images must be a nonempty [B,H,W,C] batch")
        writer = _writer()
        target = writer.output_target(
            filename_prefix,
            mount=mount,
            image_width=int(batch.shape[2]),
            image_height=int(batch.shape[1]),
        )
        for index in range(batch.shape[0]):
            writer.save_bytes(
                target,
                render_image_png(batch[index : index + 1]),
                suffix=".png",
                media_type="image/png",
            )
        return cls.outputs(images=images)


class PreviewImage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="PreviewImage",
            display_name="Preview Image",
            category="image",
            description="Preview the images without saving them to the ComfyUI output directory.",
            inputs=(InputSpec("images", _IMAGE),),
            outputs=(OutputSpec("images", _IMAGE, preview=True),),
            output_node=True,
            search_terms=("preview", "preview image", "show image", "view image", "display image"),
        )

    @classmethod
    def execute(cls, *, images: Any) -> Mapping[str, object]:
        materialize_image(images)
        return cls.outputs(images=images)


__all__ = ["PreviewImage", "SaveImage"]
