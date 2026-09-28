"""Asset-backed gaussian splat loading and PLY export."""

from __future__ import annotations

import io
import os
from collections.abc import Mapping
from typing import BinaryIO, cast

from dinkster_api.v1 import (
    GIBIBYTE,
    SAVE_TARGET_TYPE,
    AssetError,
    AssetRef,
    AssetWidget,
    AssetWriter,
    InputSpec,
    MountSnapshotWriter,
    Node,
    NodeSchema,
    OutputSpec,
    SaveTargetWidget,
    SourceFilenameSpec,
    TypeExpr,
    decode_splat_file,
    render_splat_ply,
)

SPLAT_TYPE = "dinkster.splat"
SPLAT_MEDIA_TYPE = "model/ply"

SPLAT = TypeExpr.concrete(SPLAT_TYPE)
SPLAT_ASSET = TypeExpr.asset_of(SPLAT)
SAVE_TARGET = TypeExpr.concrete(SAVE_TARGET_TYPE)

MAX_SPLAT_PLY_BYTES = GIBIBYTE


def _mount_writer() -> AssetWriter:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError(
            "saving requires filesystem mounts, but this process has no "
            "DINKSTER_MOUNTS_SNAPSHOT configured"
        )
    return AssetWriter(MountSnapshotWriter(snapshot))


class LoadGaussianSplat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_gaussian_splat",
            display_name="Load Gaussian Splat",
            category="3d",
            inputs=(
                InputSpec(
                    "splat",
                    SPLAT_ASSET,
                    widget=AssetWidget(
                        accept=(SPLAT_MEDIA_TYPE,),
                        kind="media/model3d",
                        allow_upload=True,
                    ),
                    source_filename=SourceFilenameSpec("media/model3d", "input"),
                ),
            ),
            outputs=(OutputSpec("splat", SPLAT, preview=True),),
            search_terms=("gaussian splat loader", "PLY", "3DGS", "point cloud"),
        )

    @classmethod
    def execute(cls, *, splat: AssetRef) -> Mapping[str, object]:
        return cls.outputs(splat=decode_splat_file(splat))


class SaveGaussianSplat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.save_gaussian_splat",
            display_name="Save Gaussian Splat",
            category="3d",
            inputs=(
                InputSpec("splat", SPLAT),
                InputSpec(
                    "target",
                    SAVE_TARGET,
                    required=False,
                    default={"mount": "comfy-output", "prefix": "3d/ComfyUI"},
                    widget=SaveTargetWidget(),
                ),
            ),
            outputs=(OutputSpec("splat", SPLAT_ASSET, preview=True),),
            idempotent=False,
            output_node=True,
            search_terms=("gaussian splat saver", "PLY", "3DGS", "splat export"),
        )

    @classmethod
    def execute(cls, *, splat: object, target: object = None) -> Mapping[str, object]:
        data = render_splat_ply(splat, limit=MAX_SPLAT_PLY_BYTES)
        destination = target or {"mount": "comfy-output", "prefix": "3d/ComfyUI"}
        ref = _mount_writer().save_stream(
            destination,
            cast(BinaryIO, io.BytesIO(data)),
            suffix=".ply",
            media_type=SPLAT_MEDIA_TYPE,
            limit=MAX_SPLAT_PLY_BYTES,
        )
        return cls.outputs(splat=ref)


SPLAT_NODES: tuple[type[Node], ...] = (LoadGaussianSplat, SaveGaussianSplat)

__all__ = [
    "MAX_SPLAT_PLY_BYTES",
    "SPLAT_ASSET",
    "SPLAT_MEDIA_TYPE",
    "SPLAT_NODES",
    "SPLAT_TYPE",
    "LoadGaussianSplat",
    "SaveGaussianSplat",
]
