"""Asset-backed 3D model loading and GLB pass-through saving."""

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
    decode_model3d_file,
    encode_model3d,
)

MODEL3D_TYPE = "dinkster.model3d"
MODEL3D_MEDIA_TYPE = "model/gltf-binary"

MODEL3D = TypeExpr.concrete(MODEL3D_TYPE)
MODEL3D_ASSET = TypeExpr.asset_of(MODEL3D)
SAVE_TARGET = TypeExpr.concrete(SAVE_TARGET_TYPE)

MAX_ENCODED_MODEL3D_BYTES = GIBIBYTE


def _mount_writer() -> AssetWriter:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError(
            "saving requires filesystem mounts, but this process has no "
            "DINKSTER_MOUNTS_SNAPSHOT configured"
        )
    return AssetWriter(MountSnapshotWriter(snapshot))


class LoadModel3D(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_model3d",
            display_name="Load 3D Model",
            category="3d",
            inputs=(
                InputSpec(
                    "model",
                    MODEL3D_ASSET,
                    widget=AssetWidget(
                        accept=(MODEL3D_MEDIA_TYPE,),
                        kind="media/model3d",
                        allow_upload=True,
                    ),
                    source_filename=SourceFilenameSpec("media/model3d", "input"),
                ),
            ),
            outputs=(OutputSpec("model", MODEL3D, preview=True),),
            search_terms=("3D model loader", "GLB", "glTF", "mesh"),
        )

    @classmethod
    def execute(cls, *, model: AssetRef) -> Mapping[str, object]:
        return cls.outputs(model=decode_model3d_file(model))


class SaveModel3D(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.save_model3d",
            display_name="Save 3D Model",
            category="3d",
            inputs=(
                InputSpec("model", MODEL3D),
                InputSpec(
                    "target",
                    SAVE_TARGET,
                    required=False,
                    default={"mount": "comfy-output", "prefix": "3d/ComfyUI"},
                    widget=SaveTargetWidget(),
                ),
            ),
            outputs=(OutputSpec("model", MODEL3D_ASSET, preview=True),),
            idempotent=False,
            output_node=True,
            search_terms=("3D model saver", "GLB", "glTF", "mesh export"),
        )

    @classmethod
    def execute(cls, *, model: object, target: object = None) -> Mapping[str, object]:
        data = encode_model3d(model)
        if len(data) > MAX_ENCODED_MODEL3D_BYTES:
            raise ValueError(
                f"encoded 3D model exceeds the {MAX_ENCODED_MODEL3D_BYTES}-byte output limit"
            )
        destination = target or {"mount": "comfy-output", "prefix": "3d/ComfyUI"}
        ref = _mount_writer().save_stream(
            destination,
            cast(BinaryIO, io.BytesIO(data)),
            suffix=".glb",
            media_type=MODEL3D_MEDIA_TYPE,
            limit=MAX_ENCODED_MODEL3D_BYTES,
        )
        return cls.outputs(model=ref)


class PreviewModel3D(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.preview_model3d",
            display_name="Preview 3D",
            category="3d",
            inputs=(InputSpec("model", MODEL3D),),
            outputs=(OutputSpec("model", MODEL3D, preview=True),),
            output_node=True,
            search_terms=("3D model preview", "GLB", "glTF", "mesh viewer"),
        )

    @classmethod
    def execute(cls, *, model: object) -> Mapping[str, object]:
        encode_model3d(model)
        return cls.outputs(model=model)


MODEL3D_NODES: tuple[type[Node], ...] = (LoadModel3D, SaveModel3D, PreviewModel3D)

__all__ = [
    "MAX_ENCODED_MODEL3D_BYTES",
    "MODEL3D_ASSET",
    "MODEL3D_MEDIA_TYPE",
    "MODEL3D_NODES",
    "MODEL3D_TYPE",
    "LoadModel3D",
    "PreviewModel3D",
    "SaveModel3D",
]
