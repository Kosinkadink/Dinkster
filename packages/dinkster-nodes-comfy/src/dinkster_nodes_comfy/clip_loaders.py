"""Multi-checkpoint text encoders using admitted assets and embedding mounts."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    CLIP_TYPE,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    mount_roots_for_kind,
)


def _schema(node_type: str, display_name: str, recipe: str, count: int) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category="model/loaders",
        description="Recipes:\n" + recipe,
        inputs=tuple(
            InputSpec(
                f"clip_name{index}",
                TypeExpr.concrete(ASSET_TYPE),
                widget=AssetWidget(kind="model/text-encoder"),
            )
            for index in range(1, count + 1)
        ),
        outputs=(OutputSpec("CLIP", TypeExpr.concrete(CLIP_TYPE)),),
    )


def _load_clip(names: tuple[object, ...]) -> Any:
    paths: list[str] = []
    for index, asset in enumerate(names, 1):
        if not isinstance(asset, (AssetRef, LocalAssetRef)):
            raise TypeError(f"clip_name{index} must be an AssetRef or LocalAssetRef")
        paths.append(str(asset.local_path()))
    initialize_inference_process()
    sd = importlib.import_module("dinkster_inference.sd")
    return sd.load_clip(
        ckpt_paths=paths,
        embedding_directory=[str(root) for root in mount_roots_for_kind("model/embedding")],
    )


class TripleCLIPLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema("TripleCLIPLoader", "Load CLIP (Triple)", "sd3: clip-l, clip-g, t5", 3)

    @classmethod
    def execute(
        cls, *, clip_name1: object, clip_name2: object, clip_name3: object
    ) -> Mapping[str, object]:
        return cls.outputs(CLIP=_load_clip((clip_name1, clip_name2, clip_name3)))


class QuadrupleCLIPLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _schema(
            "QuadrupleCLIPLoader",
            "Load CLIP (Quadruple)",
            "hidream: long clip-l, long clip-g, t5xxl, llama_8b_3.1_instruct",
            4,
        )

    @classmethod
    def execute(
        cls, *, clip_name1: object, clip_name2: object, clip_name3: object, clip_name4: object
    ) -> Mapping[str, object]:
        return cls.outputs(CLIP=_load_clip((clip_name1, clip_name2, clip_name3, clip_name4)))
