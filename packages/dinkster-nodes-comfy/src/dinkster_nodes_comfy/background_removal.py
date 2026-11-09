"""Native background-removal loading and foreground-mask generation."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    BACKGROUND_REMOVAL_TYPE,
    IMAGE_TYPE,
    MASK_TYPE,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)


class LoadBackgroundRemovalModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadBackgroundRemovalModel",
            display_name="Load Background Removal Model",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "bg_removal_name",
                    TypeExpr.concrete(ASSET_TYPE),
                    widget=AssetWidget(kind="model/background-removal"),
                    doc="The model used to remove backgrounds from images",
                ),
            ),
            outputs=(OutputSpec("bg_model", TypeExpr.concrete(BACKGROUND_REMOVAL_TYPE)),),
        )

    @classmethod
    def execute(cls, *, bg_removal_name: object) -> Mapping[str, object]:
        if not isinstance(bg_removal_name, (AssetRef, LocalAssetRef)):
            raise TypeError("bg_removal_name must be an AssetRef or LocalAssetRef")
        path = bg_removal_name.local_path()
        initialize_inference_process()
        loader = importlib.import_module("dinkster_inference.bg_removal_model")
        model = loader.load(str(path))
        if model is None:
            raise RuntimeError(
                "ERROR: background model file is invalid and does not contain a valid "
                "background removal model."
            )
        return cls.outputs(bg_model=model)


class RemoveBackground(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="RemoveBackground",
            display_name="Remove Background",
            category="image/background removal",
            description=(
                "Generates a foreground mask to remove the background from an image "
                "using a background removal model."
            ),
            inputs=(
                InputSpec(
                    "bg_removal_model",
                    TypeExpr.concrete(BACKGROUND_REMOVAL_TYPE),
                    doc="Background removal model used to generate the mask",
                ),
                InputSpec(
                    "image",
                    TypeExpr.concrete(IMAGE_TYPE),
                    doc="Input image to remove the background from",
                ),
            ),
            outputs=(
                OutputSpec("mask", TypeExpr.concrete(MASK_TYPE), doc="Generated foreground mask"),
            ),
        )

    @classmethod
    def execute(cls, *, bg_removal_model: Any, image: Any) -> Mapping[str, object]:
        mask = bg_removal_model.encode_image(materialize_image(image))
        return cls.outputs(mask=mask)


BACKGROUND_REMOVAL_NODES = (LoadBackgroundRemovalModel, RemoveBackground)
