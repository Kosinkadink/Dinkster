"""MoGe geometry model loading from granted checkpoint assets."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    MOGE_MODEL_TYPE,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)


class LoadMoGeModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadMoGeModel",
            display_name="Load MoGe Model",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "model_name",
                    TypeExpr.concrete(ASSET_TYPE),
                    widget=AssetWidget(
                        accept=("application/octet-stream",), kind="model/geometry-estimation"
                    ),
                ),
            ),
            outputs=(OutputSpec("MOGE_MODEL", TypeExpr.concrete(MOGE_MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls, *, model_name: object) -> Mapping[str, object]:
        if not isinstance(model_name, (AssetRef, LocalAssetRef)):
            raise TypeError("model_name must be an AssetRef or LocalAssetRef")
        path = model_name.local_path()
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        model = importlib.import_module("dinkster_inference.ldm.moge.model")
        state = utils.load_torch_file(str(path), safe_load=True)
        return cls.outputs(MOGE_MODEL=model.MoGeModel(state))
