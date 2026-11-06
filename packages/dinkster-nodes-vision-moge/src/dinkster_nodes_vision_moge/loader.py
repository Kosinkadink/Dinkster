"""Load monocular geometry models from host-granted checkpoint assets."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import replace

from dinkster_api.v1 import (
    ASSET_TYPE,
    MOGE_MODEL_TYPE,
    AssetRef,
    AssetWidget,
    InputSpec,
    LocalAssetRef,
    MappingSource,
    Node,
    NodeSchema,
    OutputSpec,
    ReplacementCase,
    ReplacementRule,
    TypeExpr,
    initialize_inference_process,
)


class LoadMoGeModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.vision.moge.load_mo_ge_model",
            display_name="Load MoGe Model",
            category="model/loaders",
            search_terms=("LoadMoGeModel",),
            inputs=(
                InputSpec(
                    "model_name",
                    TypeExpr.concrete(ASSET_TYPE),
                    widget=AssetWidget(
                        kind="model/geometry-estimation", accept=("application/octet-stream",)
                    ),
                ),
            ),
            outputs=(OutputSpec("MOGE_MODEL", TypeExpr.concrete(MOGE_MODEL_TYPE)),),
        )
        return replace(
            schema,
            replacements=(
                ReplacementRule(
                    from_type="LoadMoGeModel",
                    cases=(
                        ReplacementCase.build(
                            schema.node_type,
                            inputs={spec.id: MappingSource.copy(spec.id) for spec in schema.inputs},
                            outputs={spec.id: spec.id for spec in schema.outputs},
                        ),
                    ),
                ),
            ),
        )

    @classmethod
    def execute(cls, *, model_name: object) -> Mapping[str, object]:
        if not isinstance(model_name, (AssetRef, LocalAssetRef)):
            raise TypeError("model_name must be an AssetRef or LocalAssetRef")
        path = str(model_name.local_path())
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        model = importlib.import_module("dinkster_inference.ldm.moge.model")
        state = utils.load_torch_file(path, safe_load=True)
        return cls.outputs(MOGE_MODEL=model.MoGeModel(state))
