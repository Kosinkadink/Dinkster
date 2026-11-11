"""Native SenseNova sampling options."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    with_identity_replacement,
)


class SenseNovaSamplingOptions(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        schema = NodeSchema(
            node_type="dinkster.misc.models.sense_nova_sampling_options",
            display_name="SenseNova Sampling Options",
            category="model/patch/sensenova",
            description="Set the SenseNova flow shift.",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec(
                    "shift",
                    TypeExpr.concrete(CORE_FLOAT),
                    default=3.0,
                    widget=NumberWidget(step=0.01),
                ),
            ),
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )
        return with_identity_replacement(schema, "SenseNovaSamplingOptions")

    @classmethod
    def execute(cls, *, model: Any, shift: float) -> Mapping[str, object]:
        initialize_inference_process()
        sampling = importlib.import_module("dinkster_inference.ldm.sensenova.sampling")
        patched = model.clone()
        model_sampling = sampling.SenseNovaModelSampling(patched.model.model_config)
        model_sampling.set_parameters(shift=shift)
        patched.add_object_patch("model_sampling", model_sampling)
        return cls.outputs(MODEL=patched)
