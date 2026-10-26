"""Public custom-sampling fixture values."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    LATENT_TYPE,
    MODEL_TYPE,
    SAMPLER_TYPE,
    SIGMAS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)


class CustomSamplingSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="custom-sampling-fixture.source",
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("sampler", TypeExpr.concrete(SAMPLER_TYPE)),
                OutputSpec("sigmas", TypeExpr.concrete(SIGMAS_TYPE)),
                OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )


class CustomSamplingConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="custom-sampling-fixture.consumer",
            inputs=(
                InputSpec("output", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("denoised_output", TypeExpr.concrete(LATENT_TYPE)),
            ),
            outputs=(OutputSpec("sampled", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(
        cls, *, output: Mapping[str, object], denoised_output: Mapping[str, object]
    ) -> Mapping[str, object]:
        return cls.outputs(sampled=output["samples"] is not denoised_output["samples"])


NODES = (CustomSamplingSource, CustomSamplingConsumer)
