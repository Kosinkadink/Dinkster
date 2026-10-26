"""Public custom-sampling fixture values."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    GUIDER_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    NOISE_TYPE,
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
                OutputSpec("noise", TypeExpr.concrete(NOISE_TYPE)),
                OutputSpec("guider", TypeExpr.concrete(GUIDER_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        import torch
        from dinkster_inference import samplers
        from dinkster_inference.model_patcher import ModelPatcher
        from test_custom_sampling_torch import _RealModel

        patcher = ModelPatcher(_RealModel(), torch.device("cpu"), torch.device("cpu"))
        patcher.model.current_patcher = patcher

        class Noise:
            seed = 41

            def generate_noise(self, latent: Mapping[str, object]) -> object:
                samples = latent["samples"]
                assert isinstance(samples, torch.Tensor)
                return torch.tensor([[[[2.0, -3.0]]], [[[5.0, 7.0]]]])

        guider = samplers.CFGGuider(patcher)
        conditioning = [[torch.tensor([[1.0, -2.0]]), {}]]
        guider.set_conds(conditioning, conditioning)
        guider.set_cfg(1.0)
        return cls.outputs(
            model=patcher,
            positive=conditioning,
            negative=conditioning,
            sampler=samplers.ksampler("euler"),
            sigmas=torch.tensor([1.0, 0.0]),
            latent={
                "samples": torch.tensor([[[[2.0, -4.0]]], [[[8.0, 1.0]]]]),
                "batch_index": [6, 2],
                "noise_mask": torch.tensor([[[[1.0, 0.0]]]]),
                "origin": "fixture",
            },
            noise=Noise(),
            guider=guider,
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
        import torch

        samples = output["samples"]
        denoised = denoised_output["samples"]
        assert isinstance(samples, torch.Tensor)
        assert isinstance(denoised, torch.Tensor)
        assert output["batch_index"] == [6, 2]
        assert output["origin"] == "fixture"
        return cls.outputs(sampled=not torch.equal(samples, denoised))


NODES = (CustomSamplingSource, CustomSamplingConsumer)
